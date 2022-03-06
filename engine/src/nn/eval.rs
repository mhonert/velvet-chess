/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

use crate::align::A32;
use crate::bitboard::{v_mirror, BitBoard, BitBoards};
use crate::colors::Color;
use crate::nn::{NeuralNetParams, FEATURES_PER_BUCKET, FP_PRECISION_BITS, HL_NODES};
use crate::pieces::{Q, R};
use crate::scores::sanitize_eval_score;
use std::cmp::max;
use std::sync::{Arc, Once};

static INIT_NN_PARAMS: Once = Once::new();
static mut NN_PARAMS: Option<Arc<NeuralNetParams>> = None;

#[derive(Clone)]
pub struct NeuralNetEval {
    params: Arc<NeuralNetParams>,

    hidden_nodes_wtm: A32<[i16; HL_NODES]>, // wtm - white to move
    hidden_nodes_btm: A32<[i16; HL_NODES]>, // btm - black to move

    wtm_bucket: u8,
    wtm_offset: usize,

    btm_bucket: u8,
    btm_offset: usize,

    move_id: usize,
    updates: Vec<(bool, usize, UpdateAction)>,

    undo: bool,
    fast_undo: bool,
}

#[derive(Clone)]
enum UpdateAction {
    Add(usize, i8),
    Remove(usize, i8),
    CheckRefresh,
}

impl NeuralNetEval {
    pub fn new() -> Box<Self> {
        INIT_NN_PARAMS.call_once(|| unsafe { NN_PARAMS = Some(NeuralNetParams::new()) });
        Box::new(NeuralNetEval {
            params: unsafe { NN_PARAMS.clone().unwrap() },

            hidden_nodes_wtm: A32([0; HL_NODES]),
            hidden_nodes_btm: A32([0; HL_NODES]),

            wtm_bucket: 0,
            wtm_offset: 0,

            btm_bucket: 0,
            btm_offset: 0,

            move_id: 0,
            updates: Vec::with_capacity(32),

            undo: false,
            fast_undo: false,
        })
    }

    pub fn init_pos(&mut self, bitboards: &BitBoards) {
        self.updates.clear();
        self.fast_undo = false;
        self.move_id = 0;
        self.hidden_nodes_wtm.0.copy_from_slice(&self.params.input_biases.0);
        self.hidden_nodes_btm.0.copy_from_slice(&self.params.input_biases.0);

        let (wtm_bucket, btm_bucket) = calc_bucket(bitboards);
        self.wtm_bucket = wtm_bucket;
        self.btm_bucket = btm_bucket;

        self.wtm_offset = wtm_bucket as usize * FEATURES_PER_BUCKET;
        self.btm_offset = btm_bucket as usize * FEATURES_PER_BUCKET;

        for piece in 1..=6 {
            for pos in BitBoard(bitboards.by_piece(piece)) {
                self.add_piece_now(pos as usize, piece);
            }

            for pos in BitBoard(bitboards.by_piece(-piece)) {
                self.add_piece_now(pos as usize, -piece);
            }
        }
    }

    pub fn start_move(&mut self) {
        self.move_id += 1;
        self.fast_undo = false;
        self.undo = false;
    }

    pub fn start_undo(&mut self) {
        self.undo = true;
        self.fast_undo = false;

        // Remove all updates for the latest move
        let mut move_id: Option<usize> = None;
        while let Some((was_undo, id, _)) = self.updates.last() {
            if *was_undo {
                return;
            }

            if let Some(move_id) = move_id {
                if move_id != *id {
                    return;
                }
            } else {
                move_id = Some(*id);
                self.fast_undo = true;
            }

            self.updates.pop();
        }
    }

    pub fn add_piece(&mut self, pos: usize, piece: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::Add(pos, piece)));
        }
    }

    fn add_piece_now(&mut self, pos: usize, piece: i8) {
        let base_index = ((piece.abs() as usize - 1) * 2) as usize * 64;

        let mut idx = self.wtm_offset + base_index + pos;
        if piece < 0 {
            idx += 64;
        }

        for (nodes, weights) in
            self.hidden_nodes_wtm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_NODES).nth(idx).unwrap())
        {
            *nodes += *weights;
        }

        let mut idx = self.btm_offset + base_index + v_mirror(pos);
        if piece > 0 {
            idx += 64;
        }

        for (nodes, weights) in
            self.hidden_nodes_btm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_NODES).nth(idx).unwrap())
        {
            *nodes += *weights;
        }
    }

    pub fn remove_piece(&mut self, pos: usize, piece: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::Remove(pos, piece)));
        }
    }

    fn remove_piece_now(&mut self, pos: usize, piece: i8) {
        let base_index = ((piece.abs() as usize - 1) * 2) as usize * 64;

        let mut idx = self.wtm_offset + base_index + pos;
        if piece < 0 {
            idx += 64;
        }

        for (nodes, weights) in
            self.hidden_nodes_wtm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_NODES).nth(idx).unwrap())
        {
            *nodes -= *weights;
        }

        let mut idx = self.btm_offset + base_index + v_mirror(pos);
        if piece > 0 {
            idx += 64;
        }

        for (nodes, weights) in
            self.hidden_nodes_btm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_NODES).nth(idx).unwrap())
        {
            *nodes -= *weights;
        }
    }

    pub fn check_refresh(&mut self) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::CheckRefresh));
        }
    }

    pub fn eval(&mut self, active_player: Color, half_move_clock: u8, bitboards: &BitBoards) -> i32 {
        self.apply_updates(bitboards);

        let output = if active_player.is_white() {
            (forward_pass::<HL_NODES>(&self.hidden_nodes_wtm.0, &self.params.output_weights.0)
                + self.params.output_bias) as i32
        } else {
            -(forward_pass::<HL_NODES>(&self.hidden_nodes_btm.0, &self.params.output_weights.0)
                + self.params.output_bias) as i32
        };

        let score = sanitize_eval_score(output * 2048 / (1 << FP_PRECISION_BITS));
        adjust_eval(score, half_move_clock)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards) {
        for i in 0..self.updates.len() {
            let (_, _, update) = unsafe { self.updates.get_unchecked(i) };
            match *update {
                UpdateAction::CheckRefresh => {
                    let (wtm_bucket, btm_bucket) = calc_bucket(bitboards);
                    if wtm_bucket != self.wtm_bucket || btm_bucket != self.btm_bucket {
                        self.init_pos(bitboards);
                        self.updates.clear();
                        return;
                    }
                }

                UpdateAction::Add(pos, piece) => {
                    self.add_piece_now(pos, piece);
                }

                UpdateAction::Remove(pos, piece) => {
                    self.remove_piece_now(pos, piece);
                }
            }
        }
        self.updates.clear();
    }
}

// Scale eval score towards 0 for decreasing number of remaining half moves till the 50-move (draw) rule triggers
fn adjust_eval(score: i32, half_move_clock: u8) -> i32 {
    let remaining_half_moves = max(0, 100 - half_move_clock as i32);
    if remaining_half_moves >= 95 {
        score
    } else {
        score * remaining_half_moves / 95
    }
}

fn calc_bucket(bitboards: &BitBoards) -> (u8, u8) {
    let mut wtm_bucket = 0;
    let mut btm_bucket = 0;

    if bitboards.by_piece(Q) == 0 && bitboards.by_piece(-Q) == 0 {
        if bitboards.by_piece(R) != 0 {
            wtm_bucket |= 0b10;
            btm_bucket |= 0b01;
        }

        if bitboards.by_piece(-R) != 0 {
            wtm_bucket |= 0b01;
            btm_bucket |= 0b10;
        }
    } else {
        wtm_bucket = 4;
        btm_bucket = 4;
    }

    (wtm_bucket, btm_bucket)
}

#[inline(always)]
fn forward_pass<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
    #[cfg(target_feature = "avx2")]
    {
        avx2::forward_pass::<N>(nodes, weights)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::forward_pass::<N>(nodes, weights)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::forward_pass::<N>(nodes, weights)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use crate::nn::eval::FP_PRECISION_BITS;
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;

    #[inline(always)]
    pub fn forward_pass<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        unsafe {
            let mut acc = _mm256_setzero_si256();
            let zero = _mm256_setzero_si256();

            for i in 0..(N / 16) {
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                let n_relu = _mm256_max_epi16(n, zero);
                let w = _mm256_load_si256(transmute(weights.as_ptr().add(i * 16)));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(n_relu, w));
            }

            // Final horizontal sum of the lanes for the accumulator
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(hi64, sum128);
            let hi32 = _mm_shuffle_epi32::<0b10110001>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            (_mm_cvtsi128_si32(sum32) >> FP_PRECISION_BITS) as i16
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2 {
    use crate::nn::eval::FP_PRECISION_BITS;
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;

    #[inline(always)]
    pub fn forward_pass<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        unsafe {
            let mut acc = _mm_setzero_si128();
            let zero = _mm_setzero_si128();

            for i in 0..(N / 8) {
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                let n_relu = _mm_max_epi16(n, zero);
                let w = _mm_load_si128(transmute(weights.as_ptr().add(i * 8)));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(n_relu, w));
            }

            // Final horizontal sum of the lanes for the accumulator
            let hi64 = _mm_shuffle_epi32::<0b01001110>(acc);
            let sum64 = _mm_add_epi32(hi64, acc);
            let hi32 = _mm_shufflelo_epi16::<0b01001110>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            (_mm_cvtsi128_si32(sum32) >> FP_PRECISION_BITS) as i16
        }
    }
}

#[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
mod fallback {
    use crate::nn::eval::FP_PRECISION_BITS;

    #[inline(always)]
    pub fn forward_pass<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        (nodes.iter().zip(weights).map(|(&n, &w)| (relu(n) as i32 * w as i32)).sum::<i32>() >> FP_PRECISION_BITS) as i16
    }

    #[inline(always)]
    fn relu(v: i16) -> i16 {
        v.max(0)
    }
}
