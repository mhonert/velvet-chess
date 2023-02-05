/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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
use crate::bitboard::{h_mirror, h_mirror_i8, v_mirror, v_mirror_i8, BitBoards};
use crate::colors::Color;
use crate::nn::{bucket_size, piece_idx, FP_HIDDEN_MULTIPLIER, FP_INPUT_MULTIPLIER, HL_HALF_NODES, INPUTS, INPUT_BIASES, INPUT_WEIGHTS, KING_BUCKETS, OUTPUT_BIASES, OUTPUT_WEIGHTS, SCORE_SCALE};
use crate::pieces::{B, K, Q, R};
use crate::scores::sanitize_eval_score;

#[derive(Clone)]
pub struct NeuralNetEval {
    hidden_nodes_white: A32<[i16; HL_HALF_NODES]>, // white perspective
    hidden_nodes_black: A32<[i16; HL_HALF_NODES]>, // black perspective

    king_offset: u16,
    white_offset: u16,
    black_offset: u16,

    mirror_white_pov: bool,
    mirror_black_pov: bool,

    max_piece_id: u16,

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
        Box::new(NeuralNetEval {
            hidden_nodes_white: A32([0; HL_HALF_NODES]),
            hidden_nodes_black: A32([0; HL_HALF_NODES]),

            king_offset: 0,
            white_offset: 0,
            black_offset: 0,

            mirror_white_pov: true,
            mirror_black_pov: true,

            max_piece_id: piece_idx(Q),

            move_id: 0,
            updates: Vec::with_capacity(32),

            undo: false,
            fast_undo: false,
        })
    }

    pub fn init_pos(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        self.updates.clear();
        self.fast_undo = false;
        self.move_id = 0;
        self.hidden_nodes_white.0.fill(0);
        self.hidden_nodes_black.0.fill(0);

        let (mirror_white_pov, mirror_black_pov, king_offset, white_offset, black_offset, max_piece_id) =
            calc_bucket_offsets(bitboards, white_king, black_king);
        self.mirror_white_pov = mirror_white_pov;
        self.mirror_black_pov = mirror_black_pov;
        self.king_offset = king_offset;
        self.white_offset = white_offset;
        self.black_offset = black_offset;
        self.max_piece_id = max_piece_id;

        for piece in 1..=6 {
            for pos in bitboards.by_piece(piece) {
                self.add_piece_now(pos as usize, piece);
            }

            for pos in bitboards.by_piece(-piece) {
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
        let (white_pov_idx, black_pov_idx, idx) = self.calc_pov_weight_start(pos, piece);
        if idx > self.max_piece_id {
            return;
        }

        add_weights::<HL_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &INPUT_WEIGHTS.0 }, white_pov_idx);
        add_weights::<HL_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &INPUT_WEIGHTS.0 }, black_pov_idx);
    }

    pub fn remove_piece(&mut self, pos: usize, piece: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::Remove(pos, piece)));
        }
    }

    fn remove_piece_now(&mut self, pos: usize, piece: i8) {
        let (white_pov_idx, black_pov_idx, idx) = self.calc_pov_weight_start(pos, piece);
        if idx > self.max_piece_id {
            return;
        }

        sub_weights::<HL_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &INPUT_WEIGHTS.0 }, white_pov_idx);
        sub_weights::<HL_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &INPUT_WEIGHTS.0 }, black_pov_idx);
    }

    fn calc_pov_weight_start(&self, pos: usize, piece: i8) -> (usize, usize, u16) {
        let idx = piece_idx(piece.unsigned_abs() as i8);
        let (white_offset, black_offset) = if piece.abs() == K {
            (self.king_offset, self.king_offset)
        } else {
            (self.white_offset, self.black_offset)
        };
        let base_index = idx as usize * 64;
        let opp_offset = INPUTS / 2;

        let (white_pov_idx, black_pov_idx) = if piece > 0 {
            (
                white_offset as usize + base_index + h_mirror_if(self.mirror_white_pov, pos),
                black_offset as usize + base_index + opp_offset + v_mirror(h_mirror_if(self.mirror_black_pov, pos)),
            )
        } else {
            (
                white_offset as usize + base_index + opp_offset + h_mirror_if(self.mirror_white_pov, pos),
                black_offset as usize + base_index + v_mirror(h_mirror_if(self.mirror_black_pov, pos)),
            )
        };
        (white_pov_idx, black_pov_idx, idx)
    }

    pub fn check_refresh(&mut self) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::CheckRefresh));
        }
    }

    pub fn eval(
        &mut self, active_player: Color, half_move_clock: u8, bitboards: &BitBoards, white_king: i8, black_king: i8,
    ) -> i32 {
        self.apply_updates(bitboards, white_king, black_king);

        let (own_hidden_nodes, opp_hidden_nodes) = if active_player.is_white() {
            (&self.hidden_nodes_white.0, &self.hidden_nodes_black.0)
        } else {
            (&self.hidden_nodes_black.0, &self.hidden_nodes_white.0)
        };

        let output = (forward_pass::<HL_HALF_NODES>(
            own_hidden_nodes,
            unsafe { &OUTPUT_WEIGHTS.0[0..HL_HALF_NODES] },
            unsafe { &INPUT_BIASES.0[0..HL_HALF_NODES] },
            opp_hidden_nodes,
            unsafe { &OUTPUT_WEIGHTS.0[HL_HALF_NODES..] },
            unsafe { &INPUT_BIASES.0[HL_HALF_NODES..] },
        ) as i64
            + unsafe { *OUTPUT_BIASES.0.get_unchecked(0) } as i64 * FP_INPUT_MULTIPLIER as i64)
            * SCORE_SCALE as i64
            / (FP_INPUT_MULTIPLIER as i64 * FP_HIDDEN_MULTIPLIER as i64);

        let score = active_player.score(sanitize_eval_score(output as i32));
        adjust_eval(score, half_move_clock)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        for i in 0..self.updates.len() {
            let (_, _, update) = unsafe { self.updates.get_unchecked(i) };
            match *update {
                UpdateAction::CheckRefresh => {
                    let (mirror_white_pov, mirror_black_pov, king_offset, white_offset, black_offset, _) =
                        calc_bucket_offsets(bitboards, white_king, black_king);
                    if mirror_white_pov != self.mirror_white_pov
                        || mirror_black_pov != self.mirror_black_pov
                        || white_offset != self.white_offset
                        || black_offset != self.black_offset
                        || king_offset != self.king_offset
                    {
                        self.init_pos(bitboards, white_king, black_king);
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
    let remaining_half_moves = 0.max(100 - half_move_clock as i32);
    if remaining_half_moves >= 95 {
        score
    } else {
        score * remaining_half_moves / 95
    }
}

fn calc_bucket_offsets(
    bitboards: &BitBoards, mut white_king: i8, mut black_king: i8,
) -> (bool, bool, u16, u16, u16, u16) {
    let white_king_col = white_king & 7;
    let black_king_col = black_king & 7;

    let mirror_white_pov = white_king_col > 3;
    let mirror_black_pov = black_king_col > 3;

    if mirror_white_pov {
        white_king = h_mirror_i8(white_king);
    }
    if mirror_black_pov {
        black_king = h_mirror_i8(black_king);
    }

    let (king_bucket, bucket_offset, max_piece_id) =
        if bitboards.by_piece(Q).is_empty() && bitboards.by_piece(-Q).is_empty() {
            if bitboards.by_piece(R).is_empty() && bitboards.by_piece(-R).is_empty() {
                (2, (bucket_size(Q) + bucket_size(R)) * KING_BUCKETS, B)
            } else {
                (1, bucket_size(Q) * KING_BUCKETS, R)
            }
        } else {
            (0, 0, Q)
        };

    let bucket_size = bucket_size(max_piece_id);
    let (white_offset, black_offset) = (
        64 * 4 + bucket_offset as u16 + board_eighth(white_king) * bucket_size as u16,
        64 * 4 + bucket_offset as u16 + board_eighth(v_mirror_i8(black_king)) * bucket_size as u16,
    );

    (mirror_white_pov, mirror_black_pov, 64 * king_bucket, white_offset, black_offset, piece_idx(max_piece_id))
}

#[inline(always)]
fn forward_pass<const N: usize>(
    own_nodes: &[i16], own_weights: &[i16], own_biases: &[i16], opp_nodes: &[i16], opp_weights: &[i16],
    opp_biases: &[i16],
) -> i32 {
    #[cfg(target_feature = "avx2")]
    {
        avx2::forward_pass::<N>(own_nodes, own_weights, own_biases, opp_nodes, opp_weights, opp_biases)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::forward_pass::<N>(own_nodes, own_weights, own_biases, opp_nodes, opp_weights, opp_biases)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::forward_pass::<N>(own_nodes, own_weights, own_biases, opp_nodes, opp_weights, opp_biases)
    }
}

#[inline(always)]
pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
    #[cfg(target_feature = "avx2")]
    {
        avx2::add_weights::<N>(nodes, weights, weight_idx)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::add_weights::<N>(nodes, weights, weight_idx)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::add_weights::<N>(nodes, weights, weight_idx)
    }
}

#[inline(always)]
pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
    #[cfg(target_feature = "avx2")]
    {
        avx2::sub_weights::<N>(nodes, weights, weight_idx)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::sub_weights::<N>(nodes, weights, weight_idx)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::sub_weights::<N>(nodes, weights, weight_idx)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;

    #[inline(always)]
    pub fn forward_pass<const N: usize>(
        own_nodes: &[i16], own_weights: &[i16], own_biases: &[i16], opp_nodes: &[i16], opp_weights: &[i16],
        opp_biases: &[i16],
    ) -> i32 {
        unsafe {
            let mut acc = _mm256_setzero_si256();

            for i in 0..(N / 16) {
                acc = _mm256_add_epi32(acc, apply_weights(i, own_nodes, own_weights, own_biases));
                acc = _mm256_add_epi32(acc, apply_weights(i, opp_nodes, opp_weights, opp_biases));
            }

            // Final horizontal sum of the lanes for the accumulator
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(hi64, sum128);
            let hi32 = _mm_shuffle_epi32::<0b10110001>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            _mm_cvtsi128_si32(sum32)
        }
    }

    #[inline(always)]
    pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        let w_start = weight_idx * N;
        unsafe {
            for i in 0..(N / 16) {
                let w = _mm256_load_si256(transmute(weights.as_ptr().add(w_start + i * 16)));
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                _mm256_store_si256(transmute(nodes.as_ptr().add(i * 16)), _mm256_add_epi16(n, w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        let w_start = weight_idx * N;
        unsafe {
            for i in 0..(N / 16) {
                let w = _mm256_load_si256(transmute(weights.as_ptr().add(w_start + i * 16)));
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                _mm256_store_si256(transmute(nodes.as_ptr().add(i * 16)), _mm256_sub_epi16(n, w));
            }
        }
    }

    #[inline(always)]
    fn apply_weights(i: usize, nodes: &[i16], weights: &[i16], biases: &[i16]) -> __m256i {
        unsafe {
            let zero = _mm256_setzero_si256();

            let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
            let bias = _mm256_load_si256(transmute(biases.as_ptr().add(i * 16)));
            let n_relu = _mm256_max_epi16(_mm256_add_epi16(n, bias), zero);

            let w = _mm256_load_si256(transmute(weights.as_ptr().add(i * 16)));
            _mm256_madd_epi16(n_relu, w)
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2 {
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;

    #[inline(always)]
    pub fn forward_pass<const N: usize>(
        own_nodes: &[i16], own_weights: &[i16], own_biases: &[i16], opp_nodes: &[i16], opp_weights: &[i16],
        opp_biases: &[i16],
    ) -> i32 {
        unsafe {
            let mut acc = _mm_setzero_si128();

            for i in 0..(N / 8) {
                acc = _mm_add_epi32(acc, apply_weights(i, own_nodes, own_weights, own_biases));
                acc = _mm_add_epi32(acc, apply_weights(i, opp_nodes, opp_weights, opp_biases));
            }

            // Final horizontal sum of the lanes for the accumulator
            let hi64 = _mm_shuffle_epi32::<0b01001110>(acc);
            let sum64 = _mm_add_epi32(hi64, acc);
            let hi32 = _mm_shufflelo_epi16::<0b01001110>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            _mm_cvtsi128_si32(sum32)
        }
    }

    #[inline(always)]
    pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        let w_start = weight_idx * N;
        unsafe {
            for i in 0..(N / 8) {
                let w = _mm_load_si128(transmute(weights.as_ptr().add(w_start + i * 8)));
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                _mm_store_si128(transmute(nodes.as_ptr().add(i * 8)), _mm_add_epi16(n, w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        let w_start = weight_idx * N;
        unsafe {
            for i in 0..(N / 8) {
                let w = _mm_load_si128(transmute(weights.as_ptr().add(w_start + i * 8)));
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                _mm_store_si128(transmute(nodes.as_ptr().add(i * 8)), _mm_sub_epi16(n, w));
            }
        }
    }

    #[inline(always)]
    fn apply_weights(i: usize, nodes: &[i16], weights: &[i16], biases: &[i16]) -> __m128i {
        unsafe {
            let zero = _mm_setzero_si128();

            let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
            let bias = _mm_load_si128(transmute(biases.as_ptr().add(i * 8)));
            let n_relu = _mm_max_epi16(_mm_add_epi16(n, bias), zero);
            let w = _mm_load_si128(transmute(weights.as_ptr().add(i * 8)));
            _mm_madd_epi16(n_relu, w)
        }
    }
}

#[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
mod fallback {
    #[inline(always)]
    pub fn forward_pass<const N: usize>(
        own_nodes: &[i16], own_weights: &[i16], own_biases: &[i16], opp_nodes: &[i16], opp_weights: &[i16],
        opp_biases: &[i16],
    ) -> i32 {
        apply_weights(own_nodes, own_weights, own_biases) + apply_weights(opp_nodes, opp_weights, opp_biases)
    }

    #[inline(always)]
    pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        for (nodes, weight) in nodes.iter_mut().zip(weights.chunks_exact(N).nth(weight_idx).unwrap()) {
            *nodes += *weight as i16;
        }
    }

    #[inline(always)]
    pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        for (nodes, weight) in nodes.iter_mut().zip(weights.chunks_exact(N).nth(weight_idx).unwrap()) {
            *nodes -= *weight as i16;
        }
    }

    #[inline(always)]
    fn apply_weights(nodes: &[i16], weights: &[i16], biases: &[i16]) -> i32 {
        nodes
            .iter()
            .zip(biases.iter())
            .zip(weights.iter())
            .map(|((&n, &b), &w)| (relu(n + b) as i32 * w as i32))
            .sum::<i32>()
    }

    #[inline(always)]
    fn relu(v: i16) -> i16 {
        v.max(0)
    }
}

fn board_eighth(pos: i8) -> u16 {
    let row = pos / 8;
    let col = pos & 3;
    ((row / 2) * 2 + (col / 2)) as u16
}

fn h_mirror_if(mirror: bool, pos: usize) -> usize {
    if mirror {
        h_mirror(pos)
    } else {
        pos
    }
}
