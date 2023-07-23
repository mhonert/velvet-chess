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
use crate::bitboard::{v_mirror, v_mirror_i8, BitBoards, h_mirror_i8, h_mirror};
use crate::colors::Color;
use crate::nn::{piece_idx, HL1_HALF_NODES, KING_BUCKETS, SCORE_SCALE, board_4, IN_TO_H1_WEIGHTS, OUT_BIASES, FP_OUT_MULTIPLIER};
use crate::pieces::{Q, R};
use crate::scores::{MAX_EVAL, MIN_EVAL, sanitize_eval_score};

#[derive(Clone)]
pub struct NeuralNetEval {
    hidden_nodes_white: A32<[i16; HL1_HALF_NODES]>, // white perspective
    hidden_nodes_black: A32<[i16; HL1_HALF_NODES]>, // black perspective

    white_offset: u16,
    black_offset: u16,

    mirror_white_pov: bool,
    mirror_black_pov: bool,

    move_id: usize,
    updates: Vec<(bool, usize, bool, UpdateAction)>,

    undo: bool,
    fast_undo: bool,

    check_refresh: i32,
}

#[derive(Clone)]
enum UpdateAction {
    RemoveAdd(usize, i8, usize, i8),
    RemoveRemoveAdd(usize, i8, usize, i8, usize, i8),
    RemoveAddAdd(usize, i8, usize, i8, usize, i8),
}

impl NeuralNetEval {
    pub fn new() -> Box<Self> {
        Box::new(NeuralNetEval {
            hidden_nodes_white: A32([0; HL1_HALF_NODES]),
            hidden_nodes_black: A32([0; HL1_HALF_NODES]),

            white_offset: 0,
            black_offset: 0,

            mirror_white_pov: true,
            mirror_black_pov: true,

            move_id: 0,
            updates: Vec::with_capacity(32),

            undo: false,
            fast_undo: false,
            check_refresh: 0,
        })
    }

    pub fn init_pos(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        self.updates.clear();
        self.fast_undo = false;
        self.move_id = 0;
        self.hidden_nodes_white.0.fill(0);
        self.hidden_nodes_black.0.fill(0);

        let (mirror_white_pov, mirror_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(bitboards, white_king, black_king);
        self.mirror_white_pov = mirror_white_pov;
        self.mirror_black_pov = mirror_black_pov;
        self.white_offset = white_offset;
        self.black_offset = black_offset;

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
        while let Some((was_undo, id, was_check_refresh, _)) = self.updates.last() {
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

            if *was_check_refresh {
                self.check_refresh -= 1;
            }
            self.updates.pop();
        }
    }

    pub fn remove_add_piece(&mut self, check_refresh: bool, rem_pos: usize, rem_piece: i8, add_pos: usize, add_piece: i8) {
        if !self.fast_undo {
            if check_refresh {
                self.check_refresh += 1;
            }
            self.updates.push((self.undo, self.move_id, check_refresh, UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece)));
        }
    }

    pub fn remove_remove_add_piece(&mut self, check_refresh: bool, rem_pos1: usize, rem_piece1: i8, rem_pos2: usize, rem_piece2: i8, add_pos: usize, add_piece: i8) {
        if !self.fast_undo {
            if check_refresh {
                self.check_refresh += 1;
            }
            self.updates.push((self.undo, self.move_id, check_refresh, UpdateAction::RemoveRemoveAdd(rem_pos1, rem_piece1, rem_pos2, rem_piece2, add_pos, add_piece)));
        }
    }

    pub fn remove_add_add_piece(&mut self, check_refresh: bool, rem_pos: usize, rem_piece: i8, add_pos1: usize, add_piece1: i8, add_pos2: usize, add_piece2: i8) {
        if !self.fast_undo {
            if check_refresh {
                self.check_refresh += 1;
            }
            self.updates.push((self.undo, self.move_id, check_refresh, UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add_pos1, add_piece1, add_pos2, add_piece2)));
        }
    }

    fn add_piece_now(&mut self, pos: usize, piece: i8) {
        let (white_pov_idx, black_pov_idx) = self.calc_pov_weight_start(pos, piece);

        add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, white_pov_idx);
        add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, black_pov_idx);
    }

    fn calc_pov_weight_start(&self, pos: usize, piece: i8) -> (usize, usize) {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            (
                self.white_offset as usize + base_index + h_mirror_if(self.mirror_white_pov, pos),
                self.black_offset as usize + base_index + OPP_OFFSET + v_mirror(h_mirror_if(self.mirror_black_pov, pos)),
            )
        } else {
            (
                self.white_offset as usize + base_index + OPP_OFFSET + h_mirror_if(self.mirror_white_pov, pos),
                self.black_offset as usize + base_index + v_mirror(h_mirror_if(self.mirror_black_pov, pos)),
            )
        }
    }

    pub fn eval(
        &mut self, active_player: Color, bitboards: &BitBoards, white_king: i8, black_king: i8,
    ) -> i32 {
        self.apply_updates(bitboards, white_king, black_king);

        let (own_hidden_nodes, opp_hidden_nodes) = if active_player.is_white() {
            (&self.hidden_nodes_white.0, &self.hidden_nodes_black.0)
        } else {
            (&self.hidden_nodes_black.0, &self.hidden_nodes_white.0)
        };

        let output = (
            (forward_pass(own_hidden_nodes, opp_hidden_nodes) as i64
                + (unsafe { *OUT_BIASES.0.get_unchecked(0) } as i64)
            ) * SCORE_SCALE as i64) / FP_OUT_MULTIPLIER;

        scale_eval(output as i32)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        if self.check_refresh > 0 {
            self.check_refresh = 0;
            let (mirror_white_pov, mirror_black_pov, white_offset, black_offset) =
                calc_bucket_offsets(bitboards, white_king, black_king);
            if mirror_white_pov != self.mirror_white_pov
                || mirror_black_pov != self.mirror_black_pov
                || white_offset != self.white_offset
                || black_offset != self.black_offset
            {
                self.init_pos(bitboards, white_king, black_king);
                return;
            }
        }

        for (_, _, _, update) in self.updates.iter() {
            match *update {
                UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                    let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                    let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                    sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add_white_pov_idx);
                    sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add_black_pov_idx);
                }

                UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                    let (rem1_white_pov_idx, rem1_black_pov_idx) = self.calc_pov_weight_start(rem1_pos, rem1_piece);
                    let (rem2_white_pov_idx, rem2_black_pov_idx) = self.calc_pov_weight_start(rem2_pos, rem2_piece);
                    let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                    sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_white_pov_idx, rem2_white_pov_idx, add_white_pov_idx);
                    sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_black_pov_idx, rem2_black_pov_idx, add_black_pov_idx);
                }

                UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                    let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                    let (add1_white_pov_idx, add1_black_pov_idx) = self.calc_pov_weight_start(add1_pos, add1_piece);
                    let (add2_white_pov_idx, add2_black_pov_idx) = self.calc_pov_weight_start(add2_pos, add2_piece);

                    sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add1_white_pov_idx, add2_white_pov_idx);
                    sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0, unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add1_black_pov_idx, add2_black_pov_idx);
                }
            }
        }
        self.updates.clear();
    }
}

fn scale_eval(mut score: i32) -> i32 {
    if score > MAX_EVAL / 2  {
        score = MAX_EVAL / 2 + ((score - MAX_EVAL / 2) / 2);
        let bound = MAX_EVAL * 3 / 2;
        if score > bound {
            score = bound + ((score - bound) / 2);
            score = sanitize_eval_score(score);
        }
    } else if score < MIN_EVAL / 2 {
        score = MIN_EVAL / 2 + ((score - MIN_EVAL / 2) / 2);
        let bound = MIN_EVAL * 3 / 2;
        if score < bound {
            score = bound + ((score - bound) / 2);
            score = sanitize_eval_score(score);
        }
    }

    score
}

fn calc_bucket_offsets(
    bitboards: &BitBoards, mut white_king: i8, mut black_king: i8,
) -> (bool, bool, u16, u16) {
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

    let w_kingrel_bucket = board_4(white_king as u16);
    let b_kingrel_bucket = board_4(v_mirror_i8(black_king) as u16);

    let piece_bucket = if (bitboards.by_piece(Q) | bitboards.by_piece(-Q)).is_empty() {
        if (bitboards.by_piece(R) | bitboards.by_piece(-R)).is_empty() {
            2
        } else {
            1
        }
    } else {
        0
    };

    let w_bucket: u16 = piece_bucket * KING_BUCKETS as u16 + w_kingrel_bucket;
    let b_bucket: u16 = piece_bucket * KING_BUCKETS as u16 + b_kingrel_bucket;

    const BUCKET_SIZE: i32 = 6 * 64 * 2;
    let (white_offset, black_offset) = (
        w_bucket * BUCKET_SIZE as u16,
        b_bucket * BUCKET_SIZE as u16,
    );

    (mirror_white_pov, mirror_black_pov, white_offset, black_offset)
}

#[inline(always)]
fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
    #[cfg(target_feature = "avx2")]
    {
        avx2::forward_pass(own_nodes, opp_nodes)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::forward_pass(own_nodes, opp_nodes)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::forward_pass(own_nodes, opp_nodes)
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
pub fn sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add_weight_idx: usize) {
    #[cfg(target_feature = "avx2")]
    {
        avx2::sub_add_weights::<N>(nodes, weights, sub_weight_idx, add_weight_idx)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::sub_add_weights::<N>(nodes, weights, sub_weight_idx, add_weight_idx)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::sub_weights::<N>(nodes, weights, sub_weight_idx);
        fallback::add_weights::<N>(nodes, weights, add_weight_idx);
    }
}

#[inline(always)]
pub fn sub_sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub1_weight_idx: usize, sub2_weight_idx: usize, add_weight_idx: usize) {
    #[cfg(target_feature = "avx2")]
    {
        avx2::sub_sub_add_weights::<N>(nodes, weights, sub1_weight_idx, sub2_weight_idx, add_weight_idx)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::sub_sub_add_weights::<N>(nodes, weights, sub1_weight_idx, sub2_weight_idx, add_weight_idx)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::sub_weights::<N>(nodes, weights, sub1_weight_idx);
        fallback::sub_weights::<N>(nodes, weights, sub2_weight_idx);
        fallback::add_weights::<N>(nodes, weights, add_weight_idx);
    }
}

#[inline(always)]
pub fn sub_add_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add1_weight_idx: usize, add2_weight_idx: usize) {
    #[cfg(target_feature = "avx2")]
    {
        avx2::sub_add_add_weights::<N>(nodes, weights, sub_weight_idx, add1_weight_idx, add2_weight_idx)
    }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
    {
        sse2::sub_add_add_weights::<N>(nodes, weights, sub_weight_idx, add1_weight_idx, add2_weight_idx)
    }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
    {
        fallback::sub_weights::<N>(nodes, weights, sub_weight_idx);
        fallback::add_weights::<N>(nodes, weights, add1_weight_idx);
        fallback::add_weights::<N>(nodes, weights, add2_weight_idx);
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
    use crate::nn::{FP_MAX_RELU, FP_IN_PRECISION_BITS, H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, FP_OUT_PRECISION_BITS};

    #[inline(always)]
    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        unsafe {
            // H1 to ML
            let zero = _mm256_setzero_si256();
            let max_relu = _mm256_set1_epi16(FP_MAX_RELU);

            let mut out_accum = zero;
            for i in 0..HL1_HALF_NODES / 16 {
                let h1 = _mm256_load_si256(transmute(own_nodes.as_ptr().add(i * 16)));
                let h1_bias = _mm256_load_si256(transmute(H1_BIASES.0.as_ptr().add(i * 16)));
                let h1_relu = squared(_mm256_min_epu16(_mm256_max_epi16(_mm256_add_epi16(h1, h1_bias), zero), max_relu));
                let w = _mm256_load_si256(transmute(H1_TO_OUT_WEIGHTS.0.as_ptr().add(i * 16)));
                let h1_x_w = _mm256_madd_epi16(h1_relu, w);

                out_accum = _mm256_add_epi32(out_accum, h1_x_w);

                let h1 = _mm256_load_si256(transmute(opp_nodes.as_ptr().add(i * 16)));
                let h1_bias = _mm256_load_si256(transmute(H1_BIASES.0.as_ptr().add(i * 16 + HL1_HALF_NODES)));
                let h1_relu = squared(_mm256_min_epu16(_mm256_max_epi16(_mm256_add_epi16(h1, h1_bias), zero), max_relu));
                let w = _mm256_load_si256(transmute(H1_TO_OUT_WEIGHTS.0.as_ptr().add(i * 16 + HL1_HALF_NODES)));
                let h1_x_w = _mm256_madd_epi16(h1_relu, w);

                out_accum = _mm256_add_epi32(out_accum, h1_x_w);
            }

            // Final horizontal sum of the lanes for the accumulator
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(out_accum), _mm256_extracti128_si256::<1>(out_accum));
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(hi64, sum128);
            let hi32 = _mm_shuffle_epi32::<0b10110001>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            _mm_cvtsi128_si32(sum32) >> FP_OUT_PRECISION_BITS as i32
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
    pub fn sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add_weight_idx: usize) {
        let sub_w_start = sub_weight_idx * N;
        let add_w_start = add_weight_idx * N;
        unsafe {
            for i in 0..(N / 16) {
                let sub_w = _mm256_load_si256(transmute(weights.as_ptr().add(sub_w_start + i * 16)));
                let add_w = _mm256_load_si256(transmute(weights.as_ptr().add(add_w_start + i * 16)));
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                _mm256_store_si256(transmute(nodes.as_ptr().add(i * 16)), _mm256_add_epi16(_mm256_sub_epi16(n, sub_w), add_w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub1_weight_idx: usize, sub2_weight_idx: usize, add_weight_idx: usize) {
        let sub1_w_start = sub1_weight_idx * N;
        let sub2_w_start = sub2_weight_idx * N;
        let add_w_start = add_weight_idx * N;
        unsafe {
            for i in 0..(N / 16) {
                let sub1_w = _mm256_load_si256(transmute(weights.as_ptr().add(sub1_w_start + i * 16)));
                let sub2_w = _mm256_load_si256(transmute(weights.as_ptr().add(sub2_w_start + i * 16)));
                let add_w = _mm256_load_si256(transmute(weights.as_ptr().add(add_w_start + i * 16)));
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                _mm256_store_si256(transmute(nodes.as_ptr().add(i * 16)), _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(n, sub1_w), sub2_w), add_w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_add_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add1_weight_idx: usize, add2_weight_idx: usize) {
        let sub_w_start = sub_weight_idx * N;
        let add1_w_start = add1_weight_idx * N;
        let add2_w_start = add2_weight_idx * N;
        unsafe {
            for i in 0..(N / 16) {
                let sub_w = _mm256_load_si256(transmute(weights.as_ptr().add(sub_w_start + i * 16)));
                let add1_w = _mm256_load_si256(transmute(weights.as_ptr().add(add1_w_start + i * 16)));
                let add2_w = _mm256_load_si256(transmute(weights.as_ptr().add(add2_w_start + i * 16)));
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                _mm256_store_si256(transmute(nodes.as_ptr().add(i * 16)), _mm256_add_epi16(_mm256_add_epi16(_mm256_sub_epi16(n, sub_w), add1_w), add2_w));
            }
        }
    }

    #[inline(always)]
    unsafe fn squared(v: __m256i) -> __m256i {
        let v_scaled = _mm256_slli_epi16::<{13 - FP_IN_PRECISION_BITS as i32}>(v);
        _mm256_mulhi_epu16(v_scaled, v_scaled)
    }

}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2 {
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;
    use crate::nn::{FP_MAX_RELU, FP_IN_PRECISION_BITS, H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, FP_OUT_PRECISION_BITS};

    #[inline(always)]
    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        unsafe {
            // H1 to ML
            let zero = _mm_setzero_si128();
            let max_relu = _mm_set1_epi16(FP_MAX_RELU);

            let mut out_accum = zero;
            for i in 0..HL1_HALF_NODES / 8 {
                let h1 = _mm_load_si128(transmute(own_nodes.as_ptr().add(i * 8)));
                let h1_bias = _mm_load_si128(transmute(H1_BIASES.0.as_ptr().add(i * 8)));
                let h1_relu = squared(_mm_min_epu16(_mm_max_epi16(_mm_add_epi16(h1, h1_bias), zero), max_relu));
                let w = _mm_load_si128(transmute(H1_TO_OUT_WEIGHTS.0.as_ptr().add(i * 8)));
                let h1_x_w = _mm_madd_epi16(h1_relu, w);

                out_accum = _mm_add_epi32(out_accum, h1_x_w);

                let h1 = _mm_load_si128(transmute(opp_nodes.as_ptr().add(i * 8)));
                let h1_bias = _mm_load_si128(transmute(H1_BIASES.0.as_ptr().add(i * 8 + HL1_HALF_NODES)));
                let h1_relu = squared(_mm_min_epu16(_mm_max_epi16(_mm_add_epi16(h1, h1_bias), zero), max_relu));
                let w = _mm_load_si128(transmute(H1_TO_OUT_WEIGHTS.0.as_ptr().add(i * 8 + HL1_HALF_NODES)));
                let h1_x_w = _mm_madd_epi16(h1_relu, w);

                out_accum = _mm_add_epi32(out_accum, h1_x_w);
            }

            // Final horizontal sum of the lanes for the accumulator
            let hi64 = _mm_unpackhi_epi64(out_accum, out_accum);
            let sum64 = _mm_add_epi32(hi64, out_accum);
            let hi32 = _mm_shuffle_epi32::<0b10110001>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            _mm_cvtsi128_si32(sum32) >> FP_OUT_PRECISION_BITS as i32
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
    pub fn sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add_weight_idx: usize) {
        let sub_w_start = sub_weight_idx * N;
        let add_w_start = add_weight_idx * N;
        unsafe {
            for i in 0..(N / 8) {
                let sub_w = _mm_load_si128(transmute(weights.as_ptr().add(sub_w_start + i * 8)));
                let add_w = _mm_load_si128(transmute(weights.as_ptr().add(add_w_start + i * 8)));
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                _mm_store_si128(transmute(nodes.as_ptr().add(i * 8)), _mm_add_epi16(_mm_sub_epi16(n, sub_w), add_w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_sub_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub1_weight_idx: usize, sub2_weight_idx: usize, add_weight_idx: usize) {
        let sub1_w_start = sub1_weight_idx * N;
        let sub2_w_start = sub2_weight_idx * N;
        let add_w_start = add_weight_idx * N;
        unsafe {
            for i in 0..(N / 8) {
                let sub1_w = _mm_load_si128(transmute(weights.as_ptr().add(sub1_w_start + i * 8)));
                let sub2_w = _mm_load_si128(transmute(weights.as_ptr().add(sub2_w_start + i * 8)));
                let add_w = _mm_load_si128(transmute(weights.as_ptr().add(add_w_start + i * 8)));
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                _mm_store_si128(transmute(nodes.as_ptr().add(i * 8)), _mm_add_epi16(_mm_sub_epi16(_mm_sub_epi16(n, sub1_w), sub2_w), add_w));
            }
        }
    }

    #[inline(always)]
    pub fn sub_add_add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add1_weight_idx: usize, add2_weight_idx: usize) {
        let sub_w_start = sub_weight_idx * N;
        let add1_w_start = add1_weight_idx * N;
        let add2_w_start = add2_weight_idx * N;
        unsafe {
            for i in 0..(N / 8) {
                let sub_w = _mm_load_si128(transmute(weights.as_ptr().add(sub_w_start + i * 8)));
                let add1_w = _mm_load_si128(transmute(weights.as_ptr().add(add1_w_start + i * 8)));
                let add2_w = _mm_load_si128(transmute(weights.as_ptr().add(add2_w_start + i * 8)));
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                _mm_store_si128(transmute(nodes.as_ptr().add(i * 8)), _mm_add_epi16(_mm_add_epi16(_mm_sub_epi16(n, sub_w), add1_w), add2_w));
            }
        }
    }

    #[inline(always)]
    unsafe fn squared(v: __m128i) -> __m128i {
        let v_scaled = _mm_slli_epi16::<{13 - FP_IN_PRECISION_BITS as i32}>(v);
        _mm_mulhi_epu16(v_scaled, v_scaled)
    }
}

#[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
mod fallback {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS, H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES};

    #[inline(always)]
    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        // H1 to ML
        let zero = 0;
        let max_relu = FP_MAX_RELU;

        let mut out_accum = zero as i32;
        for i in 0..HL1_HALF_NODES {
            unsafe {
                let h1 = *own_nodes.get_unchecked(i);
                let h1_bias = *H1_BIASES.0.get_unchecked(i);
                let h1_relu = squared(max_relu.min(zero.max(h1 + h1_bias)));
                let w = *H1_TO_OUT_WEIGHTS.0.get_unchecked(i);
                let h1_x_w = h1_relu as i32 * w as i32;
                out_accum += h1_x_w;

                let h1 = *opp_nodes.get_unchecked(i);
                let h1_bias = *H1_BIASES.0.get_unchecked(i + HL1_HALF_NODES);
                let h1_relu = squared(max_relu.min(zero.max(h1 + h1_bias)));
                let w = *H1_TO_OUT_WEIGHTS.0.get_unchecked(i + HL1_HALF_NODES);
                let h1_x_w = h1_relu as i32 * w as i32;
                out_accum += h1_x_w;
            }
        }

        out_accum >> FP_OUT_PRECISION_BITS as i32
    }

    #[inline(always)]
    fn squared(v: i16) -> i16 {
        let v_scaled = (v as i32) << (13 - FP_IN_PRECISION_BITS);
        ((v_scaled * v_scaled) >> 16) as i16
    }

    #[inline(always)]
    pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        for (nodes, weight) in nodes.iter_mut().zip(weights.chunks_exact(N).nth(weight_idx).unwrap()) {
            *nodes += *weight;
        }
    }

    #[inline(always)]
    pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
        for (nodes, weight) in nodes.iter_mut().zip(weights.chunks_exact(N).nth(weight_idx).unwrap()) {
            *nodes -= *weight;
        }
    }
}

fn h_mirror_if(mirror: bool, pos: usize) -> usize {
    if mirror {
        h_mirror(pos)
    } else {
        pos
    }
}
