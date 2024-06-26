/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
use crate::bitboard::{v_mirror_i8, BitBoards};
use crate::colors::{Color};
use crate::nn::{piece_idx, HL1_HALF_NODES, SCORE_SCALE, king_bucket, IN_TO_H1_WEIGHTS, OUT_BIASES, FP_OUT_MULTIPLIER, BUCKET_SIZE, BUCKETS};
use crate::pieces::{P};
use crate::scores::{MAX_EVAL, MIN_EVAL, sanitize_eval_score};

type HiddenNodes = [i16; HL1_HALF_NODES];

#[derive(Clone)]
pub struct NeuralNetEval {
    hidden_nodes_white: A32<[HiddenNodes; BUCKETS]>, // white perspective
    hidden_nodes_black: A32<[HiddenNodes; BUCKETS]>, // black perspective

    bb_white: [BitBoards; BUCKETS],
    bb_black: [BitBoards; BUCKETS],

    white_offsets: [usize; BUCKETS],
    black_offsets: [usize; BUCKETS],

    white_xor: [usize; BUCKETS],
    black_xor: [usize; BUCKETS],

    white_bucket: usize,
    black_bucket: usize,

    white_offset: usize,
    black_offset: usize,

    xor_white_pov: usize,
    xor_black_pov: usize,

    move_id: usize,
    updates: Vec<(bool, usize, UpdateAction)>,

    undo: bool,
    fast_undo: bool,
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
            hidden_nodes_white: A32([[0; HL1_HALF_NODES]; BUCKETS]),
            hidden_nodes_black: A32([[0; HL1_HALF_NODES]; BUCKETS]),

            bb_white: [BitBoards::default(); BUCKETS],
            bb_black: [BitBoards::default(); BUCKETS],

            white_offsets: [0; BUCKETS],
            black_offsets: [0; BUCKETS],

            white_xor: [0; BUCKETS],
            black_xor: [0; BUCKETS],

            white_bucket: 0,
            black_bucket: 0,

            white_offset: 0,
            black_offset: 0,

            xor_white_pov: 0,
            xor_black_pov: 0,

            move_id: 0,
            updates: Vec::with_capacity(32),

            undo: false,
            fast_undo: false,
        })
    }

    pub fn init_pos(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        self.updates.clear();

        let (white_bucket, black_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(bitboards, white_king, black_king);

        self.update_white_pov(bitboards, white_bucket, xor_white_pov, white_offset);
        self.update_black_pov(bitboards, black_bucket, xor_black_pov, black_offset);
    }

    fn update_white_pov(&mut self, bitboards: &BitBoards, white_bucket: usize, xor_white_pov: usize, white_offset: usize) {
        if self.white_offsets[white_bucket] != white_offset || self.white_xor[white_bucket] != xor_white_pov {
            self.hidden_nodes_white.0[white_bucket].fill(0);
            self.bb_white[white_bucket] = BitBoards::default();
        }

        self.white_offsets[white_bucket] = white_offset;
        self.white_xor[white_bucket] = xor_white_pov;
        self.white_bucket = white_bucket;
        self.xor_white_pov = xor_white_pov;
        self.white_offset = white_offset;

        for piece in 1..=6 {
            let now = bitboards.by_piece(piece);
            let prev = self.bb_white[white_bucket].by_piece(piece);
            for pos in prev & !now {
                self.remove_piece_now_wpov(pos as usize, piece);
            }
            for pos in now & !prev {
                self.add_piece_now_wpov(pos as usize, piece);
            }

            let now = bitboards.by_piece(-piece);
            let prev = self.bb_white[white_bucket].by_piece(-piece);
            for pos in prev & !now {
                self.remove_piece_now_wpov(pos as usize, -piece);
            }
            for pos in now & !prev {
                self.add_piece_now_wpov(pos as usize, -piece);
            }
        }
    }

    fn update_black_pov(&mut self, bitboards: &BitBoards, black_bucket: usize, xor_black_pov: usize, black_offset: usize) {
        if self.black_xor[black_bucket] != xor_black_pov || self.black_offsets[black_bucket] != black_offset {
            self.hidden_nodes_black.0[black_bucket].fill(0);
            self.bb_black[black_bucket] = BitBoards::default();
        }

        self.black_offsets[black_bucket] = black_offset;
        self.black_xor[black_bucket] = xor_black_pov;
        self.black_bucket = black_bucket;
        self.xor_black_pov = xor_black_pov;
        self.black_offset = black_offset;

        for piece in 1..=6 {
            let now = bitboards.by_piece(piece);
            let prev = self.bb_black[black_bucket].by_piece(piece);
            for pos in prev & !now {
                self.remove_piece_now_bpov(pos as usize, piece);
            }
            for pos in now & !prev {
                self.add_piece_now_bpov(pos as usize, piece);
            }

            let now = bitboards.by_piece(-piece);
            let prev = self.bb_black[black_bucket].by_piece(-piece);
            for pos in prev & !now {
                self.remove_piece_now_bpov(pos as usize, -piece);
            }
            for pos in now & !prev {
                self.add_piece_now_bpov(pos as usize, -piece);
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

    pub fn remove_add_piece(&mut self, rem_pos: usize, rem_piece: i8, add_pos: usize, add_piece: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece)));
        }
    }

    pub fn remove_remove_add_piece(&mut self, rem_pos1: usize, rem_piece1: i8, rem_pos2: usize, rem_piece2: i8, add_pos: usize, add_piece: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::RemoveRemoveAdd(rem_pos1, rem_piece1, rem_pos2, rem_piece2, add_pos, add_piece)));
        }
    }

    pub fn remove_add_add_piece(&mut self, rem_pos: usize, rem_piece: i8, add_pos1: usize, add_piece1: i8, add_pos2: usize, add_piece2: i8) {
        if !self.fast_undo {
            self.updates.push((self.undo, self.move_id, UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add_pos1, add_piece1, add_pos2, add_piece2)));
        }
    }

    fn add_piece_now_wpov(&mut self, pos: usize, piece: i8) {
        let white_pov_idx= self.calc_wpov_weight_start(pos, piece);
        add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, white_pov_idx);
    }

    fn add_piece_now_bpov(&mut self, pos: usize, piece: i8) {
        let black_pov_idx= self.calc_bpov_weight_start(pos, piece);
        add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, black_pov_idx);
    }

    fn remove_piece_now_wpov(&mut self, pos: usize, piece: i8) {
        let white_pov_idx= self.calc_wpov_weight_start(pos, piece);
        sub_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, white_pov_idx);
    }

    fn remove_piece_now_bpov(&mut self, pos: usize, piece: i8) {
        let black_pov_idx= self.calc_bpov_weight_start(pos, piece);
        sub_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, black_pov_idx);
    }

    fn calc_pov_weight_start(&self, pos: usize, piece: i8) -> (usize, usize) {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            (
                self.white_offset + base_index + (pos ^ self.xor_white_pov),
                self.black_offset + base_index + OPP_OFFSET + (pos ^ self.xor_black_pov),
            )
        } else {
            (
                self.white_offset + base_index + OPP_OFFSET + (pos ^ self.xor_white_pov),
                self.black_offset + base_index + (pos ^ self.xor_black_pov),
            )
        }
    }

    fn calc_wpov_weight_start(&self, pos: usize, piece: i8) -> usize {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            self.white_offset + base_index + (pos ^ self.xor_white_pov)
        } else {
            self.white_offset + base_index + OPP_OFFSET + (pos ^ self.xor_white_pov)
        }
    }

    fn calc_bpov_weight_start(&self, pos: usize, piece: i8) -> usize {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            self.black_offset + base_index + OPP_OFFSET + (pos ^ self.xor_black_pov)
        } else {
            self.black_offset + base_index + (pos ^ self.xor_black_pov)
        }
    }

    pub fn eval(
        &mut self, active_player: Color, bitboards: &BitBoards, white_king: i8, black_king: i8,
    ) -> i16 {
        self.apply_updates(bitboards, white_king, black_king);

        let (own_hidden_nodes, opp_hidden_nodes) = if active_player.is_white() {
            (&self.hidden_nodes_white.0[self.white_bucket], &self.hidden_nodes_black.0[self.black_bucket])
        } else {
            (&self.hidden_nodes_black.0[self.black_bucket], &self.hidden_nodes_white.0[self.white_bucket])
        };

        let output = (
            (forward_pass(own_hidden_nodes, opp_hidden_nodes) as i64
                + (unsafe { *OUT_BIASES.0.get_unchecked(0) } as i64)
            ) * SCORE_SCALE as i64) / FP_OUT_MULTIPLIER;

        scale_eval(output as i32)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        let (white_bucket, black_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(bitboards, white_king, black_king);

        let refresh_wpov = white_bucket != self.white_bucket || xor_white_pov != self.xor_white_pov || white_offset != self.white_offset;
        let refresh_bpov = black_bucket != self.black_bucket || xor_black_pov != self.xor_black_pov || black_offset != self.black_offset;

        if refresh_wpov {
            self.update_white_pov(bitboards, white_bucket, xor_white_pov, white_offset);
        }

        if refresh_bpov {
            self.update_black_pov(bitboards, black_bucket, xor_black_pov, black_offset);
        }

        if !refresh_wpov && !refresh_bpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                        let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add_white_pov_idx);
                        sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add_black_pov_idx);
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let (rem1_white_pov_idx, rem1_black_pov_idx) = self.calc_pov_weight_start(rem1_pos, rem1_piece);
                        let (rem2_white_pov_idx, rem2_black_pov_idx) = self.calc_pov_weight_start(rem2_pos, rem2_piece);
                        let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_white_pov_idx, rem2_white_pov_idx, add_white_pov_idx);
                        sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_black_pov_idx, rem2_black_pov_idx, add_black_pov_idx);
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                        let (add1_white_pov_idx, add1_black_pov_idx) = self.calc_pov_weight_start(add1_pos, add1_piece);
                        let (add2_white_pov_idx, add2_black_pov_idx) = self.calc_pov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add1_white_pov_idx, add2_white_pov_idx);
                        sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add1_black_pov_idx, add2_black_pov_idx);
                    }
                }
            }

        } else if !refresh_wpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let rem_white_pov_idx = self.calc_wpov_weight_start(rem_pos, rem_piece);
                        let add_white_pov_idx = self.calc_wpov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add_white_pov_idx);
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_white_pov_idx = self.calc_wpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_white_pov_idx = self.calc_wpov_weight_start(rem2_pos, rem2_piece);
                        let add_white_pov_idx = self.calc_wpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_white_pov_idx, rem2_white_pov_idx, add_white_pov_idx);
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_white_pov_idx = self.calc_wpov_weight_start(rem_pos, rem_piece);
                        let add1_white_pov_idx = self.calc_wpov_weight_start(add1_pos, add1_piece);
                        let add2_white_pov_idx = self.calc_wpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_white.0[self.white_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_white_pov_idx, add1_white_pov_idx, add2_white_pov_idx);
                    }
                }
            }
        } else if !refresh_bpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let rem_black_pov_idx = self.calc_bpov_weight_start(rem_pos, rem_piece);
                        let add_black_pov_idx = self.calc_bpov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add_black_pov_idx);
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_black_pov_idx = self.calc_bpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_black_pov_idx = self.calc_bpov_weight_start(rem2_pos, rem2_piece);
                        let add_black_pov_idx = self.calc_bpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem1_black_pov_idx, rem2_black_pov_idx, add_black_pov_idx);
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_black_pov_idx = self.calc_bpov_weight_start(rem_pos, rem_piece);
                        let add1_black_pov_idx = self.calc_bpov_weight_start(add1_pos, add1_piece);
                        let add2_black_pov_idx = self.calc_bpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(&mut self.hidden_nodes_black.0[self.black_bucket], unsafe { &IN_TO_H1_WEIGHTS.0 }, rem_black_pov_idx, add1_black_pov_idx, add2_black_pov_idx);
                    }
                }
            }
        }
        self.bb_white[self.white_bucket] = *bitboards;
        self.bb_black[self.black_bucket] = *bitboards;
        self.updates.clear();
        self.fast_undo = false;
        self.move_id = 0;
    }
}

fn scale_eval(mut score: i32) -> i16 {
    if score > (MAX_EVAL / 2) as i32  {
        score = MAX_EVAL as i32 / 2 + ((score - MAX_EVAL as i32 / 2) / 2);
        let bound = MAX_EVAL as i32 / 2;
        if score > bound {
            score = bound + ((score - bound) / 2);
        }
    } else if score < (MIN_EVAL / 2) as i32 {
        score = MIN_EVAL as i32 / 2 + ((score - MIN_EVAL as i32 / 2) / 2);
        let bound = MIN_EVAL as i32 / 2;
        if score < bound {
            score = bound + ((score - bound) / 2);
        }
    }

    sanitize_eval_score(score) as i16
}

fn calc_bucket_offsets(
    bitboards: &BitBoards, mut white_king: i8, mut black_king: i8,
) -> (usize, usize, usize, usize, usize, usize) {
    let white_king_col = white_king & 7;
    let black_king_col = black_king & 7;

    let mirror_white_pov = white_king_col > 3;
    let mirror_black_pov = black_king_col > 3;

    let no_pawns = (bitboards.by_piece(P) | bitboards.by_piece(-P)).is_empty();
    let v_mirror_white_pov = no_pawns && (white_king / 8) > 3;
    let v_mirror_black_pov = no_pawns && (v_mirror_i8(black_king)) / 8 > 3;

    let mut xor_white_pov = 0;
    let mut xor_black_pov = 0;
    if mirror_white_pov {
        xor_white_pov |= 7;
    }
    if v_mirror_white_pov {
        xor_white_pov |= 56;
    }

    if mirror_black_pov {
        xor_black_pov |= 7;
    }

    if !v_mirror_black_pov {
        xor_black_pov |= 56;
    }

    white_king ^= xor_white_pov as i8;
    black_king ^= xor_black_pov as i8;

    let w_bucket = king_bucket(white_king as u16) as usize;
    let b_bucket = king_bucket(black_king as u16) as usize;
    const BASE_OFFSET: usize = 0;
    let (white_offset, black_offset) = (BASE_OFFSET + w_bucket * BUCKET_SIZE, BASE_OFFSET + b_bucket * BUCKET_SIZE);

    (w_bucket, b_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset)
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
                let h1_x_w_own = _mm256_madd_epi16(h1_relu, w);

                out_accum = _mm256_add_epi32(out_accum, h1_x_w_own);

                let h1 = _mm256_load_si256(transmute(opp_nodes.as_ptr().add(i * 16)));
                let h1_bias = _mm256_load_si256(transmute(H1_BIASES.0.as_ptr().add(i * 16 + HL1_HALF_NODES)));
                let h1_relu = squared(_mm256_min_epu16(_mm256_max_epi16(_mm256_add_epi16(h1, h1_bias), zero), max_relu));
                let w = _mm256_load_si256(transmute(H1_TO_OUT_WEIGHTS.0.as_ptr().add(i * 16 + HL1_HALF_NODES)));
                let h1_x_w_opp = _mm256_madd_epi16(h1_relu, w);

                out_accum = _mm256_add_epi32(out_accum, h1_x_w_opp);
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
        let v_scaled = _mm256_slli_epi16::<{(FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32}>(v);
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
        let v_scaled = _mm_slli_epi16::<{(FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32}>(v);

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
        let v_scaled = (v as i32) << ((FP_OUT_PRECISION_BITS + 16) / 2 - FP_IN_PRECISION_BITS);
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
