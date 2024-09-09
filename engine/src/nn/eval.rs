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

use crate::align::A64;
use crate::bitboard::{v_mirror_i8, BitBoards};
use crate::colors::Color;
use crate::nn::eval::base::{
    add_epi16, clipped_relu, horizontal_sum_32, load, multiply_add_epi16, square, store, sub_epi16, zero, Accum,
    WORDS_PER_REG,
};
use crate::nn::{
    king_bucket, piece_idx, BUCKETS, BUCKET_SIZE, FP_OUT_MULTIPLIER, FP_OUT_PRECISION_BITS, H1_BIASES,
    H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, IN_TO_H1_WEIGHTS, OUT_BIASES, SCORE_SCALE,
};
use crate::pieces::P;
use crate::scores::{sanitize_eval_score, MAX_EVAL, MIN_EVAL};

type HiddenNodes = [i16; HL1_HALF_NODES];

#[derive(Clone)]
pub struct NeuralNetEval {
    hidden_nodes_white: A64<[HiddenNodes; BUCKETS]>, // white perspective
    hidden_nodes_black: A64<[HiddenNodes; BUCKETS]>, // black perspective

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
            hidden_nodes_white: A64([[0; HL1_HALF_NODES]; BUCKETS]),
            hidden_nodes_black: A64([[0; HL1_HALF_NODES]; BUCKETS]),

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

    fn update_white_pov(
        &mut self, bitboards: &BitBoards, white_bucket: usize, xor_white_pov: usize, white_offset: usize,
    ) {
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
            let prev = unsafe { self.bb_white.get_unchecked(white_bucket) }.by_piece(piece);
            let piece_offset = self.calc_wpov_piece_offset(piece);
            for pos in prev & !now {
                self.remove_piece_now_wpov(pos as usize, piece_offset);
            }
            for pos in now & !prev {
                self.add_piece_now_wpov(pos as usize, piece_offset);
            }

            let now = bitboards.by_piece(-piece);
            let prev = unsafe { self.bb_white.get_unchecked(white_bucket) }.by_piece(-piece);
            let piece_offset = self.calc_wpov_piece_offset(-piece);
            for pos in prev & !now {
                self.remove_piece_now_wpov(pos as usize, piece_offset);
            }
            for pos in now & !prev {
                self.add_piece_now_wpov(pos as usize, piece_offset);
            }
        }
    }

    fn update_black_pov(
        &mut self, bitboards: &BitBoards, black_bucket: usize, xor_black_pov: usize, black_offset: usize,
    ) {
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
            let prev = unsafe { self.bb_black.get_unchecked(black_bucket) }.by_piece(piece);
            let piece_offset = self.calc_bpov_piece_offset(piece);
            for pos in prev & !now {
                self.remove_piece_now_bpov(pos as usize, piece_offset);
            }
            for pos in now & !prev {
                self.add_piece_now_bpov(pos as usize, piece_offset);
            }

            let now = bitboards.by_piece(-piece);
            let prev = unsafe { self.bb_black.get_unchecked(black_bucket) }.by_piece(-piece);
            let piece_offset = self.calc_bpov_piece_offset(-piece);
            for pos in prev & !now {
                self.remove_piece_now_bpov(pos as usize, piece_offset);
            }
            for pos in now & !prev {
                self.add_piece_now_bpov(pos as usize, piece_offset);
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
            self.updates.push((
                self.undo,
                self.move_id,
                UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece),
            ));
        }
    }

    pub fn remove_remove_add_piece(
        &mut self, rem_pos1: usize, rem_piece1: i8, rem_pos2: usize, rem_piece2: i8, add_pos: usize, add_piece: i8,
    ) {
        if !self.fast_undo {
            self.updates.push((
                self.undo,
                self.move_id,
                UpdateAction::RemoveRemoveAdd(rem_pos1, rem_piece1, rem_pos2, rem_piece2, add_pos, add_piece),
            ));
        }
    }

    pub fn remove_add_add_piece(
        &mut self, rem_pos: usize, rem_piece: i8, add_pos1: usize, add_piece1: i8, add_pos2: usize, add_piece2: i8,
    ) {
        if !self.fast_undo {
            self.updates.push((
                self.undo,
                self.move_id,
                UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add_pos1, add_piece1, add_pos2, add_piece2),
            ));
        }
    }

    fn add_piece_now_wpov(&mut self, pos: usize, piece_offset: usize) {
        let white_pov_idx = piece_offset + (pos ^ self.xor_white_pov);
        add_weights::<HL1_HALF_NODES>(self.hidden_nodes_white_mut(), unsafe { &IN_TO_H1_WEIGHTS.0 }, white_pov_idx);
    }

    fn add_piece_now_bpov(&mut self, pos: usize, piece_offset: usize) {
        let black_pov_idx = piece_offset + (pos ^ self.xor_black_pov);
        add_weights::<HL1_HALF_NODES>(self.hidden_nodes_black_mut(), unsafe { &IN_TO_H1_WEIGHTS.0 }, black_pov_idx);
    }

    fn remove_piece_now_wpov(&mut self, pos: usize, piece_offset: usize) {
        let white_pov_idx = piece_offset + (pos ^ self.xor_white_pov);
        sub_weights::<HL1_HALF_NODES>(self.hidden_nodes_white_mut(), unsafe { &IN_TO_H1_WEIGHTS.0 }, white_pov_idx);
    }

    fn remove_piece_now_bpov(&mut self, pos: usize, piece_offset: usize) {
        let black_pov_idx = piece_offset + (pos ^ self.xor_black_pov);
        sub_weights::<HL1_HALF_NODES>(self.hidden_nodes_black_mut(), unsafe { &IN_TO_H1_WEIGHTS.0 }, black_pov_idx);
    }

    #[inline(always)]
    fn hidden_nodes_black_mut(&mut self) -> &mut HiddenNodes {
        unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) }
    }

    #[inline(always)]
    fn hidden_nodes_white_mut(&mut self) -> &mut HiddenNodes {
        unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) }
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

    #[inline(always)]
    fn calc_wpov_weight_start(&self, pos: usize, piece: i8) -> usize {
        self.calc_wpov_piece_offset(piece) + (pos ^ self.xor_white_pov)
    }

    #[inline(always)]
    fn calc_wpov_piece_offset(&self, piece: i8) -> usize {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            self.white_offset + base_index
        } else {
            self.white_offset + base_index + OPP_OFFSET
        }
    }

    #[inline(always)]
    fn calc_bpov_weight_start(&self, pos: usize, piece: i8) -> usize {
        self.calc_bpov_piece_offset(piece) + (pos ^ self.xor_black_pov)
    }

    #[inline(always)]
    fn calc_bpov_piece_offset(&self, piece: i8) -> usize {
        let idx = piece_idx(piece.unsigned_abs() as i8);

        let base_index = idx as usize * 64 * 2;
        const OPP_OFFSET: usize = 64;

        if piece > 0 {
            self.black_offset + base_index + OPP_OFFSET
        } else {
            self.black_offset + base_index
        }
    }

    pub fn eval(&mut self, active_player: Color, bitboards: &BitBoards, white_king: i8, black_king: i8) -> i16 {
        self.apply_updates(bitboards, white_king, black_king);

        let (own_hidden_nodes, opp_hidden_nodes) = if active_player.is_white() {
            (&self.hidden_nodes_white.0[self.white_bucket], &self.hidden_nodes_black.0[self.black_bucket])
        } else {
            (&self.hidden_nodes_black.0[self.black_bucket], &self.hidden_nodes_white.0[self.white_bucket])
        };

        let output = ((forward_pass(own_hidden_nodes, opp_hidden_nodes) as i64
            + (unsafe { *OUT_BIASES.0.get_unchecked(0) } as i64))
            * SCORE_SCALE as i64)
            / FP_OUT_MULTIPLIER;

        scale_eval(output as i32)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        let (white_bucket, black_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(bitboards, white_king, black_king);

        let refresh_wpov = white_bucket != self.white_bucket
            || xor_white_pov != self.xor_white_pov
            || white_offset != self.white_offset;
        let refresh_bpov = black_bucket != self.black_bucket
            || xor_black_pov != self.xor_black_pov
            || black_offset != self.black_offset;

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

                        sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_white_pov_idx,
                            add_white_pov_idx,
                        );
                        sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let (rem1_white_pov_idx, rem1_black_pov_idx) = self.calc_pov_weight_start(rem1_pos, rem1_piece);
                        let (rem2_white_pov_idx, rem2_black_pov_idx) = self.calc_pov_weight_start(rem2_pos, rem2_piece);
                        let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem1_white_pov_idx,
                            rem2_white_pov_idx,
                            add_white_pov_idx,
                        );
                        sub_sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem1_black_pov_idx,
                            rem2_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                        let (add1_white_pov_idx, add1_black_pov_idx) = self.calc_pov_weight_start(add1_pos, add1_piece);
                        let (add2_white_pov_idx, add2_black_pov_idx) = self.calc_pov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_white_pov_idx,
                            add1_white_pov_idx,
                            add2_white_pov_idx,
                        );
                        sub_add_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_black_pov_idx,
                            add1_black_pov_idx,
                            add2_black_pov_idx,
                        );
                    }
                }
            }
        } else if !refresh_wpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let rem_white_pov_idx = self.calc_wpov_weight_start(rem_pos, rem_piece);
                        let add_white_pov_idx = self.calc_wpov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_white_pov_idx,
                            add_white_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_white_pov_idx = self.calc_wpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_white_pov_idx = self.calc_wpov_weight_start(rem2_pos, rem2_piece);
                        let add_white_pov_idx = self.calc_wpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem1_white_pov_idx,
                            rem2_white_pov_idx,
                            add_white_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_white_pov_idx = self.calc_wpov_weight_start(rem_pos, rem_piece);
                        let add1_white_pov_idx = self.calc_wpov_weight_start(add1_pos, add1_piece);
                        let add2_white_pov_idx = self.calc_wpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_white.0.get_unchecked_mut(self.white_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_white_pov_idx,
                            add1_white_pov_idx,
                            add2_white_pov_idx,
                        );
                    }
                }
            }
        } else if !refresh_bpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let rem_black_pov_idx = self.calc_bpov_weight_start(rem_pos, rem_piece);
                        let add_black_pov_idx = self.calc_bpov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_black_pov_idx = self.calc_bpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_black_pov_idx = self.calc_bpov_weight_start(rem2_pos, rem2_piece);
                        let add_black_pov_idx = self.calc_bpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem1_black_pov_idx,
                            rem2_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_black_pov_idx = self.calc_bpov_weight_start(rem_pos, rem_piece);
                        let add1_black_pov_idx = self.calc_bpov_weight_start(add1_pos, add1_piece);
                        let add2_black_pov_idx = self.calc_bpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_HALF_NODES>(
                            unsafe { self.hidden_nodes_black.0.get_unchecked_mut(self.black_bucket) },
                            unsafe { &IN_TO_H1_WEIGHTS.0 },
                            rem_black_pov_idx,
                            add1_black_pov_idx,
                            add2_black_pov_idx,
                        );
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
    if score > (MAX_EVAL / 2) as i32 {
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
pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
    // H1 to ML
    let mut out_accum = zero();
    for i in (0..HL1_HALF_NODES).step_by(WORDS_PER_REG) {
        out_accum = calc_hidden_layer(out_accum, own_nodes, i, 0);
        out_accum = calc_hidden_layer(out_accum, opp_nodes, i, HL1_HALF_NODES);
    }

    horizontal_sum_32(out_accum) >> FP_OUT_PRECISION_BITS as i32
}

#[inline(always)]
pub fn calc_hidden_layer(accum: Accum, nodes: &[i16], i: usize, offset: usize) -> Accum {
    unsafe {
        let h1 = load(nodes, i);
        let h1_bias = load(&H1_BIASES.0, i + offset);
        let h1_relu = square(clipped_relu(add_epi16(h1, h1_bias)));
        let w = load(&H1_TO_OUT_WEIGHTS.0, i + offset);
        multiply_add_epi16(accum, h1_relu, w)
    }
}

#[inline(always)]
pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
    let weight_offset = weight_idx * N;
    for i in (0..N).step_by(WORDS_PER_REG) {
        let w = load(weights, weight_offset + i);
        let n = load(nodes, i);
        store(nodes, i, add_epi16(n, w));
    }
}

#[inline(always)]
pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i16], weight_idx: usize) {
    let weight_offset = weight_idx * N;
    for i in (0..N).step_by(WORDS_PER_REG) {
        let w = load(weights, weight_offset + i);
        let n = load(nodes, i);
        store(nodes, i, sub_epi16(n, w));
    }
}

#[inline(always)]
pub fn sub_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add_weight_idx: usize,
) {
    let sub_weight_offset = sub_weight_idx * N;
    let add_weight_offset = add_weight_idx * N;
    for i in (0..N).step_by(WORDS_PER_REG) {
        let sub_w = load(weights, sub_weight_offset + i);
        let add_w = load(weights, add_weight_offset + i);
        let n = load(nodes, i);
        store(nodes, i, add_epi16(sub_epi16(n, sub_w), add_w));
    }
}

#[inline(always)]
pub fn sub_sub_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i16], sub1_weight_idx: usize, sub2_weight_idx: usize, add_weight_idx: usize,
) {
    let sub1_weight_offset = sub1_weight_idx * N;
    let sub2_weight_offset = sub2_weight_idx * N;
    let add_weight_offset = add_weight_idx * N;
    for i in (0..N).step_by(WORDS_PER_REG) {
        let sub1_w = load(weights, sub1_weight_offset + i);
        let sub2_w = load(weights, sub2_weight_offset + i);
        let add_w = load(weights, add_weight_offset + i);
        let n = load(nodes, i);
        store(nodes, i, add_epi16(sub_epi16(sub_epi16(n, sub1_w), sub2_w), add_w));
    }
}

#[inline(always)]
pub fn sub_add_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i16], sub_weight_idx: usize, add1_weight_idx: usize, add2_weight_idx: usize,
) {
    let sub_weight_offset = sub_weight_idx * N;
    let add1_weight_offset = add1_weight_idx * N;
    let add2_weight_offset = add2_weight_idx * N;
    for i in (0..N).step_by(WORDS_PER_REG) {
        let sub_w = load(weights, sub_weight_offset + i);
        let add1_w = load(weights, add1_weight_offset + i);
        let add2_w = load(weights, add2_weight_offset + i);
        let n = load(nodes, i);
        store(nodes, i, add_epi16(add_epi16(sub_epi16(n, sub_w), add1_w), add2_w));
    }
}

// AVX-512
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod base {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS};
    use core::arch::x86_64::*;

    pub const WORDS_PER_REG: usize = size_of::<__m512i>() / 2;

    pub type Accum = __m512i;

    pub fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    #[inline(always)]
    pub fn horizontal_sum_32(v: __m512i) -> i32 {
        unsafe { _mm512_reduce_add_epi32(v) }
    }

    #[inline(always)]
    pub fn multiply_add_epi16(accum: __m512i, factor1: __m512i, factor2: __m512i) -> __m512i {
        unsafe {
            let mul = _mm512_madd_epi16(factor1, factor2);
            _mm512_add_epi32(accum, mul)
        }
    }

    #[inline(always)]
    pub fn clipped_relu(v: __m512i) -> __m512i {
        unsafe {
            let relu = _mm512_max_epi16(v, _mm512_setzero_si512());
            _mm512_min_epu16(relu, _mm512_set1_epi16(FP_MAX_RELU))
        }
    }

    #[inline(always)]
    pub fn square(v: __m512i) -> __m512i {
        unsafe {
            let v_scaled =
                _mm512_slli_epi16::<{ (FP_OUT_PRECISION_BITS as u32 + 16) / 2 - FP_IN_PRECISION_BITS as u32 }>(v);
            _mm512_mulhi_epu16(v_scaled, v_scaled)
        }
    }

    #[inline(always)]
    pub fn load(data: &[i16], offset: usize) -> __m512i {
        unsafe { _mm512_load_si512(data.as_ptr().add(offset).cast()) }
    }

    #[inline(always)]
    pub fn store(data: &mut [i16], offset: usize, value: __m512i) {
        unsafe { _mm512_store_si512(data.as_mut_ptr().add(offset).cast(), value) }
    }

    #[inline(always)]
    pub fn add_epi16(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn sub_epi16(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }
}

// AVX-2
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(any(target_feature = "avx512f", target_feature = "neon"))
))]
mod base {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS};
    use core::arch::x86_64::*;

    pub const WORDS_PER_REG: usize = size_of::<__m256i>() / 2;

    pub type Accum = __m256i;

    #[inline(always)]
    pub fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    #[inline(always)]
    pub fn horizontal_sum_32(v: __m256i) -> i32 {
        unsafe {
            let sum = _mm256_hadd_epi32(v, v);
            let sum = _mm256_hadd_epi32(sum, sum);
            _mm256_extract_epi32::<0>(sum) + _mm256_extract_epi32::<4>(sum)
        }
    }

    pub fn multiply_add_epi16(accum: __m256i, factor1: __m256i, factor2: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(accum, _mm256_madd_epi16(factor1, factor2)) }
    }

    #[inline(always)]
    pub fn square(v: __m256i) -> __m256i {
        unsafe {
            let v_scaled =
                _mm256_slli_epi16::<{ (FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32 }>(v);
            _mm256_mulhi_epu16(v_scaled, v_scaled)
        }
    }

    #[inline(always)]
    pub fn clipped_relu(v: __m256i) -> __m256i {
        unsafe {
            let relu = _mm256_max_epi16(v, _mm256_setzero_si256());
            _mm256_min_epu16(relu, _mm256_set1_epi16(FP_MAX_RELU))
        }
    }

    #[inline(always)]
    pub fn load(data: &[i16], offset: usize) -> __m256i {
        unsafe { _mm256_load_si256(data.as_ptr().add(offset).cast()) }
    }

    #[inline(always)]
    pub fn store(data: &mut [i16], offset: usize, value: __m256i) {
        unsafe { _mm256_store_si256(data.as_mut_ptr().add(offset).cast(), value) }
    }

    #[inline(always)]
    pub fn add_epi16(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn sub_epi16(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi16(a, b) }
    }
}

// SSE-2
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(any(target_feature = "avx2", target_feature = "avx512f", target_feature = "neon"))
))]
mod base {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS};
    use core::arch::x86_64::*;

    pub type Accum = __m128i;

    pub const WORDS_PER_REG: usize = size_of::<__m128i>() / 2;

    #[inline(always)]
    pub fn zero() -> __m128i {
        unsafe { _mm_setzero_si128() }
    }

    #[inline(always)]
    pub fn multiply_add_epi16(accum: __m128i, factor1: __m128i, factor2: __m128i) -> __m128i {
        unsafe { _mm_add_epi32(accum, _mm_madd_epi16(factor1, factor2)) }
    }

    #[inline(always)]
    pub fn clipped_relu(v: __m128i) -> __m128i {
        unsafe {
            let relu = _mm_max_epi16(v, _mm_setzero_si128());
            _mm_min_epu16(relu, _mm_set1_epi16(FP_MAX_RELU))
        }
    }

    #[inline(always)]
    pub fn square(v: __m128i) -> __m128i {
        unsafe {
            let v_scaled =
                _mm_slli_epi16::<{ (FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32 }>(v);
            _mm_mulhi_epu16(v_scaled, v_scaled)
        }
    }

    #[inline(always)]
    pub fn horizontal_sum_32(v: __m128i) -> i32 {
        unsafe {
            let sum = _mm_hadd_epi32(v, v);
            _mm_extract_epi32::<0>(sum) + _mm_extract_epi32::<1>(sum)
        }
    }

    #[inline(always)]
    pub fn load(data: &[i16], offset: usize) -> __m128i {
        unsafe { _mm_load_si128(data.as_ptr().add(offset).cast()) }
    }

    #[inline(always)]
    pub fn store(data: &mut [i16], offset: usize, value: __m128i) {
        unsafe { _mm_store_si128(data.as_mut_ptr().add(offset).cast(), value) }
    }

    #[inline(always)]
    pub fn add_epi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi16(a, b) }
    }

    #[inline(always)]
    pub fn sub_epi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_sub_epi16(a, b) }
    }
}

// NEON
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod base {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS};
    use core::arch::aarch64::*;

    pub type Accum = int32x4_t;

    pub const WORDS_PER_REG: usize = 8; // 128 bits / 16 bits = 8 i16 values per register

    #[inline(always)]
    pub fn zero() -> int32x4_t {
        unsafe { vdupq_n_s32(0) }
    }

    #[inline(always)]
    pub fn multiply_add_epi16(accum: int32x4_t, factor1: int16x8_t, factor2: int16x8_t) -> int32x4_t {
        unsafe {
            let accum = vmlal_s16(accum, vget_low_s16(factor1), vget_low_s16(factor2));
            vmlal_s16(accum, vget_high_s16(factor1), vget_high_s16(factor2))
        }
    }

    #[inline(always)]
    pub fn clipped_relu(v: int16x8_t) -> int16x8_t {
        unsafe {
            let relu = vmaxq_s16(v, vdupq_n_s16(0)); // ReLU: max(v, 0)
            vminq_s16(relu, vdupq_n_s16(FP_MAX_RELU)) // Clip at FP_MAX_RELU
        }
    }

    #[inline(always)]
    pub fn square(v: int16x8_t) -> int16x8_t {
        unsafe {
            let v_scaled =
                vshlq_n_s16(v, ((FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32) as i32);
            vshrq_n_s16(vqdmulhq_s16(v_scaled, v_scaled), 1)
        }
    }

    #[inline(always)]
    pub fn horizontal_sum_32(v: int32x4_t) -> i32 {
        unsafe {
            vaddvq_s32(v) // Sum all elements in the 128-bit register
        }
    }

    #[inline(always)]
    pub fn load(data: &[i16], offset: usize) -> int16x8_t {
        unsafe { vld1q_s16(data.as_ptr().add(offset)) }
    }

    #[inline(always)]
    pub fn store(data: &mut [i16], offset: usize, value: int16x8_t) {
        unsafe { vst1q_s16(data.as_mut_ptr().add(offset), value) }
    }

    #[inline(always)]
    pub fn add_epi16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vaddq_s16(a, b) }
    }

    #[inline(always)]
    pub fn sub_epi16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vsubq_s16(a, b) }
    }
}

// Fallback
#[cfg(not(any(
    target_feature = "sse2",
    target_feature = "avx2",
    target_feature = "neon",
    target_feature = "avx512f"
)))]
mod base {
    use crate::nn::{FP_IN_PRECISION_BITS, FP_MAX_RELU, FP_OUT_PRECISION_BITS};

    pub type Accum = i32;

    pub const WORDS_PER_REG: usize = 1;

    pub fn zero() -> i32 {
        0
    }

    #[inline(always)]
    pub fn horizontal_sum_32(v: i32) -> i32 {
        v
    }

    pub fn multiply_add_epi16(accum: i32, factor1: i16, factor2: i16) -> i32 {
        accum.wrapping_add((factor1 as i32).wrapping_mul(factor2 as i32))
    }

    #[inline(always)]
    pub fn clipped_relu(v: i16) -> i16 {
        v.clamp(0, FP_MAX_RELU)
    }

    #[inline(always)]
    pub fn square(v: i16) -> i16 {
        let v_scaled = (v << ((FP_OUT_PRECISION_BITS as i32 + 16) / 2 - FP_IN_PRECISION_BITS as i32)) as i32;
        (v_scaled * v_scaled).wrapping_shr(16) as i16
    }

    #[inline(always)]
    pub fn load(data: &[i16], offset: usize) -> i16 {
        unsafe { *data.get_unchecked(offset) }
    }

    #[inline(always)]
    pub fn store(data: &mut [i16], offset: usize, value: i16) {
        unsafe { *data.get_unchecked_mut(offset) = value }
    }

    #[inline(always)]
    pub fn add_epi16(a: i16, b: i16) -> i16 {
        a.wrapping_add(b)
    }

    #[inline(always)]
    pub fn sub_epi16(a: i16, b: i16) -> i16 {
        a.wrapping_sub(b)
    }
}
