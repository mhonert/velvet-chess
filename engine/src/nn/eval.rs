/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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
use crate::bitboard::{BitBoards};
use crate::colors::Color;
use crate::nn::eval::base::{add_epi16, load_i8, load_i16, store_i16, sub_epi16, VALUES_PER_REG, forward_pass};
use crate::nn::{king_bucket, piece_idx, BUCKETS, BUCKET_SIZE, FP_OUT_MULTIPLIER, HL1_NODES, IN_TO_H1_WEIGHTS, OUT_BIASES, SCORE_SCALE};
use crate::scores::{sanitize_eval_score, MAX_EVAL, MIN_EVAL};
use crate::slices::SliceElementAccess;

type HiddenNodes = [i16; HL1_NODES];

#[derive(Clone)]
pub struct NeuralNetEval {
    hidden_nodes_white: A64<[HiddenNodes; BUCKETS * 2]>, // white perspective
    hidden_nodes_black: A64<[HiddenNodes; BUCKETS * 2]>, // black perspective

    bb_white: [BitBoards; BUCKETS * 2],
    bb_black: [BitBoards; BUCKETS * 2],

    white_acc_bucket: usize,
    black_acc_bucket: usize,

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
            hidden_nodes_white: A64([[0; HL1_NODES]; BUCKETS * 2]),
            hidden_nodes_black: A64([[0; HL1_NODES]; BUCKETS * 2]),

            bb_white: [BitBoards::default(); BUCKETS * 2],
            bb_black: [BitBoards::default(); BUCKETS * 2],

            white_acc_bucket: 0,
            black_acc_bucket: 0,

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
        for buckets in 0..(BUCKETS * 2) {
            self.hidden_nodes_white.0[buckets].fill(0);
            self.hidden_nodes_black.0[buckets].fill(0);
            self.bb_white[buckets] = BitBoards::default();
            self.bb_black[buckets] = BitBoards::default();
        }

        let (white_acc_bucket, black_acc_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(white_king, black_king);

        self.update_white_pov(bitboards, white_acc_bucket, xor_white_pov, white_offset);
        self.update_black_pov(bitboards, black_acc_bucket, xor_black_pov, black_offset);
    }

    fn update_white_pov(
        &mut self, bitboards: &BitBoards, white_acc_bucket: usize, xor_white_pov: usize, white_offset: usize,
    ) {
        self.white_acc_bucket = white_acc_bucket;
        self.xor_white_pov = xor_white_pov;
        self.white_offset = white_offset;

        let full_refresh_cost = bitboards.occupancy().piece_count();
        let delta_sub = self.bb_white[white_acc_bucket].occupancy() & !bitboards.occupancy();
        let delta_add = bitboards.occupancy() & !self.bb_white[white_acc_bucket].occupancy();
        let delta_cost = delta_sub.piece_count() + delta_add.piece_count();

        if delta_cost < full_refresh_cost {
            for piece in 1..=6 {
                let now = bitboards.by_piece(piece);
                let prev = self.bb_white.el(white_acc_bucket).by_piece(piece);
                let piece_offset = self.calc_wpov_piece_offset(piece);
                for pos in prev & !now {
                    self.remove_piece_now_wpov(pos as usize, piece_offset);
                }
                for pos in now & !prev {
                    self.add_piece_now_wpov(pos as usize, piece_offset);
                }

                let now = bitboards.by_piece(-piece);
                let prev = self.bb_white.el(white_acc_bucket).by_piece(-piece);
                let piece_offset = self.calc_wpov_piece_offset(-piece);
                for pos in prev & !now {
                    self.remove_piece_now_wpov(pos as usize, piece_offset);
                }
                for pos in now & !prev {
                    self.add_piece_now_wpov(pos as usize, piece_offset);
                }
            }
        } else {
            self.hidden_nodes_white.0[white_acc_bucket].fill(0);
            for piece in 1..=6 {
                let now = bitboards.by_piece(piece);
                let piece_offset = self.calc_wpov_piece_offset(piece);
                for pos in now {
                    self.add_piece_now_wpov(pos as usize, piece_offset);
                }

                let now = bitboards.by_piece(-piece);
                let piece_offset = self.calc_wpov_piece_offset(-piece);
                for pos in now {
                    self.add_piece_now_wpov(pos as usize, piece_offset);
                }
            }
        }

        self.bb_white[white_acc_bucket] = *bitboards;
    }

    fn update_black_pov(
        &mut self, bitboards: &BitBoards, black_acc_bucket: usize, xor_black_pov: usize, black_offset: usize,
    ) {
        self.black_acc_bucket = black_acc_bucket;
        self.xor_black_pov = xor_black_pov;
        self.black_offset = black_offset;

        let full_refresh_cost = bitboards.occupancy().piece_count();
        let delta_sub = self.bb_black[black_acc_bucket].occupancy() & !bitboards.occupancy();
        let delta_add = bitboards.occupancy() & !self.bb_black[black_acc_bucket].occupancy();
        let delta_cost = delta_sub.piece_count() + delta_add.piece_count();

        if delta_cost < full_refresh_cost {
            for piece in 1..=6 {
                let now = bitboards.by_piece(piece);
                let prev = self.bb_black.el(black_acc_bucket).by_piece(piece);
                let piece_offset = self.calc_bpov_piece_offset(piece);
                for pos in prev & !now {
                    self.remove_piece_now_bpov(pos as usize, piece_offset);
                }
                for pos in now & !prev {
                    self.add_piece_now_bpov(pos as usize, piece_offset);
                }

                let now = bitboards.by_piece(-piece);
                let prev = self.bb_black.el(black_acc_bucket).by_piece(-piece);
                let piece_offset = self.calc_bpov_piece_offset(-piece);
                for pos in prev & !now {
                    self.remove_piece_now_bpov(pos as usize, piece_offset);
                }
                for pos in now & !prev {
                    self.add_piece_now_bpov(pos as usize, piece_offset);
                }
            }
        } else {
            self.hidden_nodes_black.0[black_acc_bucket].fill(0);
            for piece in 1..=6 {
                let now = bitboards.by_piece(piece);
                let piece_offset = self.calc_bpov_piece_offset(piece);
                for pos in now {
                    self.add_piece_now_bpov(pos as usize, piece_offset);
                }
                let now = bitboards.by_piece(-piece);
                let piece_offset = self.calc_bpov_piece_offset(-piece);
                for pos in now {
                    self.add_piece_now_bpov(pos as usize, piece_offset);
                }
            }
        }

        self.bb_black[black_acc_bucket] = *bitboards;
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
        add_weights::<HL1_NODES>(self.hidden_nodes_white_mut(), &IN_TO_H1_WEIGHTS.0, white_pov_idx);
    }

    fn add_piece_now_bpov(&mut self, pos: usize, piece_offset: usize) {
        let black_pov_idx = piece_offset + (pos ^ self.xor_black_pov);
        add_weights::<HL1_NODES>(self.hidden_nodes_black_mut(), &IN_TO_H1_WEIGHTS.0, black_pov_idx);
    }

    fn remove_piece_now_wpov(&mut self, pos: usize, piece_offset: usize) {
        let white_pov_idx = piece_offset + (pos ^ self.xor_white_pov);
        sub_weights::<HL1_NODES>(self.hidden_nodes_white_mut(), &IN_TO_H1_WEIGHTS.0, white_pov_idx);
    }

    fn remove_piece_now_bpov(&mut self, pos: usize, piece_offset: usize) {
        let black_pov_idx = piece_offset + (pos ^ self.xor_black_pov);
        sub_weights::<HL1_NODES>(self.hidden_nodes_black_mut(), &IN_TO_H1_WEIGHTS.0, black_pov_idx);
    }

    fn hidden_nodes_black_mut(&mut self) -> &mut HiddenNodes {
        self.hidden_nodes_black.0.el_mut(self.black_acc_bucket)
    }

    fn hidden_nodes_white_mut(&mut self) -> &mut HiddenNodes {
        self.hidden_nodes_white.0.el_mut(self.white_acc_bucket)
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
        self.calc_wpov_piece_offset(piece) + (pos ^ self.xor_white_pov)
    }

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

    fn calc_bpov_weight_start(&self, pos: usize, piece: i8) -> usize {
        self.calc_bpov_piece_offset(piece) + (pos ^ self.xor_black_pov)
    }

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
            (&self.hidden_nodes_white.0[self.white_acc_bucket], &self.hidden_nodes_black.0[self.black_acc_bucket])
        } else {
            (&self.hidden_nodes_black.0[self.black_acc_bucket], &self.hidden_nodes_white.0[self.white_acc_bucket])
        };

        let raw_output = forward_pass(own_hidden_nodes, opp_hidden_nodes) as i64;
        let output = (raw_output
            + (*OUT_BIASES.0.el(0) as i64 * FP_OUT_MULTIPLIER * FP_OUT_MULTIPLIER))
            / (FP_OUT_MULTIPLIER * FP_OUT_MULTIPLIER * FP_OUT_MULTIPLIER / SCORE_SCALE as i64);

        scale_eval(output as i32)
    }

    fn apply_updates(&mut self, bitboards: &BitBoards, white_king: i8, black_king: i8) {
        let (white_acc_bucket, black_acc_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset) =
            calc_bucket_offsets(white_king, black_king);

        let refresh_wpov = white_acc_bucket != self.white_acc_bucket;
        let refresh_bpov = black_acc_bucket != self.black_acc_bucket;

        if refresh_wpov {
            self.update_white_pov(bitboards, white_acc_bucket, xor_white_pov, white_offset);
        }

        if refresh_bpov {
            self.update_black_pov(bitboards, black_acc_bucket, xor_black_pov, black_offset);
        }

        if !refresh_wpov && !refresh_bpov {
            for (_, _, update) in self.updates.iter() {
                match *update {
                    UpdateAction::RemoveAdd(rem_pos, rem_piece, add_pos, add_piece) => {
                        let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                        let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                        sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_white_pov_idx,
                            add_white_pov_idx,
                        );

                        sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let (rem1_white_pov_idx, rem1_black_pov_idx) = self.calc_pov_weight_start(rem1_pos, rem1_piece);
                        let (rem2_white_pov_idx, rem2_black_pov_idx) = self.calc_pov_weight_start(rem2_pos, rem2_piece);
                        let (add_white_pov_idx, add_black_pov_idx) = self.calc_pov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem1_white_pov_idx,
                            rem2_white_pov_idx,
                            add_white_pov_idx,
                        );

                        sub_sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem1_black_pov_idx,
                            rem2_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let (rem_white_pov_idx, rem_black_pov_idx) = self.calc_pov_weight_start(rem_pos, rem_piece);
                        let (add1_white_pov_idx, add1_black_pov_idx) = self.calc_pov_weight_start(add1_pos, add1_piece);
                        let (add2_white_pov_idx, add2_black_pov_idx) = self.calc_pov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_white_pov_idx,
                            add1_white_pov_idx,
                            add2_white_pov_idx,
                        );

                        sub_add_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
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

                        sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_white_pov_idx,
                            add_white_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_white_pov_idx = self.calc_wpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_white_pov_idx = self.calc_wpov_weight_start(rem2_pos, rem2_piece);
                        let add_white_pov_idx = self.calc_wpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem1_white_pov_idx,
                            rem2_white_pov_idx,
                            add_white_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_white_pov_idx = self.calc_wpov_weight_start(rem_pos, rem_piece);
                        let add1_white_pov_idx = self.calc_wpov_weight_start(add1_pos, add1_piece);
                        let add2_white_pov_idx = self.calc_wpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_NODES>(
                            self.hidden_nodes_white.0.el_mut(self.white_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
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

                        sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveRemoveAdd(rem1_pos, rem1_piece, rem2_pos, rem2_piece, add_pos, add_piece) => {
                        let rem1_black_pov_idx = self.calc_bpov_weight_start(rem1_pos, rem1_piece);
                        let rem2_black_pov_idx = self.calc_bpov_weight_start(rem2_pos, rem2_piece);
                        let add_black_pov_idx = self.calc_bpov_weight_start(add_pos, add_piece);

                        sub_sub_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem1_black_pov_idx,
                            rem2_black_pov_idx,
                            add_black_pov_idx,
                        );
                    }

                    UpdateAction::RemoveAddAdd(rem_pos, rem_piece, add1_pos, add1_piece, add2_pos, add2_piece) => {
                        let rem_black_pov_idx = self.calc_bpov_weight_start(rem_pos, rem_piece);
                        let add1_black_pov_idx = self.calc_bpov_weight_start(add1_pos, add1_piece);
                        let add2_black_pov_idx = self.calc_bpov_weight_start(add2_pos, add2_piece);

                        sub_add_add_weights::<HL1_NODES>(
                            self.hidden_nodes_black.0.el_mut(self.black_acc_bucket),
                            &IN_TO_H1_WEIGHTS.0,
                            rem_black_pov_idx,
                            add1_black_pov_idx,
                            add2_black_pov_idx,
                        );
                    }
                }
            }
        }
        self.bb_white[self.white_acc_bucket] = *bitboards;
        self.bb_black[self.black_acc_bucket] = *bitboards;
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

fn calc_bucket_offsets(mut white_king: i8, mut black_king: i8) -> ( usize, usize, usize, usize, usize, usize) {
    let white_king_col = white_king & 7;
    let black_king_col = black_king & 7;

    let mirror_white_pov = white_king_col > 3;
    let mirror_black_pov = black_king_col > 3;

    let mut xor_white_pov = 0;
    let mut xor_black_pov = 56;
    let mut w_acc_bucket = 0;
    let mut b_acc_bucket = 0;
    if mirror_white_pov {
        xor_white_pov |= 7;
        w_acc_bucket = BUCKETS;
    }

    if mirror_black_pov {
        xor_black_pov |= 7;
        b_acc_bucket = BUCKETS;
    }

    white_king ^= xor_white_pov as i8;
    black_king ^= xor_black_pov as i8;

    let w_nn_bucket = king_bucket(white_king as u16) as usize;
    let b_nn_bucket = king_bucket(black_king as u16) as usize;

    w_acc_bucket += w_nn_bucket;
    b_acc_bucket += b_nn_bucket;
    const BASE_OFFSET: usize = 0;
    let (white_offset, black_offset) = (BASE_OFFSET + w_nn_bucket * BUCKET_SIZE, BASE_OFFSET + b_nn_bucket * BUCKET_SIZE);

    (w_acc_bucket, b_acc_bucket, xor_white_pov, xor_black_pov, white_offset, black_offset)
}

pub fn add_weights<const N: usize>(nodes: &mut [i16], weights: &[i8], weight_idx: usize) {
    let weight_offset = weight_idx * N;
    for i in (0..N).step_by(VALUES_PER_REG) {
        let w = load_i8(weights, weight_offset + i);
        let n = load_i16(nodes, i);
        store_i16(nodes, i, add_epi16(n, w));
    }
}

pub fn sub_weights<const N: usize>(nodes: &mut [i16], weights: &[i8], weight_idx: usize) {
    let weight_offset = weight_idx * N;
    for i in (0..N).step_by(VALUES_PER_REG) {
        let w = load_i8(weights, weight_offset + i);
        let n = load_i16(nodes, i);
        store_i16(nodes, i, sub_epi16(n, w));
    }
}

pub fn sub_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i8], sub_weight_idx: usize, add_weight_idx: usize,
) {
    let sub_weight_offset = sub_weight_idx * N;
    let add_weight_offset = add_weight_idx * N;
    for i in (0..N).step_by(VALUES_PER_REG) {
        let sub_w = load_i8(weights, sub_weight_offset + i);
        let add_w = load_i8(weights, add_weight_offset + i);
        let n = load_i16(nodes, i);
        store_i16(nodes, i, add_epi16(sub_epi16(n, sub_w), add_w));
    }
}

pub fn sub_sub_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i8], sub1_weight_idx: usize, sub2_weight_idx: usize, add_weight_idx: usize,
) {
    let sub1_weight_offset = sub1_weight_idx * N;
    let sub2_weight_offset = sub2_weight_idx * N;
    let add_weight_offset = add_weight_idx * N;
    for i in (0..N).step_by(VALUES_PER_REG) {
        let sub1_w = load_i8(weights, sub1_weight_offset + i);
        let sub2_w = load_i8(weights, sub2_weight_offset + i);
        let add_w = load_i8(weights, add_weight_offset + i);
        let n = load_i16(nodes, i);
        store_i16(nodes, i, add_epi16(sub_epi16(sub_epi16(n, sub1_w), sub2_w), add_w));
    }
}

pub fn sub_add_add_weights<const N: usize>(
    nodes: &mut [i16], weights: &[i8], sub_weight_idx: usize, add1_weight_idx: usize, add2_weight_idx: usize,
) {
    let sub_weight_offset = sub_weight_idx * N;
    let add1_weight_offset = add1_weight_idx * N;
    let add2_weight_offset = add2_weight_idx * N;
    for i in (0..N).step_by(VALUES_PER_REG) {
        let sub_w = load_i8(weights, sub_weight_offset + i);
        let add1_w = load_i8(weights, add1_weight_offset + i);
        let add2_w = load_i8(weights, add2_weight_offset + i);
        let n = load_i16(nodes, i);
        store_i16(nodes, i, add_epi16(add_epi16(sub_epi16(n, sub_w), add1_w), add2_w));
    }
}

// AVX-512
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", feature="avx512"))]
mod base {
    use core::arch::x86_64::*;
    use crate::nn::{H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, HL1_NODES};

    pub const VALUES_PER_REG: usize = size_of::<__m512i>() / 2;

    fn zero() -> __m512i {
        unsafe { _mm512_setzero_si512() }
    }

    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        let mut out_accum = zero();
        let mut wi = 0;
        for ni in (0..HL1_NODES).step_by(VALUES_PER_REG * 2) {
            let v = add_epi32(
                calc_hidden_layer(own_nodes, ni, wi, 0),
                calc_hidden_layer(opp_nodes, ni, wi + HL1_HALF_NODES, HL1_NODES),
            );
            out_accum = add_epi32(out_accum, v);
            wi += VALUES_PER_REG;
        }

        horizontal_sum_32(out_accum)
    }

    #[inline(always)]
    fn calc_hidden_layer(nodes: &[i16], ni: usize, wi: usize, offset: usize) -> __m512i {
        let h1a = load_i16(nodes, ni);
        let h1b = load_i16(nodes, ni + VALUES_PER_REG);

        let h1a_bias = load_i8(&H1_BIASES.0, ni + offset);
        let h1b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG);

        let w1 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi);

        let h1a_relu = clipped_relu::<255>(add_epi16(h1a, h1a_bias));
        let h1b_relu = clipped_relu::<255>(add_epi16(h1b, h1b_bias));

        let w1_x_h1a = unsafe { _mm512_mullo_epi16(w1, h1a_relu) };

        madd_epi16(w1_x_h1a, h1b_relu)
    }

    #[inline(always)]
    fn clipped_relu<const C: i16>(v: __m512i) -> __m512i {
        unsafe {
            let relu = _mm512_max_epi16(v, _mm512_setzero_si512());
            _mm512_min_epu16(relu, _mm512_set1_epi16(C))
        }
    }

    fn horizontal_sum_32(v: __m512i) -> i32 {
        unsafe { _mm512_reduce_add_epi32(v) }
    }


    pub fn load_i16(data: &[i16], offset: usize) -> __m512i {
        unsafe { _mm512_load_si512(data.as_ptr().add(offset).cast()) }
    }

    pub fn load_i8(data: &[i8], offset: usize) -> __m512i {
        unsafe { _mm512_cvtepi8_epi16(_mm256_load_si256(data.as_ptr().add(offset).cast())) }
    }

    pub fn store_i16(data: &mut [i16], offset: usize, value: __m512i) {
        unsafe { _mm512_store_si512(data.as_mut_ptr().add(offset).cast(), value) }
    }

    pub fn add_epi16(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi16(a, b) }
    }

    pub fn sub_epi16(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_sub_epi16(a, b) }
    }

    fn add_epi32(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_add_epi32(a, b) }
    }

    fn madd_epi16(a: __m512i, b: __m512i) -> __m512i {
        unsafe { _mm512_madd_epi16(a, b) }
    }
}

// AVX-2
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(any(all(target_feature = "avx512f", feature="avx512"), target_feature = "neon"))
))]
mod base {
    use crate::nn::{H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, HL1_NODES};
    use core::arch::x86_64::*;

    pub const VALUES_PER_REG: usize = size_of::<__m256i>() / 2;

    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        let mut out_accum = zero();
        let mut wi = 0;
        for ni in (0..HL1_NODES).step_by(VALUES_PER_REG * 4) {
            let v = add_epi32(
                calc_hidden_layer(own_nodes, ni, wi, 0),
                calc_hidden_layer(opp_nodes, ni, wi + HL1_HALF_NODES, HL1_NODES),
            );
            out_accum = add_epi32(out_accum, v);
            wi += VALUES_PER_REG * 2;
        }

        horizontal_sum_32(out_accum)
    }

    #[inline(always)]
    fn calc_hidden_layer(nodes: &[i16], ni: usize, wi: usize, offset: usize) -> __m256i {
        let h1a = load_i16(nodes, ni);
        let h2a = load_i16(nodes, ni + VALUES_PER_REG);
        let h1b = load_i16(nodes, ni + VALUES_PER_REG * 2);
        let h2b = load_i16(nodes, ni + VALUES_PER_REG * 3);

        let h1a_bias = load_i8(&H1_BIASES.0, ni + offset);
        let h2a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG);
        let h1b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 2);
        let h2b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 3);

        let w1 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi);
        let w2 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG);

        let h1a_relu = clipped_relu::<255>(add_epi16(h1a, h1a_bias));
        let h2a_relu = clipped_relu::<255>(add_epi16(h2a, h2a_bias));

        let h1b_relu = clipped_relu::<255>(add_epi16(h1b, h1b_bias));
        let h2b_relu = clipped_relu::<255>(add_epi16(h2b, h2b_bias));

        let w1_x_h1a = unsafe { _mm256_mullo_epi16(w1, h1a_relu) };
        let w2_x_h2a = unsafe { _mm256_mullo_epi16(w2, h2a_relu) };

        add_epi32(
            madd_epi16(w1_x_h1a, h1b_relu),
            madd_epi16(w2_x_h2a, h2b_relu),
        )
    }

    fn horizontal_sum_32(v: __m256i) -> i32 {
        unsafe {
            let sum = _mm256_hadd_epi32(v, v);
            let sum = _mm256_hadd_epi32(sum, sum);
            _mm256_extract_epi32::<0>(sum) + _mm256_extract_epi32::<4>(sum)
        }
    }

    fn zero() -> __m256i {
        unsafe { _mm256_setzero_si256() }
    }

    fn clipped_relu<const C: i16>(v: __m256i) -> __m256i {
        unsafe {
            let relu = _mm256_max_epi16(v, _mm256_setzero_si256());
            _mm256_min_epu16(relu, _mm256_set1_epi16(C))
        }
    }
    
    pub fn load_i8(data: &[i8], offset: usize) -> __m256i {
        unsafe { _mm256_cvtepi8_epi16(_mm_load_si128(data.as_ptr().add(offset).cast())) }
    }

    pub fn load_i16(data: &[i16], offset: usize) -> __m256i {
        unsafe { _mm256_load_si256(data.as_ptr().add(offset).cast()) }
    }

    pub fn store_i16(data: &mut [i16], offset: usize, value: __m256i) {
        unsafe { _mm256_store_si256(data.as_mut_ptr().add(offset).cast(), value) }
    }

    pub fn add_epi16(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi16(a, b) }
    }

    pub fn sub_epi16(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_sub_epi16(a, b) }
    }

    fn add_epi32(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_add_epi32(a, b) }
    }

    fn madd_epi16(a: __m256i, b: __m256i) -> __m256i {
        unsafe { _mm256_madd_epi16(a, b) }
    }
}

// SSE-4.1
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse4.1",
    not(any(target_feature = "avx2", target_feature = "avx512f", target_feature = "neon"))
))]
mod base {
    use crate::nn::{H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, HL1_NODES};
    use core::arch::x86_64::*;

    pub const VALUES_PER_REG: usize = size_of::<__m128i>() / 2;

    fn zero() -> __m128i {
        unsafe { _mm_setzero_si128() }
    }

    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        let mut out_accum = zero();
        let mut wi = 0;
        for ni in (0..HL1_NODES).step_by(VALUES_PER_REG * 8) {
            let v = add_epi32(
                calc_hidden_layer(own_nodes, ni, wi, 0),
                calc_hidden_layer(opp_nodes, ni, wi + HL1_HALF_NODES, HL1_NODES),
            );
            out_accum = add_epi32(out_accum, v);
            wi += VALUES_PER_REG * 4;
        }

        horizontal_sum_32(out_accum)
    }

    #[inline(always)]
    fn calc_hidden_layer(nodes: &[i16], ni: usize, wi: usize, offset: usize) -> __m128i {
        let h1a = load_i16(nodes, ni);
        let h2a = load_i16(nodes, ni + VALUES_PER_REG);
        let h3a = load_i16(nodes, ni + VALUES_PER_REG * 2);
        let h4a = load_i16(nodes, ni + VALUES_PER_REG * 3);
        let h1b = load_i16(nodes, ni + VALUES_PER_REG * 4);
        let h2b = load_i16(nodes, ni + VALUES_PER_REG * 5);
        let h3b = load_i16(nodes, ni + VALUES_PER_REG * 6);
        let h4b = load_i16(nodes, ni + VALUES_PER_REG * 7);

        let h1a_bias = load_i8(&H1_BIASES.0, ni + offset);
        let h2a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG);
        let h3a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 2);
        let h4a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 3);
        let h1b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 4);
        let h2b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 5);
        let h3b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 6);
        let h4b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 7);

        let w1 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi);
        let w2 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG);
        let w3 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG * 2);
        let w4 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG * 3);

        let h1a_relu = clipped_relu::<255>(add_epi16(h1a, h1a_bias));
        let h2a_relu = clipped_relu::<255>(add_epi16(h2a, h2a_bias));
        let h3a_relu = clipped_relu::<255>(add_epi16(h3a, h3a_bias));
        let h4a_relu = clipped_relu::<255>(add_epi16(h4a, h4a_bias));

        let h1b_relu = clipped_relu::<255>(add_epi16(h1b, h1b_bias));
        let h2b_relu = clipped_relu::<255>(add_epi16(h2b, h2b_bias));
        let h3b_relu = clipped_relu::<255>(add_epi16(h3b, h3b_bias));
        let h4b_relu = clipped_relu::<255>(add_epi16(h4b, h4b_bias));

        let w1_x_h1a = unsafe { _mm_mullo_epi16(w1, h1a_relu) };
        let w2_x_h2a = unsafe { _mm_mullo_epi16(w2, h2a_relu) };
        let w3_x_h3a = unsafe { _mm_mullo_epi16(w3, h3a_relu) };
        let w4_x_h4a = unsafe { _mm_mullo_epi16(w4, h4a_relu) };

        add_epi32(
            add_epi32(
                madd_epi16(w1_x_h1a, h1b_relu),
                madd_epi16(w2_x_h2a, h2b_relu),
            ),
            add_epi32(
                madd_epi16(w3_x_h3a, h3b_relu),
                madd_epi16(w4_x_h4a, h4b_relu),
            ),
        )
    }

    fn clipped_relu<const C: i16>(v: __m128i) -> __m128i {
        unsafe {
            let relu = _mm_max_epi16(v, _mm_setzero_si128());
            _mm_min_epu16(relu, _mm_set1_epi16(C))
        }
    }

    fn horizontal_sum_32(v: __m128i) -> i32 {
        unsafe {
            let sum = _mm_hadd_epi32(v, v);
            _mm_extract_epi32::<0>(sum) + _mm_extract_epi32::<1>(sum)
        }
    }

    pub fn load_i8(data: &[i8], offset: usize) -> __m128i {
        unsafe { _mm_cvtepi8_epi16(_mm_loadl_epi64(data.as_ptr().add(offset).cast())) }
    }

    pub fn load_i16(data: &[i16], offset: usize) -> __m128i {
        unsafe { _mm_load_si128(data.as_ptr().add(offset).cast()) }
    }

    pub fn store_i16(data: &mut [i16], offset: usize, value: __m128i) {
        unsafe { _mm_store_si128(data.as_mut_ptr().add(offset).cast(), value) }
    }

    pub fn add_epi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi16(a, b) }
    }

    pub fn sub_epi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_sub_epi16(a, b) }
    }

    fn add_epi32(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_add_epi32(a, b) }
    }

    fn madd_epi16(a: __m128i, b: __m128i) -> __m128i {
        unsafe { _mm_madd_epi16(a, b) }
    }
}

// NEON
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod base {
    use core::arch::aarch64::*;
    use crate::nn::{H1_BIASES, H1_TO_OUT_WEIGHTS, HL1_HALF_NODES, HL1_NODES};

    pub const VALUES_PER_REG: usize = 8; // 128 bits / 16 bits = 8 i16 values per register

    fn zero() -> int32x4_t {
        unsafe { vdupq_n_s32(0) }
    }

    pub fn forward_pass(own_nodes: &[i16], opp_nodes: &[i16]) -> i32 {
        let mut out_accum = zero();
        let mut wi = 0;
        for ni in (0..HL1_NODES).step_by(VALUES_PER_REG * 8) {
            let v = add_epi32(
                calc_hidden_layer(own_nodes, ni, wi, 0),
                calc_hidden_layer(opp_nodes, ni, wi + HL1_HALF_NODES, HL1_NODES),
            );
            out_accum = add_epi32(out_accum, v);
            wi += VALUES_PER_REG * 4;
        }

        horizontal_sum_32(out_accum)
    }

    #[inline(always)]
    fn calc_hidden_layer(nodes: &[i16], ni: usize, wi: usize, offset: usize) -> int32x4_t {
        let h1a = load_i16(nodes, ni);
        let h2a = load_i16(nodes, ni + VALUES_PER_REG);
        let h3a = load_i16(nodes, ni + VALUES_PER_REG * 2);
        let h4a = load_i16(nodes, ni + VALUES_PER_REG * 3);
        let h1b = load_i16(nodes, ni + VALUES_PER_REG * 4);
        let h2b = load_i16(nodes, ni + VALUES_PER_REG * 5);
        let h3b = load_i16(nodes, ni + VALUES_PER_REG * 6);
        let h4b = load_i16(nodes, ni + VALUES_PER_REG * 7);

        let h1a_bias = load_i8(&H1_BIASES.0, ni + offset);
        let h2a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG);
        let h3a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 2);
        let h4a_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 3);
        let h1b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 4);
        let h2b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 5);
        let h3b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 6);
        let h4b_bias = load_i8(&H1_BIASES.0, ni + offset + VALUES_PER_REG * 7);

        let w1 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi);
        let w2 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG);
        let w3 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG * 2);
        let w4 = load_i8(&H1_TO_OUT_WEIGHTS.0, wi + VALUES_PER_REG * 3);

        let h1a_relu = clipped_relu::<255>(add_epi16(h1a, h1a_bias));
        let h2a_relu = clipped_relu::<255>(add_epi16(h2a, h2a_bias));
        let h3a_relu = clipped_relu::<255>(add_epi16(h3a, h3a_bias));
        let h4a_relu = clipped_relu::<255>(add_epi16(h4a, h4a_bias));

        let h1b_relu = clipped_relu::<255>(add_epi16(h1b, h1b_bias));
        let h2b_relu = clipped_relu::<255>(add_epi16(h2b, h2b_bias));
        let h3b_relu = clipped_relu::<255>(add_epi16(h3b, h3b_bias));
        let h4b_relu = clipped_relu::<255>(add_epi16(h4b, h4b_bias));

        let w1_x_h1a = unsafe { vmulq_s16(w1, h1a_relu) };
        let w2_x_h2a = unsafe { vmulq_s16(w2, h2a_relu) };
        let w3_x_h3a = unsafe { vmulq_s16(w3, h3a_relu) };
        let w4_x_h4a = unsafe { vmulq_s16(w4, h4a_relu) };

        add_epi32(
            add_epi32(
                madd_epi16(w1_x_h1a, h1b_relu),
                madd_epi16(w2_x_h2a, h2b_relu),
            ),
            add_epi32(
                madd_epi16(w3_x_h3a, h3b_relu),
                madd_epi16(w4_x_h4a, h4b_relu),
            ),
        )
    }

    fn clipped_relu<const C: i16>(v: int16x8_t) -> int16x8_t {
        unsafe {
            let relu = vmaxq_s16(v, vdupq_n_s16(0));
            vminq_s16(relu, vdupq_n_s16(C))
        }
    }

    pub fn horizontal_sum_32(v: int32x4_t) -> i32 {
        unsafe {
            vaddvq_s32(v)
        }
    }

    pub fn load_i8(data: &[i8], offset: usize) -> int16x8_t {
        unsafe { vmovl_s8(vld1_s8(data.as_ptr().add(offset))) }
    }

    pub fn load_i16(data: &[i16], offset: usize) -> int16x8_t {
        unsafe { vld1q_s16(data.as_ptr().add(offset)) }
    }

    pub fn store_i16(data: &mut [i16], offset: usize, value: int16x8_t) {
        unsafe { vst1q_s16(data.as_mut_ptr().add(offset), value) }
    }

    pub fn add_epi16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vaddq_s16(a, b) }
    }

    pub fn sub_epi16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
        unsafe { vsubq_s16(a, b) }
    }

    fn add_epi32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
        unsafe { vaddq_s32(a, b) }
    }

    fn madd_epi16(a: int16x8_t, b: int16x8_t) -> int32x4_t {
        unsafe {
            let lo = vmull_s16(vget_low_s16(a), vget_low_s16(b));
            let hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
            vpaddq_s32(lo, hi)
        }
    }
}
