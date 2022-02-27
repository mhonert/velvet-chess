/*
 * Velvet Chess Engine
 * Copyright (C) 2021 mhonert (https://github.com/mhonert)
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

use std::collections::HashSet;
use std::sync::mpsc::Receiver;
use velvet::engine::Message;
use velvet::fen::write_fen;
use velvet::moves::{Move, NO_MOVE};
use velvet::pieces::{B, N, P, Q, R};
use velvet::search::{Search};

// Code for generating quiet training positions for tuning and NN training
pub trait GenQuietPos {
    fn is_quiet_pv(&mut self, pv: &[Move], base_mat_score: i32) -> bool;
    fn material_score(&self) -> i32;
    fn piece_count(&self) -> i32;
    fn eval_pv_end_pos(&mut self, rx: Option<&Receiver<Message>>, duplicate_check: &mut HashSet<u64>, pv: &[Move], ply: u16) -> Option<(String, u16, i32)>;
}

impl GenQuietPos for Search {
    fn is_quiet_pv(&mut self, pv: &[Move], base_piece_count: i32) -> bool {
        if let Some((m, rest_pv)) = pv.split_first() {
            let (previous_piece, move_state) = self.board.perform_move(*m);
            let is_quiet = move_state == 0 && m.is_quiet() && !self.board.is_in_check(self.board.active_player()) && self.is_quiet_pv(rest_pv, base_piece_count);
            self.board.undo_move(*m, previous_piece, move_state);

            is_quiet

        } else {
            self.piece_count() == base_piece_count
        }
    }

    fn material_score(&self) -> i32 {
        (self.board.get_bitboard(P).count_ones() as i32 - self.board.get_bitboard(-P).count_ones() as i32) * 100 +
            (self.board.get_bitboard(N).count_ones() as i32 - self.board.get_bitboard(-N).count_ones() as i32) * 300 +
            (self.board.get_bitboard(B).count_ones() as i32 - self.board.get_bitboard(-B).count_ones() as i32) * 330 +
            (self.board.get_bitboard(R).count_ones() as i32 - self.board.get_bitboard(-R).count_ones() as i32) * 550 +
            (self.board.get_bitboard(Q).count_ones() as i32 - self.board.get_bitboard(-Q).count_ones() as i32) * 990
    }

    fn piece_count(&self) -> i32 {
        self.board.get_occupancy_bitboard().count_ones() as i32
    }

    fn eval_pv_end_pos(&mut self, rx: Option<&Receiver<Message>>, duplicate_check: &mut HashSet<u64>, pv: &[Move], ply: u16) -> Option<(String, u16, i32)> {
        if let Some((m, rest_pv)) = pv.split_first() {
            let (previous_piece, move_state) = self.board.perform_move(*m);
            let result = self.eval_pv_end_pos(rx, duplicate_check, rest_pv, ply + 1);
            self.board.undo_move(*m, previous_piece, move_state);

            return result;
        }

        let hash = self.board.get_hash();
        if duplicate_check.contains(&hash) {
            return None;
        }

        duplicate_check.insert(hash);

        if self.board.is_in_check(self.board.active_player()) {
            return None;
        }

        self.set_node_limit(500);
        let (best_move, pv) = self.find_best_move(rx, 4, &[]);
        if best_move == NO_MOVE {
            return None;
        }

        if pv.moves().len() >= 2 && !self.is_quiet_pv(&pv.moves()[..2], self.piece_count()) {
            return None;
        }

        self.set_node_limit(30000);
        let (best_move, pv) = self.find_best_move(rx, 8, &[]);
        if pv.moves().len() >= 4 && !self.is_quiet_pv(&pv.moves()[..4], self.piece_count()) {
            return None;
        }

        let score = self.board.active_player().score(best_move.score());
        if score.abs() > 3000 {
            return None;
        }

        Some((write_fen(&self.board), ply, score))
    }
}
