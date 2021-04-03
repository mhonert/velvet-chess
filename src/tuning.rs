/*
 * Velvet Chess Engine
 * Copyright (C) 2020 mhonert (https://github.com/mhonert)
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

use crate::colors::{BLACK, WHITE};
use crate::engine::Engine;
use crate::pieces::{EMPTY};
use crate::score_util::{ MAX_SCORE, MIN_SCORE };
use crate::transposition_table::{get_untyped_move, ScoreType};
use crate::moves::{Move, NO_MOVE};
use crate::move_gen::{is_likely_valid_move};

// Code for generating quiet training positions for tuning
pub trait Tuning {
    fn is_quiet_position(&mut self) -> bool;

    fn is_quiet_pv(&mut self, m: Move, depth: i32) -> bool;

    fn make_quiet_position(&mut self) -> bool;

    fn static_quiescence_search(&mut self, alpha: i32, beta: i32, ply: i32) -> i32;
}

impl Tuning for Engine {
    fn is_quiet_position(&mut self) -> bool {
        if self.board.is_in_check(WHITE) || self.board.is_in_check(BLACK) {
            return false;
        }

        self.movegen.enter_ply(self.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE);
        while let Some(m) = self.movegen.next_capture_move(&mut self.board)
        {
            let previous_piece = self.board.get_item(m.start());
            let previous_piece_id = previous_piece.abs();
            let captured_piece_id = self.board.get_item(m.end()).abs();

            // skip capture moves with a SEE score below the given threshold
            if self.board.see_score(-self.board.active_player(), m.start(), m.end(), previous_piece_id, captured_piece_id) > 0 {
                self.movegen.leave_ply();
                return false;
            }
        }

        self.movegen.leave_ply();
        true
    }

    fn is_quiet_pv(&mut self, m: Move, depth: i32) -> bool {
        if self.board.is_in_check(WHITE) || self.board.is_in_check(BLACK) {
            return false;
        }

        if depth == 0 {
            return true;
        }

        let (previous_piece, move_state) = self.board.perform_move(m);

        if move_state != EMPTY {
            self.board.undo_move(m, previous_piece, move_state);
            return false;
        }

        let entry = self.tt.get_entry(self.board.get_hash());
        let mut next_move = if entry != 0 {
            get_untyped_move(entry)
        } else {
            NO_MOVE
        };

        if next_move == NO_MOVE {
            self.board.undo_move(m, previous_piece, move_state);
            return false;
        }

        next_move = next_move.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()));

        let active_player = self.board.active_player();
        let is_valid_followup_move = next_move != NO_MOVE && is_likely_valid_move(&self.board, active_player, next_move);
        let is_quiet = if is_valid_followup_move {
            self.is_quiet_pv(next_move, depth - 1)
        } else {
            false
        };

        self.board.undo_move(m, previous_piece, move_state);

        is_quiet
    }

    // Updates the current position until it is quiet
    fn make_quiet_position(&mut self) -> bool {
        let _ = self.static_quiescence_search(MIN_SCORE, MAX_SCORE, 0) as i16;

        loop {
            let entry = self.tt.get_entry(self.board.get_hash());
            if entry == 0 {
                return true;
            }

            let mut m = get_untyped_move(entry);
            if m == NO_MOVE || !is_likely_valid_move(&self.board, self.board.active_player(), m) {
                return true;
            }

            m = m.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()));

            self.board.perform_move(m);
            if self.board.is_in_check(self.board.active_player()) {
                return false;
            }
        }
    }

    fn static_quiescence_search(&mut self, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        let active_player = self.board.active_player();

        let position_score = self.board.get_static_score() as i32 * active_player as i32;

        if ply >= 60 as i32 {
            return position_score;
        }

        if position_score >= beta {
            return beta;
        }

        if alpha < position_score {
            alpha = position_score;
        }

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut best_move = NO_MOVE;

        while let Some(m) = self.movegen.next_capture_move(&mut self.board) {
            let start = m.start();
            let end = m.end();
            let previous_piece = self.board.get_item(start);
            let previous_piece_id = previous_piece.abs();
            let captured_piece_id = self.board.get_item(end).abs();

            // skip capture moves with a SEE score below the given threshold
            if captured_piece_id != EMPTY && self.board.see_score(-active_player, start, end, previous_piece_id, captured_piece_id) < 0 {
                continue;
            }

            let (_, move_state) = self.board.perform_move(m);

            if self.board.is_in_check(active_player) {
                // Invalid move
                self.board.undo_move(m, previous_piece, move_state);
                continue;
            }

            let score = -self.static_quiescence_search(-beta, -alpha, ply + 1);
            self.board.undo_move(m, previous_piece, move_state);

            if score >= beta {
                self.tt.write_entry(self.board.get_hash(), 60 - ply, m, ScoreType::LowerBound);
                self.movegen.leave_ply();
                return beta;
            }

            if score > alpha {
                best_move = m;
                alpha = score;
            }
        }

        if best_move != NO_MOVE {
            self.tt.write_entry(self.board.get_hash(), 60 - ply, best_move, ScoreType::Exact);
        }

        self.movegen.leave_ply();
        alpha
    }

}
