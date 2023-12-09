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

use crate::board::Board;
use crate::colors::Color;
use crate::history_heuristics::HistoryHeuristics;
use crate::move_gen::MoveGenerator;
use crate::moves::Move;

pub struct SearchContext {
    ply: usize,
    idx: usize,
    movegen: MoveGenerator,
}

impl Default for SearchContext {
    fn default() -> Self {
        SearchContext{
            ply: 0,
            idx: 3, // start with 3 to remove the need for bounds checks when accessing the ply infos
            movegen: MoveGenerator::new(),
        }
    }
}

impl SearchContext {

    pub fn enter_ply(&mut self) {
        self.ply += 1;
        self.idx += 1;
        self.movegen.enter_ply();
    }

    pub fn leave_ply(&mut self) {
        self.ply += 1;
        self.idx += 1;
        self.movegen.leave_ply();
    }

    pub fn next_move(&mut self, ply: usize, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.movegen.next_move(ply, hh, board)
    }

    pub fn generate_qs_captures(&mut self, board: &mut Board) {
        self.movegen.generate_qs_captures(board);
    }

    pub fn next_good_capture_move(&mut self, board: &mut Board, threshold: i16) -> Option<Move> {
        self.movegen.next_good_capture_move(board, threshold)
    }

    pub fn reset_root_moves(&mut self) {
        self.movegen.reset_root_moves();
    }

    pub fn next_root_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.movegen.next_root_move(hh, board)
    }

    pub fn update_root_move(&mut self, m: Move) {
        self.movegen.update_root_move(m);
    }

    pub fn reorder_root_moves(&mut self, best_move: Move, sort_other_moves: bool) {
        self.movegen.reorder_root_moves(best_move, sort_other_moves);
    }

    pub fn prepare_moves(&mut self, active_player: Color, hash_move: Move, prev_own_move: Move, opp_move: Move) {
        self.movegen.init(active_player, hash_move, prev_own_move, opp_move);
    }
}

#[macro_export]
macro_rules! next_ply {
    ($ctx:expr, $func_call:expr) => {
        {
            $ctx.enter_ply();
            let result = $func_call;
            $ctx.leave_ply();
            result
        }
    };
}