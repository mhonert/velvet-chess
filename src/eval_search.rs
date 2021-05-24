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

use crate::moves::{Move, NO_MOVE};
use crate::engine::Engine;
use crate::eval::Eval;
use crate::score_util::{MIN_SCORE, MAX_SCORE};
use crate::search::Search;

pub trait EvalSearch {
    fn find_best_move_by_eval(&mut self) -> Move;
}

impl EvalSearch for Engine {
    fn find_best_move_by_eval(&mut self) -> Move {
        let player_color = self.board.active_player();

        self.movegen.enter_ply(player_color, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut move_num = 0;
        let mut best_move = NO_MOVE;
        let mut best_score = MIN_SCORE;
        while let Some(m) = self.movegen.next_legal_move(&self.hh, &mut self.board) {
            move_num += 1;

            let (previous_piece, removed_piece_id) = self.board.perform_move(m);

            // let score = self.get_score();
            let score = -self.quiescence_search(-player_color, MIN_SCORE, MAX_SCORE, 0);
            if score > best_score {
                best_score = score;
                best_move = m;
            }

            self.board.undo_move(m, previous_piece, removed_piece_id);
        }

        self.movegen.leave_ply();

        if move_num == 0 {
            return NO_MOVE;
        }

        best_move
    }
}
