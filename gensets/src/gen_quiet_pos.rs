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

use velvet::moves::{Move};
use velvet::search::{Search};

// Code for generating quiet training positions for tuning and NN training
pub trait GenQuietPos {
    fn is_quiet_pv(&mut self, pv: &[Move], base_mat_score: i32) -> bool;
}

impl GenQuietPos for Search {
    fn is_quiet_pv(&mut self, pv: &[Move], base_mat_score: i32) -> bool {
        if let Some((m, rest_pv)) = pv.split_first() {
            let (previous_piece, move_state) = self.board.perform_move(*m);
            let is_quiet = self.is_quiet_pv(rest_pv, base_mat_score);
            self.board.undo_move(*m, previous_piece, move_state);

            is_quiet

        } else {
            self.board.material_score() == base_mat_score
        }
    }
}
