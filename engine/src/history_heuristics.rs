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

use crate::colors::{Color, WHITE};
use crate::transposition_table::MAX_DEPTH;
use crate::moves::{Move, NO_MOVE};

const HISTORY_SIZE: usize = 2 * 8 * 64;

const HEURISTICS_THRESHOLD: i32 = 5;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    cut_off_history: Vec<(u64, u64)>,
    entries: u64,
}

impl Default for HistoryHeuristics {
    fn default() -> Self {
        HistoryHeuristics::new()
    }
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            cut_off_history: vec![(0, 0); HISTORY_SIZE],
            entries: 0
        }
    }

    pub fn clear(&mut self) {
        self.entries = 0;

        self.killers.fill((NO_MOVE, NO_MOVE));
        self.cut_off_history.fill((0, 0));
    }

    #[inline]
    pub fn get_killer_moves(&self, ply: i32) -> (Move, Move) {
        unsafe { *self.killers.get_unchecked(ply as usize) }
    }

    #[inline]
    pub fn update(&mut self, depth: i32, ply: i32, color: Color, m: Move) {
        if depth >= HEURISTICS_THRESHOLD {
            let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
            unsafe { self.cut_off_history.get_unchecked_mut(color_offset + m.calc_piece_end_index()).0 += 1 };
        }
        self.update_killer_moves(ply, m);
    }

    #[inline]
    fn update_killer_moves(&mut self, ply: i32, m: Move) {
        let entry = unsafe { self.killers.get_unchecked_mut(ply as usize) };
        if entry.0 != m.without_score() {
            entry.1 = entry.0;
            entry.0 = m.without_score();
        }
    }

    #[inline]
    pub fn update_played_moves(&mut self, depth: i32, color: Color, m: Move) {
        if depth >= HEURISTICS_THRESHOLD {
            self.entries += 1;
            let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
            unsafe {
                self.cut_off_history.get_unchecked_mut(color_offset + m.calc_piece_end_index()).1 += 1;
            }
        }
    }

    #[inline]
    pub fn get_history_score(&self, color: Color, m: Move) -> i32 {
        let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
        let index = color_offset + m.calc_piece_end_index();

        let entry = unsafe { *self.cut_off_history.get_unchecked(index) };
        let move_count = entry.1;
        if move_count <= (self.entries / 2304) {
            return -1;
        }

        (entry.0 * 512 / move_count) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pieces::{Q, R};
    use crate::moves::MoveType;

    #[test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::new();

        let move_a = Move::new(MoveType::Quiet, Q, 1, 2);
        let move_b = Move::new(MoveType::Quiet, R, 4, 5);
        hh.update(3, 1, WHITE, move_a);
        hh.update(3, 1, WHITE, move_b);

        let (primary_killer, secondary_killer) = hh.get_killer_moves(1);

        assert_eq!(
            primary_killer, move_b,
            "move_b should be the primary killer move"
        );
        assert_eq!(
            secondary_killer, move_a,
            "move_a should be the secondary killer move"
        );
    }
}
