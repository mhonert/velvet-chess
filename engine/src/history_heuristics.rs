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

pub struct HistoryHeuristics {
    primary_killers: Vec<Move>,
    secondary_killers: Vec<Move>,
    cut_off_history: Vec<u64>,
    played_move_history: Vec<u64>,
    entries: u64
}

impl Default for HistoryHeuristics {
    fn default() -> Self {
        HistoryHeuristics::new()
    }
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            primary_killers: vec![NO_MOVE; MAX_DEPTH],
            secondary_killers: vec![NO_MOVE; MAX_DEPTH],
            cut_off_history: vec![0; HISTORY_SIZE],
            played_move_history: vec![0; HISTORY_SIZE],
            entries: 0
        }
    }

    pub fn clear(&mut self) {
        self.entries = 0;

        self.primary_killers.fill(NO_MOVE);
        self.secondary_killers.fill(NO_MOVE);
        self.cut_off_history.fill(0);
        self.played_move_history.fill(0);
    }

    #[inline]
    pub fn get_primary_killer(&self, ply: i32) -> Move {
        unsafe { *self.primary_killers.get_unchecked(ply as usize) }
    }

    #[inline]
    pub fn get_secondary_killer(&self, ply: i32) -> Move {
        unsafe { *self.secondary_killers.get_unchecked(ply as usize) }
    }

    #[inline]
    pub fn update(&mut self, depth: i32, ply: i32, color: Color, m: Move) {
        if depth >= HEURISTICS_THRESHOLD {
            let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
            unsafe { *self.cut_off_history.get_unchecked_mut(color_offset + m.to_index(0x1FF)) += 1 };
        }
        self.update_killer_moves(ply, m);
    }

    #[inline]
    fn update_killer_moves(&mut self, ply: i32, m: Move) {
        let current_primary = unsafe { self.primary_killers.get_unchecked_mut(ply as usize) };
        if *current_primary != m.without_score() {
            unsafe { *self.secondary_killers.get_unchecked_mut(ply as usize) = *current_primary };
            *current_primary = m.without_score();
        }
    }

    #[inline]
    pub fn update_played_moves(&mut self, depth: i32, color: Color, m: Move) {
        if depth >= HEURISTICS_THRESHOLD {
            self.entries += 1;
            let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
            unsafe {
                *self.played_move_history.get_unchecked_mut(color_offset + m.to_index(0x1FF)) += 1;
            }
        }
    }

    #[inline]
    pub fn get_history_score(&self, color: Color, m: Move) -> i32 {
        let color_offset = if color == WHITE { 0 } else { HISTORY_SIZE / 2 };
        let index = color_offset + m.to_index(0x1FF);

        let move_count = unsafe { *self.played_move_history.get_unchecked(index) };
        if move_count <= (self.entries / 2304) {
            return -1;
        }

        (unsafe { *self.cut_off_history.get_unchecked(index) } * 512 / move_count) as i32
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

        let primary_killer = hh.get_primary_killer(1);
        let secondary_killer = hh.get_secondary_killer(1);

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
