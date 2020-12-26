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

const HISTORY_SIZE: usize = 2 * 64 * 64;

const HEURISTICS_THRESHOLD: i32 = 5;

pub struct HistoryHeuristics {
    primary_killers: [Move; MAX_DEPTH],
    secondary_killers: [Move; MAX_DEPTH],
    cut_off_history: [u64; HISTORY_SIZE],
    played_move_history: [u64; HISTORY_SIZE],
}

impl Default for HistoryHeuristics {
    fn default() -> Self {
        HistoryHeuristics::new()
    }
}

const PLAYED_MOVE_THRESHOLDS: [u64; MAX_DEPTH] = calc_move_thresholds();

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            primary_killers: [NO_MOVE; MAX_DEPTH],
            secondary_killers: [NO_MOVE; MAX_DEPTH],
            cut_off_history: [0; HISTORY_SIZE],
            played_move_history: [0; HISTORY_SIZE],
        }
    }

    pub fn clear(&mut self) {
        for i in 0..MAX_DEPTH {
            self.primary_killers[i] = NO_MOVE;
            self.secondary_killers[i] = NO_MOVE;
        }

        for i in 0..HISTORY_SIZE {
            self.cut_off_history[i] = 0;
            self.played_move_history[i] = 0;
        }
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
    pub fn update(&mut self, depth: i32, ply: i32, color: Color, start: i32, end: i32, m: Move) {
        if depth >= HEURISTICS_THRESHOLD {
            let color_offset = if color == WHITE { 0 } else { 64 * 64 };
            self.cut_off_history[(color_offset + start + end * 64) as usize] += 1;
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
    pub fn update_played_moves(&mut self, depth: i32, color: Color, start: i32, end: i32) {
        if depth >= HEURISTICS_THRESHOLD {
            let color_offset = if color == WHITE { 0 } else { 64 * 64 };
            unsafe {
                *self.played_move_history.get_unchecked_mut((color_offset + start + end * 64) as usize) += 1;
            }
        }
    }

    #[inline]
    pub fn get_history_score(&self, color: Color, start: i32, end: i32) -> i32 {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        let index = (color_offset + start + end * 64) as usize;

        let move_count = unsafe { *self.played_move_history.get_unchecked(index) };
        if move_count == 0 {
            return -1;
        }

        (unsafe { *self.cut_off_history.get_unchecked(index) } * 512 / move_count) as i32
    }

    // Returns true, if the history contains sufficient information about the given move, to indicate
    // that it is very unlikely to cause a cut-off during search
    #[inline]
    pub fn has_negative_history(&self, color: Color, depth: i32, start: i32, end: i32) -> bool {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        let index = (color_offset + start + end * 64) as usize;

        let move_count = unsafe { *self.played_move_history.get_unchecked(index) };
        if move_count < unsafe { *PLAYED_MOVE_THRESHOLDS.get_unchecked(depth as usize) } {
            return false;
        }

        (unsafe { *self.cut_off_history.get_unchecked(index) } * 512 / move_count) == 0
    }
}

const fn calc_move_thresholds() -> [u64; MAX_DEPTH] {
    let mut thresholds: [u64; MAX_DEPTH] = [0; MAX_DEPTH];
    let mut threshold: u64 = 20;

    let mut depth = 0;
    while depth < MAX_DEPTH {
        thresholds[depth] = threshold / 10 as u64;

        if threshold < u64::max_value() / 16 {
            threshold *= 16;
            threshold /= 10;
        }

        depth += 1;
    }

    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pieces::{Q, R};

    #[test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::new();

        let move_a = Move::new(Q, 1, 2);
        let move_b = Move::new(R, 4, 5);
        hh.update(3, 1, WHITE, 1, 2, move_a);
        hh.update(3, 1, WHITE, 4, 5, move_b);

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
