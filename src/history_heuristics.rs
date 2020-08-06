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

use crate::move_gen::{Move, NO_MOVE};
use crate::colors::{Color, WHITE};
use crate::transposition_table::MAX_DEPTH;

const HISTORY_SIZE: usize = 2 * 64 * 64;

pub struct HistoryHeuristics {
    primary_killers: [Move; MAX_DEPTH + 1],
    secondary_killers: [Move; MAX_DEPTH + 1],
    cut_off_history: [u64; HISTORY_SIZE],
    played_move_history: [u64; HISTORY_SIZE],
    played_move_thresholds: [u64; MAX_DEPTH + 1],
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        HistoryHeuristics{
            primary_killers: [NO_MOVE; MAX_DEPTH + 1],
            secondary_killers: [NO_MOVE; MAX_DEPTH + 1],
            cut_off_history: [0; HISTORY_SIZE],
            played_move_history: [0; HISTORY_SIZE],
            played_move_thresholds: calc_move_thresholds(),
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

    pub fn get_primary_killer(&self, ply: i32) -> Move {
        self.primary_killers[ply as usize]
    }

    pub fn get_secondary_killer(&self, ply: i32) -> Move {
        self.secondary_killers[ply as usize]
    }

    pub fn update(&mut self, ply: i32, color: Color, start: i32, end: i32, m: Move) {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        self.cut_off_history[(color_offset + start + end * 64) as usize] += 1;
        self.update_killer_moves(ply, m);
    }

    fn update_killer_moves(&mut self, ply: i32, m: Move) {
        let current_primary = self.primary_killers[ply as usize];
        if current_primary == m {
            return;
        }

        self.primary_killers[ply as usize] = m;
        self.secondary_killers[ply as usize] = current_primary;
    }

    pub fn update_played_moves(&mut self, color: Color, start: i32, end: i32) {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        self.played_move_history[(color_offset + start + end * 64) as usize] += 1;
    }

    pub fn get_history_score(&self, color: Color, start: i32, end: i32) -> i32 {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        let index = (color_offset + start + end * 64) as usize;

        let move_count = self.played_move_history[index];
        if move_count == 0 {
            return 0;
        }

        (self.cut_off_history[index] * 512 / move_count) as i32
    }

    // Returns true, if the history contains sufficient information about the given move, to indicate
    // that it is very unlikely to cause a cut-off during search
    pub fn has_negative_history(&self, color: Color, depth: i32, start: i32, end: i32) -> bool {
        let color_offset = if color == WHITE { 0 } else { 64 * 64 };
        let index = (color_offset + start + end * 64) as usize;

        let move_count = self.played_move_history[index];
        if move_count < self.played_move_thresholds[depth as usize] {
            return false;
        }

        (self.cut_off_history[index] * 512 / move_count) == 0
    }


}

fn calc_move_thresholds() -> [u64; MAX_DEPTH + 1] {
    let mut thresholds: [u64; MAX_DEPTH + 1] = [0; MAX_DEPTH + 1];
    let mut threshold: f32 = 2.0;

    for depth in 0..=MAX_DEPTH {
        thresholds[depth] = threshold as u64;
        threshold *= 1.6;
    }

    thresholds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_gen::encode_move;
    use crate::pieces::{Q, R};

    # [test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::new();

        let move_a = encode_move(Q, 1, 2);
        let move_b = encode_move(R, 4, 5);
        hh.update(1, WHITE, 1, 2, move_a);
        hh.update(1, WHITE, 4, 5, move_b);

        let primary_killer = hh.get_primary_killer(1);
        let secondary_killer = hh.get_secondary_killer(1);

        assert_eq!(primary_killer, move_b, "move_b should be the primary killer move");
        assert_eq!(secondary_killer, move_a, "move_a should be the secondary killer move");
    }

}