/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use crate::colors::Color;
use crate::moves::{Move, NO_MOVE};
use crate::transposition_table::MAX_DEPTH;
use std::cmp::min;

const HISTORY_SIZE: usize = 2 * 8 * 64;

pub const MIN_HISTORY_SCORE: i32 = -512;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    counters: Vec<[Move; 64]>,
    cut_off_history: Vec<i16>,
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            counters: vec![[NO_MOVE; 64]; 64 * 8],
            cut_off_history: vec![0; HISTORY_SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.killers.fill((NO_MOVE, NO_MOVE));
        self.cut_off_history.fill(0);
        self.counters.fill([NO_MOVE; 64]);
    }

    #[inline]
    pub fn get_killer_moves(&self, ply: i32) -> (Move, Move) {
        self.killers[ply as usize]
    }

    #[inline]
    pub fn get_counter_move(&self, opponent_move: Move) -> Move {
        if opponent_move == NO_MOVE {
            return NO_MOVE;
        }
        self.counters[opponent_move.calc_piece_end_index()][opponent_move.start() as usize]
    }

    #[inline(always)]
    pub fn calc_counter_scale(&self, depth: i32) -> i32 {
        min(512, depth * depth)
    }

    #[inline]
    pub fn update(&mut self, ply: i32, active_player: Color, opponent_move: Move, m: Move, counter_scale: i32) {
        self.update_cut_off_history(active_player, m, counter_scale);
        self.update_killer_moves(ply, m);
        self.update_counter_move(opponent_move, m);
    }

    #[inline]
    fn update_cut_off_history(&mut self, active_player: Color, m: Move, counter_scale: i32) {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let entry = unsafe { self.cut_off_history.get_unchecked_mut(color_offset + m.calc_piece_end_index()) };
        *entry = entry.saturating_add((counter_scale * 32 - *entry as i32 * counter_scale.abs() / 512) as i16);
    }

    #[inline]
    pub fn update_killer_moves(&mut self, ply: i32, m: Move) {
        let entry = unsafe { self.killers.get_unchecked_mut(ply as usize) };
        if entry.0 != m.without_score() {
            entry.1 = entry.0;
            entry.0 = m.without_score();
        }
    }

    #[inline]
    pub fn update_counter_move(&mut self, opponent_move: Move, counter_move: Move) {
        if opponent_move != NO_MOVE {
            self.counters[opponent_move.calc_piece_end_index()][opponent_move.start() as usize] = counter_move;
        }
    }

    #[inline]
    pub fn update_played_moves(&mut self, active_player: Color, m: Move, counter_scale: i32) {
        self.update_cut_off_history(active_player, m, -counter_scale);
    }

    #[inline]
    pub fn get_history_score(&self, color: Color, m: Move) -> i32 {
        let color_offset = if color.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let index = color_offset + m.calc_piece_end_index();

        unsafe { (*self.cut_off_history.get_unchecked(index) / 32) as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors::WHITE;
    use crate::moves::MoveType;
    use crate::pieces::{Q, R};

    #[test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::new();

        let move_a = Move::new(MoveType::Quiet, Q, 1, 2);
        let move_b = Move::new(MoveType::Quiet, R, 4, 5);
        hh.update(1, WHITE, NO_MOVE, move_a, 1);
        hh.update(1, WHITE, NO_MOVE, move_b, 1);

        let (primary_killer, secondary_killer) = hh.get_killer_moves(1);

        assert_eq!(primary_killer, move_b, "move_b should be the primary killer move");
        assert_eq!(secondary_killer, move_a, "move_a should be the secondary killer move");
    }
}
