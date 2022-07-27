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

const HISTORY_SIZE: usize = 4 * 8 * 64 * 8 * 64;

pub const MIN_HISTORY_SCORE: i32 = -128;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    counters: Vec<[Move; 64]>,
    follow_up_history: Vec<i8>,
    counter_history: Vec<i8>,
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            counters: vec![[NO_MOVE; 64]; 64 * 8],
            follow_up_history: vec![0; HISTORY_SIZE],
            counter_history: vec![0; HISTORY_SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.killers.fill((NO_MOVE, NO_MOVE));
        self.follow_up_history.fill(0);
        self.counter_history.fill(0);
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

    #[inline]
    pub fn update(&mut self, ply: i32, active_player: Color, prev_own_m: Move, opponent_move: Move, m: Move) {
        self.update_follow_up_history(active_player, prev_own_m, m, 1);
        self.update_counter_history(active_player, opponent_move, m, 1);

        self.update_killer_moves(ply, m);
        self.update_counter_move(opponent_move, m);
    }

    #[inline]
    fn update_follow_up_history(&mut self, active_player: Color, prev_m: Move, m: Move, counter_scale: i8) {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let type_offset = if prev_m.is_quiet() { 0 } else { HISTORY_SIZE / 4 };
        let entry = unsafe {
            self.follow_up_history.get_unchecked_mut(
                color_offset + type_offset + prev_m.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            )
        };
        *entry = entry.saturating_add(counter_scale * 4 - *entry / 32);
    }

    #[inline]
    fn update_counter_history(&mut self, active_player: Color, opp_m: Move, m: Move, counter_scale: i8) {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let type_offset = if opp_m.is_quiet() { 0 } else { HISTORY_SIZE / 4 };
        let entry = unsafe {
            self.counter_history.get_unchecked_mut(
                color_offset + type_offset + opp_m.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            )
        };
        *entry = entry.saturating_add(counter_scale * 4 - *entry / 32);
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
    pub fn update_played_moves(&mut self, active_player: Color, prev_own_m: Move, opp_m: Move, m: Move) {
        self.update_follow_up_history(active_player, prev_own_m, m, -1);
        self.update_counter_history(active_player, opp_m, m, -1);
    }

    #[inline]
    pub fn get_history_score(&self, active_player: Color, prev_m: Move, opp_m: Move, m: Move) -> i32 {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let type_offset = if prev_m.is_quiet() { 0 } else { HISTORY_SIZE / 4 };
        let fu_score = unsafe {
            *self.follow_up_history.get_unchecked(
                color_offset + type_offset + prev_m.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            ) as i32
        };

        let type_offset = if opp_m.is_quiet() { 0 } else { HISTORY_SIZE / 4 };
        let cm_score = unsafe {
            *self.counter_history.get_unchecked(
                color_offset + type_offset + opp_m.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            ) as i32
        };

        fu_score + cm_score
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
        hh.update(1, WHITE, NO_MOVE, NO_MOVE, move_a);
        hh.update(1, WHITE, NO_MOVE, NO_MOVE, move_b);

        let (primary_killer, secondary_killer) = hh.get_killer_moves(1);

        assert_eq!(primary_killer, move_b, "move_b should be the primary killer move");
        assert_eq!(secondary_killer, move_a, "move_a should be the secondary killer move");
    }
}
