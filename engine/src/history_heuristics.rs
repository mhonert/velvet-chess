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

use crate::colors::Color;
use crate::moves::{Move, NO_MOVE};
use crate::transposition_table::MAX_DEPTH;

const HISTORY_SIZE: usize = 4 * 8 * 64 * 8 * 64;

pub const MIN_HISTORY_SCORE: i16 = -128;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    counters: Vec<Move>,
    follow_up_history: Vec<i8>,
    counter_history: Vec<i8>,
}

impl HistoryHeuristics {
    pub fn new() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            counters: vec![NO_MOVE; 64 * 8 * 64],
            follow_up_history: vec![0; HISTORY_SIZE],
            counter_history: vec![0; HISTORY_SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.killers.fill((NO_MOVE, NO_MOVE));
        self.follow_up_history.fill(0);
        self.counter_history.fill(0);
        self.counters.fill(NO_MOVE);
    }

    pub fn clear_killers(&mut self, ply: usize) {
        if let Some(k) = self.killers.get_mut(ply) {
            *k = (NO_MOVE, NO_MOVE);
        }
    }

    #[inline]
    pub fn get_killer_moves(&self, ply: usize) -> (Move, Move) {
        unsafe { *self.killers.get_unchecked(ply) }
    }

    #[inline]
    pub fn get_counter_move(&self, opponent_move: Move) -> Move {
        if opponent_move == NO_MOVE {
            return NO_MOVE;
        }
        unsafe { *self.counters.get_unchecked(opponent_move.calc_piece_end_index() * 64 + opponent_move.start() as usize) }
    }

    #[inline]
    pub fn update(&mut self, ply: usize, active_player: Color, move_history: MoveHistory, m: Move) {
        self.update_follow_up_history(active_player, move_history.prev_own, m, 1);
        self.update_counter_history(active_player, move_history.last_opp, m, 1);

        self.update_killer_moves(ply, m);
        self.update_counter_move(move_history.last_opp, m);
    }

    #[inline]
    fn update_follow_up_history(&mut self, active_player: Color, prev_m: Move, m: Move, counter_scale: i8) {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let type_offset = if prev_m.is_capture() { 0 } else { HISTORY_SIZE / 4 };
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
        let type_offset = if opp_m.is_capture() { 0 } else { HISTORY_SIZE / 4 };
        let entry = unsafe {
            self.counter_history.get_unchecked_mut(
                color_offset + type_offset + opp_m.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            )
        };
        *entry = entry.saturating_add(counter_scale * 4 - *entry / 32);
    }

    #[inline]
    pub fn update_killer_moves(&mut self, ply: usize, m: Move) {
        let entry = unsafe { self.killers.get_unchecked_mut(ply) };
        if entry.0 != m.without_score() {
            entry.1 = entry.0;
            entry.0 = m.without_score();
        }
    }

    #[inline]
    pub fn update_counter_move(&mut self, opponent_move: Move, counter_move: Move) {
        if opponent_move != NO_MOVE {
            *unsafe { self.counters.get_unchecked_mut(opponent_move.calc_piece_end_index() * 64 + opponent_move.start() as usize) } = counter_move;
        }
    }

    #[inline]
    pub fn update_played_moves(&mut self, active_player: Color, move_history: MoveHistory, m: Move) {
        self.update_follow_up_history(active_player, move_history.prev_own, m, -1);
        self.update_counter_history(active_player, move_history.last_opp, m, -1);
    }

    #[inline]
    pub fn get_history_score(&self, active_player: Color, move_history: MoveHistory, m: Move) -> i16 {
        let color_offset = if active_player.is_white() { 0 } else { HISTORY_SIZE / 2 };
        let type_offset = if move_history.prev_own.is_capture() { 0 } else { HISTORY_SIZE / 4 };
        let fu_score = unsafe {
            *self.follow_up_history.get_unchecked(
                color_offset + type_offset + move_history.prev_own.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            ) as i16
        };

        let type_offset = if move_history.last_opp.is_capture() { 0 } else { HISTORY_SIZE / 4 };
        let cm_score = unsafe {
            *self.counter_history.get_unchecked(
                color_offset + type_offset + move_history.last_opp.calc_piece_end_index() * 64 * 8 + m.calc_piece_end_index(),
            ) as i16
        };

        fu_score + cm_score
    }
}

#[derive(Copy, Clone, Default)]
pub struct MoveHistory {
    pub last_opp: Move,
    pub prev_own: Move,
    pub second_last_own: Move,
}

pub const EMPTY_HISTORY: MoveHistory = MoveHistory{last_opp: NO_MOVE, prev_own: NO_MOVE, second_last_own: NO_MOVE};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors::WHITE;
    use crate::moves::MoveType;

    #[test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::new();

        let move_a = Move::new(MoveType::QueenQuiet, 1, 2);
        let move_b = Move::new(MoveType::RookQuiet, 4, 5);
        hh.update(1, WHITE, EMPTY_HISTORY, move_a);
        hh.update(1, WHITE, EMPTY_HISTORY, move_b);

        let (primary_killer, secondary_killer) = hh.get_killer_moves(1);

        assert_eq!(primary_killer, move_b, "move_b should be the primary killer move");
        assert_eq!(secondary_killer, move_a, "move_a should be the secondary killer move");
    }
}
