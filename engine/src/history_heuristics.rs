/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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

const HISTORY_SIZE: usize =
    2 *      // side to move
    2 *      // is own or opponent previous move
    2 *      // previous move type (capture or quiet)
    8 * 64 * // previous move piece and target
    8 * 64;  // follow up move piece and target

pub const MIN_HISTORY_SCORE: i16 = -128;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    counters: Vec<Move>,
    follow_up_history: Vec<i8>,
}

impl Default for HistoryHeuristics {
    fn default() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            counters: vec![NO_MOVE; 64 * 8 * 64],
            follow_up_history: vec![0; HISTORY_SIZE],
        }
    }
}

impl HistoryHeuristics {

    pub fn clear(&mut self) {
        self.killers.fill((NO_MOVE, NO_MOVE));
        self.follow_up_history.fill(0);
        self.counters.fill(NO_MOVE);
    }
    
    pub fn is_empty(&self) -> bool {
        self.killers.iter().all(|e| e.0 == NO_MOVE)
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
    pub fn get_counter_move(&self, opp_m: Move) -> Move {
        if opp_m == NO_MOVE {
            return NO_MOVE;
        }
        unsafe { *self.counters.get_unchecked(opp_m.calc_piece_end_index() * 64 + opp_m.start() as usize) }
    }

    #[inline]
    pub fn update(&mut self, ply: usize, active_player: Color, move_history: MoveHistory, m: Move, has_positive_history: bool) {
        let offset = base_offset(active_player, m);
        let bonus = if has_positive_history { 1 } else { 4 };

        self.update_history(offset, false, move_history.prev_own, bonus);
        self.update_history(offset, true, move_history.last_opp, bonus);

        self.update_killer_moves(ply, m);
        self.update_counter_move(move_history.last_opp, m);
    }

    #[inline]
    fn update_history(&mut self, base_offset: usize, prev_is_opp: bool, prev_m: Move, counter_scale: i8) {
        let idx = history_idx(base_offset, prev_is_opp, prev_m);
        let entry = unsafe { self.follow_up_history.get_unchecked_mut(idx) };
        *entry = entry.saturating_add(counter_scale * 4 - *entry / 32);
    }

    #[inline]
    pub fn update_killer_moves(&mut self, ply: usize, m: Move) {
        let entry = unsafe { self.killers.get_unchecked_mut(ply) };
        if entry.0 != m {
            entry.1 = entry.0;
            entry.0 = m;
        }
    }

    #[inline]
    pub fn update_counter_move(&mut self, opp_m: Move, counter_m: Move) {
        *unsafe { self.counters.get_unchecked_mut(opp_m.calc_piece_end_index() * 64 + opp_m.start() as usize) } = counter_m;
    }

    #[inline]
    pub fn update_played_moves(&mut self, active_player: Color, move_history: MoveHistory, m: Move) {
        let offset = base_offset(active_player, m);
        self.update_history(offset, false, move_history.prev_own, -1);
        self.update_history(offset, true, move_history.last_opp, -1);
    }

    #[inline]
    pub fn score(&self, active_player: Color, move_history: MoveHistory, m: Move) -> i16 {
        let offset = base_offset(active_player, m);

        self.history_score(offset, false, move_history.prev_own)
            + self.history_score(offset, true, move_history.last_opp)
    }

    #[inline]
    fn history_score(&self, offset: usize, prev_is_opp: bool, prev_m: Move) -> i16 {
        unsafe { *self.follow_up_history.get_unchecked(history_idx(offset, prev_is_opp, prev_m) ) as i16 }
    }
}

#[inline]
fn base_offset(active_player: Color, m: Move) -> usize {
    let color_offset = if active_player.is_white() { HISTORY_SIZE / 2 } else { 0 };
    color_offset + m.calc_piece_end_index()
}

#[inline]
fn history_idx(base_offset: usize, prev_is_opp: bool, prev_m: Move) -> usize {
    let opp_offset = if prev_is_opp { HISTORY_SIZE / 4 } else { 0 };
    let type_offset = if prev_m.is_capture() { HISTORY_SIZE / 8 } else { 0 };
    base_offset + opp_offset + type_offset + prev_m.calc_piece_end_index() * 64 * 8
}

#[derive(Copy, Clone, Default)]
pub struct MoveHistory {
    pub last_opp: Move,
    pub prev_own: Move,
}

pub const EMPTY_HISTORY: MoveHistory = MoveHistory{last_opp: NO_MOVE, prev_own: NO_MOVE};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colors::WHITE;
    use crate::moves::MoveType;

    #[test]
    fn updates_killer_moves() {
        let mut hh = HistoryHeuristics::default();

        let move_a = Move::new(MoveType::QueenQuiet, 1, 2);
        let move_b = Move::new(MoveType::RookQuiet, 4, 5);
        hh.update(1, WHITE, EMPTY_HISTORY, move_a, true);
        hh.update(1, WHITE, EMPTY_HISTORY, move_b, true);

        let (primary_killer, secondary_killer) = hh.get_killer_moves(1);

        assert_eq!(primary_killer, move_b, "move_b should be the primary killer move");
        assert_eq!(secondary_killer, move_a, "move_a should be the secondary killer move");
    }
}
