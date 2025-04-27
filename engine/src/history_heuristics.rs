/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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
use crate::board::PieceHashes;
use crate::colors::{Color, BLACK, WHITE};
use crate::moves::{Move, NO_MOVE};
use crate::slices::SliceElementAccess;
use crate::transposition_table::MAX_DEPTH;

pub const MIN_HISTORY_SCORE: i16 = -128;

pub const CORR_HISTORY_SIZE: usize = 16384;

#[derive(Clone)]
pub struct HistoryHeuristics {
    killers: Vec<(Move, Move)>,
    counters: Vec<Move>,
    history: Box<HistoryTable>,
    pawn_corr_history: Box<[[CorrHistoryValue; CORR_HISTORY_SIZE]; 2]>,
    non_pawn_corr_history:Box<[[[CorrHistoryValue; CORR_HISTORY_SIZE]; 2]; 2]>,
    move_corr_history: Box<[[CorrHistoryValue; CORR_HISTORY_SIZE]; 2]>,
}

impl Default for HistoryHeuristics {
    fn default() -> Self {
        Self {
            killers: vec![(NO_MOVE, NO_MOVE); MAX_DEPTH],
            counters: vec![NO_MOVE; 512],
            history: Default::default(),
            pawn_corr_history: Box::new([[CorrHistoryValue(0); CORR_HISTORY_SIZE]; 2]),
            non_pawn_corr_history: Box::new([[[CorrHistoryValue(0); CORR_HISTORY_SIZE]; 2]; 2]),
            move_corr_history: Box::new([[CorrHistoryValue(0); CORR_HISTORY_SIZE]; 2]),
        }
    }
}

impl HistoryHeuristics {

    pub fn clear(&mut self) {
        self.killers.fill((NO_MOVE, NO_MOVE));
        self.counters.fill(NO_MOVE);
        self.history.clear();
        self.pawn_corr_history.iter_mut().for_each(|e| e.fill(CorrHistoryValue(0)));
        self.non_pawn_corr_history.iter_mut().for_each(|e| e.iter_mut().for_each(|e| e.fill(CorrHistoryValue(0))));
        self.move_corr_history.iter_mut().for_each(|e| e.fill(CorrHistoryValue(0)));
    }

    pub fn is_empty(&self) -> bool {
        self.killers.iter().all(|e| e.0 == NO_MOVE)
    }

    pub fn clear_killers(&mut self, ply: usize) {
        if let Some(k) = self.killers.get_mut(ply) {
            *k = (NO_MOVE, NO_MOVE);
        }
    }

    pub fn get_killer_moves(&self, ply: usize) -> (Move, Move) {
        self.killers[ply]
    }

    pub fn get_counter_move(&self, opp_m: Move) -> Move {
        if opp_m == NO_MOVE {
            return NO_MOVE;
        }
        self.counters[opp_m.calc_piece_end_index()]
    }

    pub fn update(&mut self, ply: usize, active_player: Color, move_history: MoveHistory, m: Move, has_positive_history: bool) {
        let bonus = if has_positive_history { 1 } else { 4 };
        self.update_history(active_player, move_history, m, bonus);

        self.update_killer_moves(ply, m);
        self.update_counter_move(move_history.last_opp, m);
    }

    fn update_history(&mut self, active_player: Color, move_history: MoveHistory, m: Move, scale: i8) {
        self.history.update_follow_up(active_player, move_history.prev_own, m, scale);
        self.history.update_counter(active_player, move_history.last_opp, m, scale);
    }

    fn update_killer_moves(&mut self, ply: usize, m: Move) {
        let entry = &mut self.killers[ply];
        if entry.0 != m {
            entry.1 = entry.0;
            entry.0 = m;
        }
    }

    pub fn update_counter_move(&mut self, opp_m: Move, counter_m: Move) {
        self.counters[opp_m.calc_piece_end_index()] = counter_m;
    }
    
    pub fn update_played_moves(&mut self, active_player: Color, move_history: MoveHistory, m: Move) {
        self.update_history(active_player, move_history, m, -1);
    }
    
    pub fn update_corr_histories(&mut self, active_player: Color, depth: i32, hashes: PieceHashes, move_history_hash: u16, score_diff: i16) {
        self.pawn_corr_history[active_player.idx()][hashes.pawn as usize & (CORR_HISTORY_SIZE - 1)].update(score_diff, depth);
        self.non_pawn_corr_history[active_player.idx()][WHITE.idx()][hashes.white_non_pawn as usize & (CORR_HISTORY_SIZE - 1)].update(score_diff, depth);
        self.non_pawn_corr_history[active_player.idx()][BLACK.idx()][hashes.black_non_pawn as usize & (CORR_HISTORY_SIZE - 1)].update(score_diff, depth);
        self.move_corr_history[active_player.idx()][(move_history_hash as usize) & (CORR_HISTORY_SIZE - 1)].update(score_diff, depth);
    }

    pub fn score(&self, active_player: Color, move_history: MoveHistory, m: Move) -> i16 {
        let follow_up_score = self.history.follow_up_score(active_player, move_history.prev_own, m);
        let counter_score = self.history.counter_score(active_player, move_history.last_opp, m);

        follow_up_score + counter_score
    }
    
    pub fn corr_eval(&self, active_player: Color, hashes: PieceHashes, move_history_hash: u16) -> i16 {
        let pawn_corr = self.pawn_corr_history[active_player.idx()][hashes.pawn as usize & (CORR_HISTORY_SIZE - 1)].score();
        let white_non_pawn_corr = self.non_pawn_corr_history[active_player.idx()][WHITE.idx()][hashes.white_non_pawn as usize & (CORR_HISTORY_SIZE - 1)].score();
        let black_non_pawn_corr = self.non_pawn_corr_history[active_player.idx()][BLACK.idx()][hashes.black_non_pawn as usize & (CORR_HISTORY_SIZE - 1)].score();
        let move_corr = self.move_corr_history[active_player.idx()][(move_history_hash as usize) & (CORR_HISTORY_SIZE - 1)].score();

        (pawn_corr + white_non_pawn_corr + black_non_pawn_corr + move_corr).clamp(-255, 255)
    }
}

#[derive(Default, Clone, Copy)]
struct HistoryValue(i8);

impl HistoryValue {
    fn update(&mut self, scale: i8) {
        self.0 = self.0.saturating_add(scale * 4 - self.0 / 32);
    }

    fn score(&self) -> i16 {
        self.0 as i16
    }
}

const CORR_HISTORY_GRAIN: i32 = 256;
const CORR_HISTORY_MAX: i32 = 64 * CORR_HISTORY_GRAIN;
const CORR_HISTORY_MAX_WEIGHT: i32 = MAX_DEPTH as i32 + 1;

#[derive(Default, Clone, Copy)]
struct CorrHistoryValue(i16);

impl CorrHistoryValue {
    fn update(&mut self, diff: i16, depth: i32) {
        let weight = depth;
        let weighted_diff = diff as i32 * weight * CORR_HISTORY_GRAIN;
        self.0 = (((CORR_HISTORY_MAX_WEIGHT - weight) * self.0 as i32 + weighted_diff) / CORR_HISTORY_MAX_WEIGHT).clamp(-CORR_HISTORY_MAX, CORR_HISTORY_MAX) as i16;
    }

    fn score(&self) -> i16 {
        self.0 / CORR_HISTORY_GRAIN as i16
    }
}

#[derive(Clone)]
struct HistoryTable([[[(HistoryValue, HistoryValue); 512]; 2]; 512]);

impl Default for HistoryTable {
    fn default() -> Self {
        HistoryTable([[[(HistoryValue::default(), HistoryValue::default()); 512]; 2]; 512])
    }
}

impl HistoryTable {
    fn clear(&mut self) {
        self.0.fill([[(HistoryValue::default(), HistoryValue::default()); 512]; 2]);
    }

    fn update_counter(&mut self, active_player: Color, rel_m: Move, m: Move, scale: i8) {
        self.entry_mut(active_player, rel_m, m).0.update(scale);
    }

    fn update_follow_up(&mut self, active_player: Color, rel_m: Move, m: Move, scale: i8) {
        self.entry_mut(active_player, rel_m, m).1.update(scale);
    }

    fn counter_score(&self, active_player: Color, rel_m: Move, m: Move) -> i16 {
        self.entry(active_player, rel_m, m).0.score()
    }

    fn follow_up_score(&self, active_player: Color, rel_m: Move, m: Move) -> i16 {
        self.entry(active_player, rel_m, m).1.score()
    }

    fn entry_mut(&mut self, active_player: Color, rel_m: Move, m: Move) -> &mut (HistoryValue, HistoryValue) {
        self.0.el_mut(rel_m.calc_piece_end_index()).el_mut(active_player.idx()).el_mut(m.calc_piece_end_index())
    }

    fn entry(&self, active_player: Color, rel_m: Move, m: Move) -> &(HistoryValue, HistoryValue) {
         self.0.el(rel_m.calc_piece_end_index()).el(active_player.idx()).el(m.calc_piece_end_index())
    }
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
