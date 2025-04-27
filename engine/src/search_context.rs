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

use std::array;
use crate::board::Board;
use crate::colors::Color;
use crate::history_heuristics::{EMPTY_HISTORY, HistoryHeuristics, MoveHistory};
use crate::move_gen::{MoveList};
use crate::moves::{Move, NO_MOVE};
use crate::slices::SliceElementAccess;
use crate::transposition_table::MAX_DEPTH;
use crate::zobrist::piece_zobrist_key;

pub struct SearchContext {
    ply: usize,

    movelists: [MoveList; MAX_DEPTH + 1],
    ply_entries: [PlyEntry; MAX_DEPTH + 4 + 2 + 1],
    
    root_move_randomization: bool,
}

impl Default for SearchContext {
    fn default() -> Self {
        SearchContext{
            ply: 0,
            movelists: array::from_fn(|_| MoveList::default()),
            ply_entries: [PlyEntry::default(); MAX_DEPTH + 4 + 2 + 1],
            root_move_randomization: false,
        }
    }
}

impl SearchContext {
    pub fn enter_ply(&mut self) {
        self.ply += 1;
        self.ply_entry_mut(self.pe_idx()).double_extensions = self.ply_entry(self.pe_idx() - 1).double_extensions;
    }

    pub fn leave_ply(&mut self) {
        self.ply -= 1;
    }

    pub fn max_qs_depth_reached(&self) -> bool {
        self.ply >= MAX_DEPTH
    }

    pub fn max_search_depth_reached(&self) -> bool {
        self.ply >= MAX_DEPTH - 16
    }

    fn movelist_mut(&mut self) -> &mut MoveList {
        self.movelists.el_mut(self.ply)
    }

    fn movelist(&self) -> &MoveList {
        self.movelists.el(self.ply)
    }

    pub fn set_root_move_randomization(&mut self, state: bool) {
        self.root_move_randomization = state;
    }

    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &Board) -> Option<Move> {
        let ply = self.ply;
        self.movelist_mut().next_move(ply, hh, board)
    }

    pub fn generate_qs_captures(&mut self, board: &Board) {
        self.movelist_mut().generate_qs_captures(board);
    }

    pub fn next_good_capture_move(&mut self, board: &Board) -> Option<Move> {
        self.movelist_mut().next_good_capture_move(board)
    }

    pub fn is_bad_capture_move(&self) -> bool {
        self.movelist().is_bad_capture_move()
    }

    pub fn reset_root_moves(&mut self) {
        self.movelist_mut().reset_root_moves();
    }

    pub fn next_root_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        let randomize = self.root_move_randomization;
        self.movelist_mut().next_root_move(hh, board, randomize)
    }

    pub fn update_root_move(&mut self, m: Move) {
        self.movelist_mut().update_root_move(m);
    }

    pub fn reorder_root_moves(&mut self, best_move: Move, sort_other_moves: bool) {
        self.movelist_mut().reorder_root_moves(best_move, sort_other_moves);
    }

    pub fn prepare_moves(&mut self, active_player: Color, hash_move: Move, move_history: MoveHistory) {
        self.movelist_mut().init(active_player, hash_move, move_history);
    }

    pub fn move_history(&self) -> MoveHistory {
        let curr = self.ply_entry(self.pe_idx());
        let prev_opp = self.ply_entry(self.pe_idx() - 1);

        MoveHistory {
            last_opp: curr.opp_move,
            prev_own: prev_opp.opp_move,
        }
    }
    
    pub fn move_history_hash(&self) -> u16 {
        let curr = self.ply_entry(self.pe_idx());
        let prev_opp = self.ply_entry(self.pe_idx() - 1);

        ((opp_move_hash(curr.opp_move) ^ own_move_hash(prev_opp.opp_move)) & 0xFFFF) as u16
    }

    pub fn root_move_count(&self) -> usize {
        self.movelist().root_move_count()
    }
    
    #[inline(always)]
    fn pe_idx(&self) -> usize {
        self.ply + 4
    }

    fn ply_entry(&self, idx: usize) -> &PlyEntry {
        self.ply_entries.el(idx)
    }

    fn ply_entry_mut(&mut self, idx: usize) -> &mut PlyEntry {
        self.ply_entries.el_mut(idx)
    }

    pub fn is_improving(&self) -> bool {
        let curr_ply = self.ply_entry(self.pe_idx());

        if curr_ply.in_check {
            return false;
        }
        let prev_own_ply = self.ply_entry(self.pe_idx() - 2);
        if self.ply >= 2 && !prev_own_ply.in_check {
            return prev_own_ply.eval < curr_ply.eval;
        }

        let prev_prev_own_ply = self.ply_entry(self.pe_idx() - 4);
        if self.ply >= 4 && !prev_prev_own_ply.in_check {
            return prev_prev_own_ply.eval < curr_ply.eval;
        }

        true
    }

    pub fn eval(&self) -> i16 {
        self.ply_entry(self.pe_idx()).eval
    }

    pub fn in_check(&self) -> bool {
        self.ply_entry(self.pe_idx()).in_check
    }

    pub fn update_next_ply_entry(&mut self, opp_m: Move, gives_check: bool) {
        let entry = self.ply_entry_mut(self.pe_idx() + 1);
        entry.opp_move = opp_m;
        entry.in_check = gives_check;
    }

    pub fn set_eval(&mut self, score: i16) {
        self.ply_entry_mut(self.pe_idx()).eval = score;
    }

    pub fn inc_double_extensions(&mut self) {
        self.ply_entry_mut(self.pe_idx()).double_extensions += 1;
    }

    pub fn double_extensions(&self) -> i16 {
        self.ply_entry(self.pe_idx()).double_extensions
    }

    pub fn has_any_legal_move(&mut self, active_player: Color, hh: &HistoryHeuristics, board: &mut Board) -> bool {
        self.prepare_moves(active_player, NO_MOVE, EMPTY_HISTORY);

        while let Some(m) = self.next_move(hh, board) {
            let (previous_piece, removed_piece_id) = board.perform_move(m);
            let is_legal = !board.is_in_check(active_player);
            board.undo_move(m, previous_piece, removed_piece_id);
            if is_legal {
                return true;
            }
        }

        false
    }
    
    pub fn clear_cutoff_count(&mut self) {
        self.ply_entry_mut(self.pe_idx() + 2).cutoff_count = 0;
    }
    
    pub fn inc_cutoff_count(&mut self) {
        self.ply_entry_mut(self.pe_idx()).cutoff_count += 1;
    }
    
    pub fn next_ply_cutoff_count(&self) -> u32 {
        self.ply_entry(self.pe_idx() + 1).cutoff_count
    }
    
    pub fn ply(&self) -> usize {
        self.ply
    }
}

#[inline(always)]
fn own_move_hash(m: Move) -> u64 {
    piece_zobrist_key(m.move_type().piece_id(), m.end() as usize)
}

#[inline(always)]
fn opp_move_hash(m: Move) -> u64 {
    piece_zobrist_key(-m.move_type().piece_id(), m.end() as usize)
}

#[derive(Copy, Clone, Default)]
pub struct PlyEntry {
    eval: i16,
    in_check: bool,
    opp_move: Move,
    double_extensions: i16,
    cutoff_count: u32,
}

#[macro_export]
macro_rules! next_ply {
    ($ctx:expr, $func_call:expr) => {
        {
            $ctx.enter_ply();
            let result = $func_call;
            $ctx.leave_ply();
            result
        }
    };
}
