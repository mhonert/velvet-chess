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
use crate::board::Board;
use crate::history_heuristics::{EMPTY_HISTORY, HistoryHeuristics};
use crate::moves::NO_MOVE;
use crate::next_ply;
use crate::search_context::SearchContext;

/* Perft (performance test, move path enumeration) test helper function to verify the move generator.
  It generates all possible moves up to the specified depth and counts the number of leaf nodes.
  This number can then be compared to precalculated numbers that are known to be correct

  Another use case for this function is to test the performance of the move generator.
*/
pub fn perft(ctx: &mut SearchContext, hh: &HistoryHeuristics, board: &mut Board, depth: i32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes: u64 = 0;

    let active_player = board.active_player();
    ctx.prepare_moves(active_player, NO_MOVE, EMPTY_HISTORY);

    let in_check = board.is_in_check(active_player);

    while let Some(m) = ctx.next_move(0, hh, board) {
        let (previous_piece, removed_piece_id) = board.perform_move(m);

        if !board.is_left_in_check(active_player, in_check, m) {
            nodes += next_ply!(ctx, perft(ctx, hh, board, depth - 1));
        }

        board.undo_move(m, previous_piece, removed_piece_id);
    }

    nodes
}
