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
use crate::board::Board;
use crate::move_gen::{decode_end_index, decode_piece_id, decode_start_index, generate_moves};
use crate::transposition_table::{TranspositionTable, get_scored_move, EXACT, get_depth};
use crate::score_util::ScoredMove;

/* Perft (performance test, move path enumeration) test helper function to verify the move generator.
  It generates all possible moves up to the specified depth and counts the number of leaf nodes.
  This number can then be compared to precalculated numbers that are known to be correct

  Another use case for this function is to test the performance of the move generator.
*/
pub fn perft(tt: &mut TranspositionTable, board: &mut Board, depth: i32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let entry = tt.get_entry(board.get_hash());
    if entry != 0 && get_depth(entry) == depth {
        return get_scored_move(entry) as u64;
    }

    let mut nodes: u64 = 0;

    let active_player = board.active_player();
    let moves = generate_moves(board, active_player);

    for m in moves {
        let target_piece_id = decode_piece_id(m);
        let move_start = decode_start_index(m);
        let move_end = decode_end_index(m);

        let previous_piece = board.get_item(move_start);
        let removed_piece = board.perform_move(target_piece_id as i8, move_start, move_end);

        if !board.is_in_check(active_player) {
            nodes += perft(tt, board, depth - 1);
        }

        board.undo_move(previous_piece, move_start, move_end, removed_piece);
    }

    tt.write_entry(board.get_hash(), depth, nodes as ScoredMove, EXACT);

    return nodes;
}
