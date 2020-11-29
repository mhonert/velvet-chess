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

use crate::board::{Board, interpolate_score};
use crate::colors::{Color};
use crate::history_heuristics::HistoryHeuristics;
use crate::move_gen::{decode_end_index, decode_piece_id, decode_start_index, generate_capture_moves, generate_moves, Move, NO_MOVE};
use crate::pieces::{EMPTY};
use crate::score_util::{decode_score, encode_scored_move, ScoredMove, unpack_score, unpack_eg_score, decode_move};

const CAPTURE_ORDER_SIZE: usize = 5 + 5 * 8 + 1;

const PRIMARY_KILLER_SCORE_BONUS: i32 = -2267;
const SECONDARY_KILLER_SCORE_BONUS: i32 = -3350;

const CAPTURE_ORDER_SCORES: [i32; CAPTURE_ORDER_SIZE] = calc_capture_order_scores();

#[inline]
fn get_capture_order_score(attacker_id: i32, victim_id: i32) -> i32 {
    CAPTURE_ORDER_SCORES[((attacker_id - 1) * 8 + (victim_id - 1)) as usize]
}

enum Stage {
    HashMove,
    GeneratedMoves,
}

pub struct SortedMoves {
    scored_hash_move: ScoredMove,
    primary_killer: Move,
    secondary_killer: Move,
    moves: Option<Vec<ScoredMove>>,
    stage: Stage,
    index: usize,
}

impl SortedMoves {
    pub fn new(scored_hash_move: ScoredMove, primary_killer: Move, secondary_killer: Move) -> Self {
        SortedMoves {
            scored_hash_move,
            primary_killer,
            secondary_killer,
            moves: None,
            stage: Stage::HashMove,
            index: 0
        }
    }

    pub fn reset(&mut self) {
        self.stage = Stage::HashMove;
    }

    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<ScoredMove> {
        match self.stage {
            Stage::HashMove => {
                self.index = 0;
                self.stage = Stage::GeneratedMoves;

                if self.scored_hash_move != NO_MOVE {
                    Some(self.scored_hash_move)
                } else {
                    self.next_move(hh, board)
                }
            },

            Stage::GeneratedMoves => {
                if self.moves == None {
                   self.moves = Some(gen_moves(hh, board, self.primary_killer, self.secondary_killer));
                }

                match &self.moves {
                    Some(moves) => {
                        while self.index < moves.len() {
                            self.index += 1;
                            let m = unsafe { *moves.get_unchecked(self.index - 1) };
                            if decode_move(m) != decode_move(self.scored_hash_move) {

                                return Some(m);
                            }
                        }
                    },

                    None => ()
                }

                None
            }

        }
    }

    pub fn next_capture_move(&mut self, board: &mut Board) -> Option<ScoredMove> {
        if self.moves == None {
            self.moves = Some(gen_capture_moves(board));
        }

        match &self.moves {
            Some(moves) => {
                if self.index < moves.len() {
                    self.index += 1;
                    let m = unsafe { *moves.get_unchecked(self.index - 1) };
                    return Some(m);
                }
                None
            }

            None => None,
        }
    }

}

fn gen_moves(hh: &HistoryHeuristics, board: &Board, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove> {
    let active_player = board.active_player();
    let mut moves = generate_moves(board, active_player);
    sort_by_score(hh, board, &mut moves, primary_killer, secondary_killer, active_player);
    moves
}

fn gen_capture_moves(board: &Board) -> Vec<ScoredMove> {
    let active_player = board.active_player();
    let mut moves = generate_capture_moves(board, active_player);

    for m in moves.iter_mut() {
        let score = evaluate_capture_move_score(board, active_player, *m);
        *m = encode_scored_move(*m, score);
    }

    sort_by_score_desc(&mut moves);

    moves
}

fn sort_by_score(
    hh: &HistoryHeuristics,
    board: &Board,
    moves: &mut Vec<Move>,
    primary_killer: Move,
    secondary_killer: Move,
    active_player: Color
) {
    let phase = board.calc_phase_value();
    for m in moves.iter_mut() {
        let score = if *m == primary_killer {
            PRIMARY_KILLER_SCORE_BONUS
        } else if *m == secondary_killer {
            SECONDARY_KILLER_SCORE_BONUS
        } else {
            evaluate_move_score(phase, hh,  board, active_player, *m)
        };
        *m = encode_scored_move(*m, score);
    }

    sort_by_score_desc(moves);
}

fn evaluate_capture_move_score(board: &Board, active_player: Color, m: Move) -> i32 {
    let start = decode_start_index(m);
    let end = decode_end_index(m);
    let captured_piece = board.get_item(end);

    let original_piece_id = active_player * board.get_item(start);
    let captured_piece_id = captured_piece.abs();

    get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
}

const fn calc_capture_order_scores() -> [i32; CAPTURE_ORDER_SIZE] {
    let mut scores: [i32; CAPTURE_ORDER_SIZE] = [0; CAPTURE_ORDER_SIZE];
    let mut score: i32 = 0;

    let mut victim = 0;
    while victim <= 5 {

        let mut attacker = 5;
        while attacker >= 0 {
            scores[(victim + attacker * 8) as usize] = score * 64;
            score += 1;

            attacker -= 1;
        }

        victim += 1;
    }

    scores
}

pub struct SortedLegalMoves {
    moves: Option<Vec<ScoredMove>>,
    index: usize,
}

impl SortedLegalMoves {
    pub fn new() -> Self {
        SortedLegalMoves {
            moves: None,
            index: 0,
        }
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }

    pub fn resort(&mut self) {
        if let Some(moves) = self.moves.as_mut() {
            sort_by_score_desc(moves);
        };
    }

    pub fn next_legal_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<ScoredMove> {
        if self.moves == None {
            self.moves = Some(gen_legal_moves(hh, board));
        }

        match &self.moves {
            Some(moves) => {
                if self.index < moves.len() {
                    self.index += 1;
                    let m = moves[self.index - 1];
                    return Some(m);
                }
                None
            }

            None => None,
        }
    }

    pub fn update_move(&mut self, scored_move: ScoredMove) {
        if let Some(moves) = self.moves.as_mut() {
            moves[self.index - 1] = scored_move;
        };
    }

}

fn gen_legal_moves(hh: &HistoryHeuristics, board: &mut Board) -> Vec<ScoredMove> {
    let active_player = board.active_player();
    let mut moves = generate_moves(board, active_player);
    {
        moves.retain(|&m| {
            board.is_legal_move(
                active_player,
                decode_piece_id(m) as i8,
                decode_start_index(m),
                decode_end_index(m),
            )
        });
    }

    let phase = board.calc_phase_value();
    for m in moves.iter_mut() {
        let score = evaluate_move_score(phase, hh,  board, active_player, *m);
        *m = encode_scored_move(*m, score);
    }

    sort_by_score_desc(&mut moves);
    moves
}

// Move evaluation heuristic for initial move ordering (high values are better for the active player)
fn evaluate_move_score(
    phase: i32,
    hh: &HistoryHeuristics,
    board: &Board,
    active_player: Color,
    m: Move,
) -> i32 {
    let start = decode_start_index(m);
    let end = decode_end_index(m);
    let captured_piece = board.get_item(end);

    if captured_piece == EMPTY {
        let history_score = hh.get_history_score(active_player, start, end);
        return if history_score == -1 {
            // No history score -> use difference between piece square scores
            let original_piece = board.get_item(start);

            let start_packed_score = board.pst.get_packed_score(original_piece, start as usize);
            let end_packed_score = board.pst.get_packed_score(original_piece, end as usize);

            let mg_diff = (unpack_score(end_packed_score) - unpack_score(start_packed_score)) as i32;
            let eg_diff = (unpack_eg_score(end_packed_score) - unpack_eg_score(start_packed_score)) as i32;

            let diff = interpolate_score(phase, mg_diff, eg_diff) * active_player as i32;

            -4096 + diff
        } else if history_score == 0 {

            -5000
        } else {

            -3600 + history_score
        }
    }

    let original_piece_id = board.get_item(start).abs();
    let captured_piece_id = captured_piece.abs();

    get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
}

fn sort_by_score_desc(moves: &mut Vec<Move>) {
    // Basic insertion sort
    for i in 1..moves.len() {
        let x = unsafe { *moves.get_unchecked(i) };
        let x_score = decode_score(x);
        let mut j = i as i32 - 1;
        while j >= 0 {
            let y = unsafe { *moves.get_unchecked(j as usize) };
            if decode_score(y) >= x_score {
                break;
            }

            unsafe { *moves.get_unchecked_mut(j as usize + 1) = y };
            j -= 1;
        }
        unsafe { *moves.get_unchecked_mut((j + 1) as usize) = x };
    }
}

