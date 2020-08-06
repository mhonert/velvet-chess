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

use std::iter::Iterator;
use crate::score_util::{ScoredMove, encode_scored_move, decode_score};
use crate::move_gen::{Move, NO_MOVE, generate_moves, decode_piece_id, decode_start_index, decode_end_index, generate_capture_moves};
use crate::board::Board;
use crate::colors::{Color, WHITE, BLACK};
use crate::history_heuristics::HistoryHeuristics;
use crate::pieces::{EMPTY, K};

const CAPTURE_ORDER_SIZE: usize = 5 + 5 * 8 + 1;

const PRIMARY_KILLER_SCORE_BONUS: i32 = -2267;
const SECONDARY_KILLER_SCORE_BONUS: i32 = -3350;

pub struct SortedMoveGenerator {
    pub capture_order_scores: [i32; CAPTURE_ORDER_SIZE]
}

impl SortedMoveGenerator {
    pub fn new() -> Self {
        SortedMoveGenerator{capture_order_scores: calc_capture_order_scores()}
    }

    pub fn get_capture_order_score(&self, attacker_id: i32, victim_id: i32) -> i32 {
        self.capture_order_scores[((attacker_id - 1) * 8 + (victim_id - 1)) as usize]
    }

    pub fn gen_moves(scored_hash_move: ScoredMove, primary_killer: Move, secondary_killer: Move) -> SortedMoves {
        SortedMoves::new(scored_hash_move, primary_killer, secondary_killer)
    }

    pub fn gen_legal_moves() -> SortedMoves {
        SortedMoves::new(NO_MOVE, NO_MOVE, NO_MOVE)
    }

    pub fn gen_capture_moves() -> SortedMoves {
        SortedMoves::new(NO_MOVE, NO_MOVE, NO_MOVE)
    }
}

pub struct SortedMoves {
    scored_hash_move: ScoredMove,
    primary_killer: Move,
    secondary_killer: Move,
    moves: Option<Vec<ScoredMove>>,
    hash_move_visited: bool,
    index: usize,
}

impl SortedMoves {
    pub fn new(scored_hash_move: ScoredMove, primary_killer: Move, secondary_killer: Move) -> Self {
        SortedMoves{scored_hash_move, primary_killer, secondary_killer, moves: None, hash_move_visited: false, index: 0}
    }

    pub fn reset(&mut self) {
        self.hash_move_visited = false;
        self.index = 0;

    }

    pub fn resort(&mut self) {
        if let Some(moves) = self.moves.as_mut() {
            moves.sort_unstable_by(|&a, &b| decode_score(b).cmp(&decode_score(a)))
        };
    }

    pub fn next_move(&mut self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &mut Board) -> Option<ScoredMove> {
        if !self.hash_move_visited {
            self.hash_move_visited = true;
            if self.scored_hash_move != NO_MOVE {
                return Some(self.scored_hash_move);
            }
        }

        if self.moves == None {
            self.moves = Some(self.gen_moves(gen, hh, board));
        }

        self.get_next_move()
    }

    pub fn next_legal_move(&mut self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &mut Board) -> Option<ScoredMove> {
        if self.moves == None {
            self.moves = Some(self.gen_legal_moves(gen, hh, board));
        }

        self.get_next_move()
    }

    pub fn next_capture_move(&mut self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &mut Board) -> Option<ScoredMove> {
        if self.moves == None {
            self.moves = Some(self.gen_capture_moves(gen, hh, board));
        }

        self.get_next_move()
    }

    pub fn update_move(&mut self, scored_move: ScoredMove) {
        if let Some(moves) = self.moves.as_mut() {
            moves[self.index - 1] = scored_move;
        };
    }

    fn get_next_move(&mut self) -> Option<u32> {
        match &self.moves {
            Some(moves) => {
                if self.index >= moves.len() {
                    None
                } else {
                    self.index += 1;
                    Some(moves[self.index - 1])
                }
            },

            None => None
        }
    }

    fn gen_legal_moves(&mut self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &mut Board) -> Vec<ScoredMove> {
        let active_player = board.active_player();
        let mut moves = generate_moves(board, active_player);
        {
            moves.retain(|&m| board.is_legal_move(active_player, decode_piece_id(m) as i8, decode_start_index(m), decode_end_index(m)));
        }

        self.sort_by_score(gen, hh, board, &moves, active_player, 0, 0)
    }

    fn gen_moves(&self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &Board) -> Vec<ScoredMove> {
        let active_player = board.active_player();
        let moves = generate_moves(board, active_player);
        self.sort_by_score(gen, hh, board, &moves, active_player, self.primary_killer, self.secondary_killer)
    }

    fn gen_capture_moves(&self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &Board) -> Vec<ScoredMove> {
        let active_player = board.active_player();
        let moves = generate_capture_moves(board, active_player);
        self.sort_by_score(gen, hh, board, &moves, active_player, self.primary_killer, self.secondary_killer)
    }

    fn sort_by_score(&self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &Board, moves: &Vec<Move>, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove> {
        let mut scored_moves: Vec<ScoredMove> = moves.iter()
            .map(|&m| {
                let score = if m == primary_killer {
                    PRIMARY_KILLER_SCORE_BONUS * active_player as i32
                } else if m == secondary_killer {
                    SECONDARY_KILLER_SCORE_BONUS * active_player as i32
                } else {
                    self.evaluate_move_score(gen, hh, board, active_player, m)
                };

                encode_scored_move(m, score)
            })
            .collect();

        if active_player == WHITE {
            scored_moves.sort_unstable_by(|&a, &b| decode_score(b).cmp(&decode_score(a)))
        } else {
            scored_moves.sort_unstable_by(|&a, &b| decode_score(a).cmp(&decode_score(b)))
        }

        scored_moves
    }

    // Move evaluation heuristic for initial move ordering
    // (low values are better for black and high values are better for white)
    fn evaluate_move_score(&self, gen: &SortedMoveGenerator, hh: &HistoryHeuristics, board: &Board, active_player: Color, m: Move) -> i32 {
        let start = decode_start_index(m);
        let end = decode_end_index(m);
        let captured_piece = board.get_item(end);

        if captured_piece == EMPTY {
            let history_score = hh.get_history_score(active_player, start, end) * active_player as i32;
            return -active_player as i32 * 4096 + history_score;
        }

        let original_piece_id = board.get_item(start).abs();
        let captured_piece_id = captured_piece.abs();

        if original_piece_id == 0 {
            eprintln!("Move {}/{} from {} to {}: capture {}", original_piece_id, decode_piece_id(m), start, end, captured_piece_id);
            println!("{}, {}", board.king_pos(WHITE), board.king_pos(BLACK));
            for i in 0..64 {
                if board.get_item(i).abs() == K {
                    println!("King at {}", i);
                }
            }
        }

        active_player as i32 * gen.get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
    }

}

fn calc_capture_order_scores() -> [i32; CAPTURE_ORDER_SIZE] {
    let mut scores: [i32; CAPTURE_ORDER_SIZE] = [0; CAPTURE_ORDER_SIZE];
    let mut score: i32 = 0;

    for victim in 0..=5 {
        for attacker in (0..=5).rev() {
            scores[(victim + attacker * 8) as usize] = score * 64;
            score += 1;
        }
    }

    scores
}


