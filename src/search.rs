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

use crate::engine::Engine;
use std::time::Instant;
use crate::move_gen::{generate_moves, decode_piece_id, decode_start_index, decode_end_index, Move, NO_MOVE};
use crate::colors::Color;

pub trait Search {
    fn find_best_move(&mut self, min_depth: i32, is_strict_timelimit: bool) -> Move;
    fn sort_by_score(&self, moves: &Vec<Move>, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove>;
    fn evaluate_move_score(&self, active_player: Color, m: Move) -> i32;
    fn gen_legal_moves(&mut self, active_player: Color) -> Vec<ScoredMove>;
}

const MIN_SCORE: i32 = -16383;
const MAX_SCORE: i32 = 16383;

const WHITE_MATE_SCORE: i32 = -16000;
const BLACK_MATE_SCORE: i32 = 16000;

const CANCEL_SEARCH: i32 = i32::max_value() - 1;

const LMR_THRESHOLD: i32 = 4;
const LMR_REDUCTIONS: i32 = 2;

const FUTILITY_MARGIN_MULTIPLIER: i32 = 51;

const FUTILE_MOVE_REDUCTIONS: i32 = 2;
const LOSING_MOVE_REDUCTIONS: i32 = 2;

const QS_SEE_THRESHOLD: i32 = 104;
const QS_PRUNE_MARGIN: i32 = 989;

const PRIMARY_KILLER_SCORE_BONUS: i32 = -2267;
const SECONDARY_KILLER_SCORE_BONUS: i32 = -3350;

const TIMEEXT_MULTIPLIER: i32 = 5;
const TIMEEXT_SCORE_CHANGE_THRESHOLD: i32 = 80;
const TIMEEXT_SCORE_FLUCTUATION_THRESHOLD: i32 = 130;
const TIMEEXT_SCORE_FLUCTUATION_REDUCTIONS: i32 = 90; // reduction percentage per search iteration

const RAZOR_MARGIN: i32 = 130;

impl Search for Engine {
    fn find_best_move(&mut self, min_depth: i32, is_strict_timelimit: bool) -> Move {
        let alpha = MIN_SCORE;
        let beta = MAX_SCORE;

        // TODO: clear history heuristics

        self.starttime = Instant::now();

        let moves = self.gen_legal_moves(self.board.active_player());

        if moves.len() > 0 {
            decode_move(moves[0])
        } else {
            NO_MOVE
        }
    }

    fn gen_legal_moves(&mut self, active_player: Color) -> Vec<ScoredMove> {
        let mut moves= generate_moves(&self.board, active_player);
        {
            let board = &mut self.board;
            moves.retain(|&m| board.is_legal_move(active_player, decode_piece_id(m) as i8, decode_start_index(m), decode_end_index(m)));
        }

        self.sort_by_score(&moves, active_player, 0, 0)
    }

    fn sort_by_score(&self, moves: &Vec<Move>, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove> {
        let scored_moves = moves.into_iter()
            .map(|&m| {
                let score = match m {
                    prrimary_killer => PRIMARY_KILLER_SCORE_BONUS * active_player as i32,
                    secondary_killer => SECONDARY_KILLER_SCORE_BONUS * active_player as i32,
                    _ => self.evaluate_move_score(active_player, m)
                };

                encode_scored_move(m, score)
            })
            .collect();

        scored_moves
    }

    // Move evaluation heuristic for initial move ordering
    // (low values are better for black and high values are better for white)
    fn evaluate_move_score(&self, active_player: Color, m: Move) -> i32 {
        let start = decode_start_index(m);
        let end = decode_end_index(m);
        let captured_piece = self.board.get_item(end);

        match captured_piece {
            EMPTY => {
                let history_score = self.hh.get_history_score(active_player, start, end) * active_player as i32;
                -active_player as i32 * 4096 + history_score
            }
        }

    }

}


type ScoredMove = u32;

fn encode_scored_move(m: Move, score: i32) -> ScoredMove {
    if score < 0 {
        (m | 0x80000000 | ((-score as u32) << 17)) as ScoredMove

    } else {
        (m | ((score as u32) << 17)) as ScoredMove
    }
}

fn decode_score(m: ScoredMove) -> i32 {
    if m & 0x80000000 != 0 {
        -(((m & 0x7FFE0000) >> 17) as i32)
    } else {
        (m >> 17) as i32
    }
}

fn decode_move(m: ScoredMove) -> Move {
    (m & 0x1FFFF) as Move
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_gen::encode_move;
    use crate::pieces::Q;

    #[test]
    fn scored_move_for_max_score() {
        let m = encode_move(Q, 2, 63);
        let scored_move = encode_scored_move(m, MAX_SCORE);

        assert_eq!(m, decode_move(scored_move));
        assert_eq!(MAX_SCORE, decode_score(scored_move));
    }

    #[test]
    fn scored_move_for_min_score() {
        let m = encode_move(Q, 2, 63);
        let scored_move = encode_scored_move(m, MIN_SCORE);

        assert_eq!(m, decode_move(scored_move));
        assert_eq!(MIN_SCORE, decode_score(scored_move));
    }
}
