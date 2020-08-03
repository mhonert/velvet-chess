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
use std::time::{Instant, Duration};
use crate::move_gen::{generate_moves, decode_piece_id, decode_start_index, decode_end_index, Move, NO_MOVE, is_valid_move, has_valid_moves};
use crate::colors::{Color, WHITE, BLACK};
use crate::pieces::EMPTY;
use std::cmp::{min, max};
use crate::uci_move::UCIMove;
use crate::transposition_table::{MAX_DEPTH, get_scored_move, get_depth, get_score_type, EXACT, UPPER_BOUND, LOWER_BOUND};
use crate::score_util::{ScoredMove, decode_move, encode_scored_move, decode_score};
use crate::board::{EN_PASSANT, Board};

pub trait Search {
    fn find_best_move(&mut self, min_depth: i32, is_strict_timelimit: bool) -> Move;

    fn rec_find_best_move(&mut self, alpha: i32, beta: i32, player_color: Color, depth: i32, ply: i32, nullmove_performed: bool,
                          nullmove_verification: bool, is_in_check: bool) -> i32;

    fn quiescence_search(&mut self, player_color: Color, alpha: i32, beta: i32, ply: i32) -> i32;

    fn sort_by_score(&self, moves: &Vec<Move>, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove>;

    fn evaluate_move_score(&self, active_player: Color, m: Move) -> i32;

    fn gen_legal_moves(&mut self, active_player: Color) -> Vec<ScoredMove>;

    fn gen_moves(&mut self, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove>;

    fn get_base_stats(&self, duration: Duration) -> String;

    fn extract_pv(&mut self, m: Move, depth: i32) -> String;

    fn terminal_score(&mut self, active_player: Color, ply: i32) -> i32;
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
        let mut alpha = MIN_SCORE;
        let beta = MAX_SCORE;

        self.hh.clear();

        self.starttime = Instant::now();

        let mut moves = self.gen_legal_moves(self.board.active_player());
        if moves.len() == 1 {
            return decode_move(moves[0]);
        }

        if moves.is_empty() {
            // no more moves possible (i.e. check mate or stale mate
            return NO_MOVE
        }

        self.cancel_possible = false;
        self.node_count = 0;
        self.log_every_second = false;

        let mut current_best_move: ScoredMove = NO_MOVE;

        let mut already_extended_timelimit = false;
        let mut iteration_duration: i64 = 0;

        let mut previous_best_move: Move = NO_MOVE;
        let mut previous_best_score: i32 = 0;

        let mut fluctuation_count: i32 = 0;
        let mut score_fluctuations: i32 = 0;

        let player_color = self.board.active_player();

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in min(min_depth, 2)..=(MAX_DEPTH as i32) {
            let mut best_score: i32 = MIN_SCORE;
            let mut scored_moves = 0;

            let mut previous_alpha = alpha;
            let mut previous_beta = beta;

            let iteration_start_time = Instant::now;

            let mut best_move: Move = 0;

            let mut a = -beta; // Search principal variation node with full window

            let mut evaluated_move_count = 0;

            let mut iteration_cancelled = false;

            for move_num in 0..moves.len() {
                let scored_move = moves[move_num];
                let m = decode_move(scored_move);

                if depth > 12 {
                    let now = Instant::now();

                    let total_duration = now.duration_since(self.starttime);
                    if total_duration.as_millis() >= 1000 {
                        self.log_every_second = true;
                        self.last_log_time = now;
                        self.current_depth = depth;
                        let uci_move = UCIMove::from_encoded_move(&self.board, m).to_uci();
                        let base_stats = self.get_base_stats(total_duration);
                        println!("info depth {} {} currmove {} currmovenumber {}", depth, base_stats, uci_move, move_num);
                    }
                }

                let target_piece_id = decode_piece_id(m);
                let start = decode_start_index(m);
                let end = decode_end_index(m);
                let previous_piece = self.board.get_item(start);

                let removed_piece_id = self.board.perform_move(target_piece_id as i8, start, end);

                let gives_check = self.board.is_in_check(-player_color);

                // Use principal variation search
                let mut result = self.rec_find_best_move(a, -alpha, -player_color, depth - 1, 1, false, true, gives_check);
                if result == CANCEL_SEARCH {
                    iteration_cancelled = true;
                } else {

                    // Repeat search if it falls outside the window
                    if -result > alpha && -result < beta {
                        result = self.rec_find_best_move(-beta, -alpha, -player_color, depth - 1, 1, false, true, gives_check);
                        if result == CANCEL_SEARCH {
                            iteration_cancelled = true;
                        }
                    }
                }

                self.board.undo_move(previous_piece, start, end, removed_piece_id);

                if iteration_cancelled {
                    if best_move != NO_MOVE && previous_best_move != NO_MOVE {
                        score_fluctuations = score_fluctuations * 100 / TIMEEXT_SCORE_FLUCTUATION_REDUCTIONS;
                        score_fluctuations += (best_score - previous_best_score).abs();

                        if best_score.abs() >= (BLACK_MATE_SCORE - MAX_DEPTH as i32) {
                            // Reset score fluctuation statistic, if a check mate is found
                            score_fluctuations = 0;
                        }
                    }

                    if !is_strict_timelimit
                        && !already_extended_timelimit
                        && should_extend_timelimit(best_move, best_score, previous_best_move, previous_best_score, score_fluctuations, fluctuation_count) {

                        already_extended_timelimit = true;
                        self.timelimit_ms *= TIMEEXT_MULTIPLIER;

                        iteration_cancelled = false;
                        continue;
                    }
                    break;
                }

                let score = -result;
                if score > best_score {
                    best_score = score;
                    best_move = m;
                    alpha = max(alpha, best_score);

                    if self.log_every_second {
                        let pv = self.extract_pv(best_move, depth - 1);
                        println!("depth {} score {} pv {}", depth, get_score_info(best_score), pv)
                    }
                }

                evaluated_move_count += 1;
                a = -(alpha + 1); // Search all other moves (after principal variation) with a zero window

                moves[move_num] = encode_scored_move(m, score);
                scored_moves += 1;
            }
        }

        decode_move(moves[0])
    }

    // Recursively calls itself with alternating player colors to
    // find the best possible move in response to the current board position.
    fn rec_find_best_move(&mut self, mut alpha: i32, beta: i32, player_color: Color, mut depth: i32, ply: i32, nullmove_performed: bool,
                          mut nullmove_verification: bool, is_in_check: bool) -> i32 {
        if self.node_count & 1023 == 0 && self.cancel_possible {
            let current_time = Instant::now();
            let total_duration = current_time.duration_since(self.starttime);
            if total_duration.as_millis() as i32 >= self.timelimit_ms {
                // Cancel search if the time limit has been reached or exceeded
                return CANCEL_SEARCH;
            }

            if depth > 3 && self.log_every_second && current_time.duration_since(self.last_log_time).as_millis() >= 1000 {
                self.last_log_time = current_time;
                let base_stats = self.get_base_stats(total_duration);
                println!("info depth {}{}", depth, base_stats);
            }
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        if self.board.is_engine_draw() {
            self.node_count += 1;
            return 0;
        }

        if is_in_check {
            // Extend search when in check
            if depth < 0 {
                depth = 1;
            } else {
                depth -= 1;
            }
        } else if depth == 1 && (self.board.get_score() * player_color as i32) < alpha - RAZOR_MARGIN {
            // Directly jump to quiescence search, if current position score is below a certain threshold
            depth = 0;
        }

        // Quiescence search
        if depth < 0 {
            let score = self.quiescence_search(player_color, alpha, beta, ply);
            if score == BLACK_MATE_SCORE {
                return score - ply;
            } else if score == WHITE_MATE_SCORE {
                return score + ply;
            }
            return score;
        }

        self.node_count += 1;

        // Check transposition table
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get_entry(hash);

        let mut scored_move = get_scored_move(tt_entry);
        let mut moves: Option<Vec<Move>> = None;

        let mut m = NO_MOVE;
        let mut hash_move = NO_MOVE;
        let mut move_index = 0;

        if scored_move != NO_MOVE {
            if get_depth(tt_entry) >= depth {
                let score = decode_score(scored_move);

                match get_score_type(tt_entry) {
                    EXACT => {
                        return score;
                    },
                    UPPER_BOUND => {
                        if score <= alpha {
                            return alpha;
                        }
                    },

                    LOWER_BOUND => {
                        if score >= beta {
                            return beta;
                        }
                    },
                    _ => ()
                };
            }

            m = decode_move(scored_move);

            // Validate hash move for additional protection against hash collisions
            if m == NO_MOVE || !is_valid_move(&mut self.board, player_color, m) {
                scored_move = NO_MOVE;
                m = NO_MOVE;
            }

            hash_move = m;
        }

        let mut fail_high = false;

        // Null move pruning
        if !is_pv && !nullmove_performed && depth > 2 && !is_in_check {
            self.board.perform_null_move();
            let result = self.rec_find_best_move(-beta, -beta + 1, -player_color, depth - 4, ply + 1, true, false, false);
            self.board.undo_null_move();
            if result == CANCEL_SEARCH {
                return CANCEL_SEARCH;
            }
            if -result >= beta {
                if nullmove_verification {
                    depth -= 1;
                    nullmove_verification = false;
                    fail_high = true;
                } else {
                    return -result;
                }
            }
        }

        let primary_killer = self.hh.get_primary_killer(ply);
        let secondary_killer = self.hh.get_secondary_killer(ply);

        if scored_move == 0 {
            // Generate moves, if no valid moves were found in the transposition table
            let generated_moves = self.gen_moves(player_color, primary_killer, secondary_killer);
            if generated_moves.len() == 0 {
                // no more moves possible (i.e. check mate or stale mate)
                return self.terminal_score(player_color, ply) * player_color as i32;
            }
            scored_move = generated_moves[0];
            moves = Some(generated_moves);
            m = decode_move(scored_move);
            move_index += 1;
        }

        let mut best_move = NO_MOVE;
        let mut best_score = MIN_SCORE;
        let mut score_type = UPPER_BOUND;
        let mut evaluated_move_count = 0;
        let mut has_valid_moves = false;

        let allow_reductions = depth > 2 && !is_in_check;

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        let mut prune_low_score = 0;
        if !is_pv && depth <= 4 {
            prune_low_score = self.board.get_score() * player_color as i32 + depth * FUTILITY_MARGIN_MULTIPLIER;
            allow_futile_move_pruning = prune_low_score <= alpha;
        }

        let mut gives_check = false;

        loop {
            let target_piece_id = decode_piece_id(m);
            let start = decode_start_index(m);
            let end = decode_end_index(m);
            let previous_piece = self.board.get_item(start);

            let move_state = self.board.perform_move(target_piece_id as i8, start, end);
            let removed_piece_id = move_state & !EN_PASSANT;

            let mut skip = self.board.is_in_check(player_color); // skip if move would put own king in check

            let mut reductions = 0;

            if !skip {
                has_valid_moves = true;
                gives_check = self.board.is_in_check(-player_color);

                if removed_piece_id == EMPTY {
                    let has_negative_history = self.hh.has_negative_history(player_color, depth, start, end);
                    let own_moves_left = (depth + 1) / 2;
                    if !gives_check && allow_reductions && evaluated_move_count > LMR_THRESHOLD
                        && !self.board.is_pawn_move_close_to_promotion(previous_piece, end, own_moves_left - 1) {
                        // Reduce search depth for late moves (i.e. after trying the most promising moves)
                        reductions = LMR_REDUCTIONS;
                        if has_negative_history || self.board.see_score(-player_color, start, end, target_piece_id, EMPTY) < 0 {
                            // Reduce more, if move has negative history or SEE score
                            reductions += 1;
                        }
                    } else if !gives_check && allow_futile_move_pruning && target_piece_id as i8 == previous_piece.abs() {
                        if own_moves_left <= 1 || (has_negative_history && self.board.see_score(-player_color, start, end, target_piece_id, EMPTY) < 0) {
                            // Prune futile move
                            skip = true;
                            if prune_low_score > best_score {
                                best_move = NO_MOVE;
                                best_score = prune_low_score; // remember score with added margin for cases when all moves are pruned
                            }
                        } else {
                            // Reduce futile move
                            reductions = FUTILE_MOVE_REDUCTIONS;
                        }
                    } else if has_negative_history || self.board.see_score(-player_color, start, end, target_piece_id, EMPTY) < 0 {
                        // Reduce search depth for moves with negative history or negative SEE score
                        reductions = LOSING_MOVE_REDUCTIONS;
                    }
                } else if removed_piece_id <= previous_piece.abs() && self.board.see_score(-player_color, start, end, target_piece_id, removed_piece_id) < 0 {
                    // Reduce search depth for moves with negative capture moves with negative SEE score
                    reductions = LOSING_MOVE_REDUCTIONS;
                }
            }

            if skip {
                self.board.undo_move(previous_piece, start, end, move_state);

            } else {
                if removed_piece_id == EMPTY {
                    self.hh.update_played_moves(player_color, start, end);
                }

                has_valid_moves = true;
                evaluated_move_count += 1;

                let a = if evaluated_move_count > 1 { -(alpha + 1) } else { -beta };
                let mut result = self.rec_find_best_move(a, -alpha, -player_color, depth - reductions - 1, ply + 1, false, nullmove_verification, gives_check);
                if result == CANCEL_SEARCH {
                    self.board.undo_move(previous_piece, start, end, move_state);
                    return CANCEL_SEARCH;
                }

                if -result > alpha && (-result < beta || reductions > 0) {
                    // Repeat search without reduction and with full window
                    result = self.rec_find_best_move(-beta, -alpha, -player_color, depth - 1, ply + 1, false, nullmove_verification, gives_check);
                    if result == CANCEL_SEARCH {
                        self.board.undo_move(previous_piece, start, end, move_state);
                        return CANCEL_SEARCH;
                    }
                }

                let score = -result;
                self.board.undo_move(previous_piece, start, end, move_state);

                let mut improved_alpha = false;

                if score > best_score {
                    best_score = score;
                    best_move = m;

                    // Alpha-beta pruning
                    if best_score > alpha {
                        alpha = best_score;
                        score_type = EXACT;
                        improved_alpha = true;
                    }

                    if alpha >= beta {
                        self.tt.write_entry(hash, depth, encode_scored_move(best_move, best_score), LOWER_BOUND);

                        if removed_piece_id == EMPTY {
                            self.hh.update(ply, player_color, start, end, best_move);
                        }

                        return alpha;
                    }
                }
            }

            if moves == None {
                let generated_moves = self.gen_moves(player_color, primary_killer, secondary_killer);
                if generated_moves.len() == 0 {
                    // no more moves possible
                    break;
                }
                moves = Some(generated_moves);
            } else if move_index == moves.unwrap().len() {
                if fail_high && has_valid_moves {
                    // research required, because a Zugzwang position was detected (fail-high report by null search, but no found cutoff)
                    depth += 1;
                    nullmove_verification = true;
                    fail_high = false;
                    move_index = 0;
                    m = best_move;
                    continue;

                } else {
                    // Last move has been evaluated
                    break;
                }
            }

            let gen_moves = moves.unwrap();
            scored_move = gen_moves[move_index];
            m = decode_move(scored_move);

            move_index += 1;
            while move_index < gen_moves.len() && decode_move(gen_moves[move_index]) == hash_move {
                move_index += 1;
            }
        }

        if !has_valid_moves {
            if is_in_check {
                // Check mate
                return WHITE_MATE_SCORE + ply;
            }

            // Stale mate
            return 0;
        }

        self.tt.write_entry(hash, depth, encode_scored_move(best_move, best_score), score_type);

        best_score
    }

    fn quiescence_search(&mut self, active_player: Color, alpha: i32,  beta: i32, ply: i32) -> i32 {
        self.node_count += 1;

        if self.board.is_engine_draw() {
            return 0;
        }

        let position_score = self.board.get_score() * active_player as i32;
        position_score
    }

    // If a check mate position can be achieved, then earlier check mates should have a better score than later check mates
    // to prevent unnecessary delays.
    fn terminal_score(&mut self, active_player: Color, ply: i32) -> i32 {
        if active_player == WHITE {
            if is_check_mate(&mut self.board, WHITE) {
                return WHITE_MATE_SCORE + ply;
            } else {
                return 0; // Stale mate
            }
        }

        if is_check_mate(&mut self.board, BLACK) {
            return BLACK_MATE_SCORE - ply;
        }

        0 // Stale mate
    }

    fn extract_pv(&mut self, m: Move, depth: i32) -> String {
        let uci_move = UCIMove::from_encoded_move(&self.board, m).to_uci();
        if depth == 0 {
            return uci_move;
        }

        let target_piece_id = decode_piece_id(m);
        let start = decode_start_index(m);
        let end = decode_end_index(m);
        let previous_piece = self.board.get_item(start);

        let removed_piece_id = self.board.perform_move(target_piece_id as i8, start, end);

        let entry = self.tt.get_entry(self.board.get_hash());
        let next_move = if entry != 0 { decode_move(get_scored_move(entry)) } else { NO_MOVE };

        let active_player = self.board.active_player();
        let is_valid_followup_move = next_move != NO_MOVE && is_valid_move(&mut self.board, active_player, next_move);
        let followup_uci_moves = if is_valid_followup_move {
            format!(" {}", self.extract_pv(next_move, depth - 1))
        } else {
            String::from("")
        };

        self.board.undo_move(previous_piece, start, end, removed_piece_id);

        format!("{}{}", uci_move, followup_uci_moves)
    }

    fn get_base_stats(&self, duration: Duration) -> String {
        let duration_micros = duration.as_micros();
        let nodes_per_second = if duration_micros > 0 {  self.node_count as u128 * 1_000_000 / duration_micros } else { 0 };

        if nodes_per_second > 0 {
            format!(" nodes {} nps {} time {}", self.node_count, nodes_per_second, duration_micros / 1000)
        } else {
            format!(" nodes {} time {}", self.node_count, duration_micros / 1000)
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

    fn gen_moves(&mut self, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove> {
        let mut moves= generate_moves(&self.board, active_player);
        self.sort_by_score(&moves, active_player, primary_killer, secondary_killer)
    }


    fn sort_by_score(&self, moves: &Vec<Move>, active_player: Color, primary_killer: Move, secondary_killer: Move) -> Vec<ScoredMove> {
        let mut scored_moves: Vec<ScoredMove> = moves.iter()
            .map(|&m| {
                let score = if m == primary_killer {
                    PRIMARY_KILLER_SCORE_BONUS * active_player as i32
                } else if m == secondary_killer {
                    SECONDARY_KILLER_SCORE_BONUS * active_player as i32
                } else {
                    self.evaluate_move_score(active_player, m)
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
    fn evaluate_move_score(&self, active_player: Color, m: Move) -> i32 {
        let start = decode_start_index(m);
        let end = decode_end_index(m);
        let captured_piece = self.board.get_item(end);

        if captured_piece == EMPTY {
            let history_score = self.hh.get_history_score(active_player, start, end) * active_player as i32;
            return -active_player as i32 * 4096 + history_score;
        }

        let original_piece_id = self.board.get_item(start).abs();
        let captured_piece_id = captured_piece.abs();

        active_player as i32 * self.get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
    }

}

fn is_check_mate(board: &mut Board, player_color: Color) -> bool {
    if !board.is_in_check(player_color) {
        return false;
    }

    has_valid_moves(board, player_color)
}

fn should_extend_timelimit(new_move: Move, new_score: i32, previous_move: Move, previous_score: i32, score_fluctuations: i32, fluctuation_count: i32) -> bool {
   if previous_move == 0 || new_move == 0 {
       return false;
   }

    let avg_fluctuations = if fluctuation_count > 0 { score_fluctuations / fluctuation_count } else { 0 };

    new_move != previous_move
        || (new_score - previous_score).abs() >= TIMEEXT_SCORE_CHANGE_THRESHOLD
        || avg_fluctuations >= TIMEEXT_SCORE_FLUCTUATION_THRESHOLD
}

fn get_score_info(score: i32) -> String {
    if score <= WHITE_MATE_SCORE + MAX_DEPTH as i32{
        return format!("mate {}", (WHITE_MATE_SCORE - score - 1) / 2);
    } else if score >= BLACK_MATE_SCORE - MAX_DEPTH as i32 {
        return format!("mate {}", (BLACK_MATE_SCORE - score + 1) / 2);
    }

    format!("cp {}", score)
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
