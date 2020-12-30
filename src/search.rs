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

use crate::colors::{Color, BLACK, WHITE};
use crate::engine::Engine;
use crate::pieces::{EMPTY, R};
use crate::score_util::{
    BLACK_MATE_SCORE, MAX_SCORE,
    MIN_SCORE, WHITE_MATE_SCORE,
};
use crate::transposition_table::{get_depth, get_score_type, get_untyped_move, MAX_DEPTH, ScoreType};
use crate::uci_move::UCIMove;
use std::cmp::{max, min};
use std::time::{Duration, Instant};
use crate::eval::Eval;
use crate::moves::{Move, NO_MOVE};
use crate::move_gen::is_likely_valid_move;

pub trait Search {
    fn find_best_move(&mut self, min_depth: i32, is_strict_timelimit: bool) -> Move;

    fn rec_find_best_move(&mut self, alpha: i32, beta: i32, player_color: Color, depth: i32, ply: i32, null_move_performed: bool, is_in_check: bool) -> i32;

    fn quiescence_search(&mut self, player_color: Color, alpha: i32, beta: i32, ply: i32) -> i32;

    fn get_base_stats(&self, duration: Duration) -> String;

    fn extract_pv(&mut self, m: Move, depth: i32) -> String;

    fn make_quiet_position(&mut self) -> bool;

    fn static_quiescence_search(&mut self, alpha: i32, beta: i32, ply: i32) -> i32;

    fn is_quiet_position(&mut self) -> bool;

    fn is_quiet_pv(&mut self, m: Move, depth: i32) -> bool;
}

const CANCEL_SEARCH: i32 = i32::max_value() - 1;

const LMR_THRESHOLD: i32 = 4;
const LMR_REDUCTIONS: i32 = 2;

const FUTILE_MOVE_REDUCTIONS: i32 = 2;
const LOSING_MOVE_REDUCTIONS: i32 = 2;

const TIMEEXT_MULTIPLIER: i32 = 5;

impl Search for Engine {
    fn find_best_move(&mut self, min_depth: i32, is_strict_timelimit: bool) -> Move {
        let mut alpha = MIN_SCORE;
        let mut beta = MAX_SCORE;

        self.hh.clear();

        self.starttime = Instant::now();

        self.cancel_possible = false;
        self.node_count = 0;
        self.next_check_node_count = 10000;
        self.log_every_second = false;
        self.is_stopped = false;

        let mut current_best_move: Move = NO_MOVE;

        let mut already_extended_timelimit = false;

        let mut previous_best_move: Move = NO_MOVE;
        let mut previous_best_score: i32 = 0;

        let mut fluctuation_count: i32 = 0;
        let mut score_fluctuations: i32 = 0;

        let player_color = self.board.active_player();

        let mut scored_move_count = 0;

        self.movegen.enter_ply(player_color, NO_MOVE, NO_MOVE, NO_MOVE);

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in min(min_depth, 2)..(MAX_DEPTH as i32) {
            self.current_depth = depth;

            let mut best_score: i32 = MIN_SCORE;

            let previous_alpha = alpha;
            let previous_beta = beta;

            let iteration_start_time = Instant::now();

            let mut best_move: Move = NO_MOVE;

            let mut a = -beta; // Search principal variation node with full window

            let mut iteration_cancelled = false;

            let mut move_num = 0;

            while let Some(m) = self.movegen.next_legal_move(&self.hh, &mut self.board) {
                move_num += 1;

                if depth > 12 {
                    let now = Instant::now();

                    let total_duration = now.duration_since(self.starttime);
                    if total_duration.as_millis() >= 1000 {
                        self.log_every_second = true;
                        self.last_log_time = now;
                        let uci_move = UCIMove::from_encoded_move(&self.board, m).to_uci();
                        let base_stats = self.get_base_stats(total_duration);
                        println!(
                            "info depth {}{} currmove {} currmovenumber {}",
                            depth, base_stats, uci_move, move_num
                        );
                    }
                }

                let (previous_piece, move_state) = self.board.perform_move(m);

                let gives_check = self.board.is_in_check(-player_color);

                // Use principal variation search
                let mut result = self.rec_find_best_move(a, -alpha, -player_color, depth - 1, 1, false, gives_check);
                if result == CANCEL_SEARCH {
                    iteration_cancelled = true;
                } else {
                    // Repeat search if it falls outside the window
                    if -result > alpha && -result < beta {
                        result = self.rec_find_best_move(-beta, -alpha, -player_color, depth - 1, 1, false, gives_check);
                        if result == CANCEL_SEARCH {
                            iteration_cancelled = true;
                        }
                    }
                }

                self.board.undo_move(m, previous_piece, move_state);

                if iteration_cancelled {
                    if best_move != NO_MOVE && previous_best_move != NO_MOVE {
                        score_fluctuations =
                            score_fluctuations * 100 / self.board.options.get_timeext_score_fluctuation_reductions();
                        score_fluctuations += (best_score - previous_best_score).abs();

                        if best_score.abs() >= (BLACK_MATE_SCORE - MAX_DEPTH as i32) {
                            // Reset score fluctuation statistic, if a check mate is found
                            score_fluctuations = 0;
                        }
                    }

                    if !is_strict_timelimit
                        && !self.is_stopped
                        && !already_extended_timelimit
                        && should_extend_timelimit(
                            best_move,
                            best_score,
                            previous_best_move,
                            previous_best_score,
                            score_fluctuations,
                            fluctuation_count,
                            self.board.options.get_timeext_score_change_threshold(),
                            self.board.options.get_timeext_score_fluctuation_threshold(),
                        )
                    {
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
                    best_move = m.with_score(score);
                    alpha = max(alpha, best_score);

                    if self.log_every_second {
                        let pv = self.extract_pv(best_move, depth - 1);
                        println!(
                            "info depth {} score {} pv {}",
                            depth,
                            get_score_info(best_score),
                            pv
                        );

                        if self.is_search_stopped() {
                            self.is_stopped = true;
                            iteration_cancelled = true;
                        }
                    }
                }

                a = -(alpha + 1); // Search all other moves (after principal variation) with a zero window

                self.movegen.update_root_move(m.with_score(score));
                scored_move_count += 1;
            }

            let current_time = Instant::now();
            let iteration_duration = current_time.duration_since(iteration_start_time);
            let total_duration = current_time.duration_since(self.starttime);
            let remaining_time = self.timelimit_ms - total_duration.as_millis() as i32;

            if !iteration_cancelled {
                if previous_best_move != NO_MOVE {
                    score_fluctuations = score_fluctuations * 100 / self.board.options.get_timeext_score_fluctuation_reductions();
                    score_fluctuations += (best_score - previous_best_score).abs();
                    fluctuation_count += 1;
                }

                self.cancel_possible = depth >= min_depth;
                if self.cancel_possible
                    && (remaining_time <= (iteration_duration.as_millis() as i32 * 2))
                {
                    // Not enough time left for another iteration

                    if is_strict_timelimit
                        || already_extended_timelimit
                        || !should_extend_timelimit(
                            best_move,
                            best_score,
                            previous_best_move,
                            previous_best_score,
                            score_fluctuations,
                            fluctuation_count,
                            self.board.options.get_timeext_score_change_threshold(),
                            self.board.options.get_timeext_score_fluctuation_threshold(),
                        )
                    {
                        iteration_cancelled = true;
                    }
                }
            }

            if best_move == NO_MOVE {
                best_move = previous_best_move;
                best_score = previous_best_score;
            }

            println!(
                "info depth {} score {}{} pv {}",
                depth,
                get_score_info(best_score),
                self.get_base_stats(total_duration),
                self.extract_pv(best_move, depth - 1)
            );

            current_best_move = best_move.with_score(best_score);

            if iteration_cancelled || scored_move_count <= 1 {
                // stop searching, if iteration has been cancelled or there is no valid move or only a single valid move

                break;
            }

            if (BLACK_MATE_SCORE - best_score.abs()) <= depth / 3 {
                // stop searching, if a close mate has already been found
                break;
            }

            previous_best_move = best_move;
            previous_best_score = best_score;

            alpha = previous_alpha;
            beta = previous_beta;

            self.movegen.reset();
            self.movegen.resort();
        }

        self.movegen.leave_ply();

        if scored_move_count == 0 {
            return NO_MOVE;
        }

        current_best_move
    }

    // Recursively calls itself with alternating player colors to
    // find the best possible move in response to the current board position.
    fn rec_find_best_move(&mut self, mut alpha: i32, beta: i32, player_color: Color, mut depth: i32, ply: i32, null_move_performed: bool, is_in_check: bool) -> i32 {
        if self.node_count >= self.next_check_node_count {
            self.next_check_node_count = self.node_count + 20000;
            let current_time = Instant::now();
            let total_duration = current_time.duration_since(self.starttime);

            if self.cancel_possible {
                if total_duration.as_millis() as i32 >= self.timelimit_ms {
                    // Cancel search if the time limit has been reached
                    return CANCEL_SEARCH;
                } else if self.is_search_stopped() {
                    self.is_stopped = true;
                    return CANCEL_SEARCH;
                }
            }

            if depth > 3 && self.log_every_second && current_time.duration_since(self.last_log_time).as_millis() >= 1000 {
                self.last_log_time = current_time;
                let base_stats = self.get_base_stats(total_duration);
                println!("info depth {}{}", self.current_depth, base_stats);
            }
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        if self.board.is_engine_draw() {
            self.node_count += 1;
            return 0;
        }

        if is_in_check {
            // Extend search when in check
            depth = max(1, depth + 1);
        } else if depth == 1 && (self.board.get_score() * player_color as i32) < alpha - self.board.options.get_razor_margin() {
            // Directly jump to quiescence search, if current position score is below a certain threshold
            depth = 0;
        }

        // Quiescence search
        if depth <= 0 || ply >= MAX_DEPTH as i32 {
            return self.quiescence_search(player_color, alpha, beta, ply);
        }

        self.node_count += 1;

        // Check transposition table
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get_entry(hash);

        let mut m = get_untyped_move(tt_entry);

        if m != NO_MOVE {
            m = m.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()));

            let mut can_use_hash_score = get_depth(tt_entry) >= depth;
            // Validate hash move for additional protection against hash collisions
            if !is_likely_valid_move(&self.board, player_color, m) {
                m = NO_MOVE;
                can_use_hash_score = false;
            }

            if can_use_hash_score {
                let score = m.score();

                match get_score_type(tt_entry) {
                    ScoreType::Exact => {
                        return score;
                    }

                    ScoreType::UpperBound => {
                        if score <= alpha {
                            return score;
                        }
                    }

                    ScoreType::LowerBound => {
                        if score > alpha {
                            alpha = score;
                            if alpha >= beta {
                                return score;
                            }
                        }
                    }
                };
            }

        }

        // Reduce nodes without hash move from transposition table
        if !is_pv && m == NO_MOVE && depth > 7 {
            depth -= 1;
        }

        let mut fail_high = false;

        // Null move reductions
        let original_depth = depth;
        if !is_pv && !null_move_performed && depth > 3 && !is_in_check {
            let r = if depth > 6 { 4 } else { 3 };
            self.board.perform_null_move();
            let result = self.rec_find_best_move(-beta, -beta + 1, -player_color, depth - r - 1, ply + 1, true, false);
            self.board.undo_null_move();
            if result == CANCEL_SEARCH {
                return CANCEL_SEARCH;
            }
            if -result >= beta {
                depth -= 4;
                fail_high = true;

                if depth <= 0 {
                    return self.quiescence_search(player_color, alpha, beta, ply);
                }
            }
        }

        let primary_killer = self.hh.get_primary_killer(ply);
        let secondary_killer = self.hh.get_secondary_killer(ply);

        let mut best_move = NO_MOVE;
        let mut best_score = MIN_SCORE;
        let mut score_type = ScoreType::UpperBound;
        let mut evaluated_move_count = 0;
        let mut has_valid_moves = false;

        let allow_reductions = depth > 2 && !is_in_check && !m.is_promotion();

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        let mut prune_low_score = 0;
        if !is_pv && depth <= 4 {
            prune_low_score = self.board.get_score() * player_color as i32 + depth * self.board.options.get_futility_margin_multiplier();
            allow_futile_move_pruning = prune_low_score <= alpha;
        }

        let opponent_pieces = self.board.get_all_piece_bitboard(-player_color);

        self.movegen.enter_ply(player_color, m, primary_killer, secondary_killer);

        loop {
            m = match self.movegen.next_move(&self.hh, &mut self.board) {
                Some(next_move) => next_move,
                None => {
                    if fail_high && has_valid_moves {
                        // research required, because a fail-high was reported by null search, but no cutoff was found during reduced search
                        depth = original_depth;
                        fail_high = false;
                        evaluated_move_count = 0;

                        self.movegen.reset();
                        continue;
                    } else {
                        // Last move has been evaluated
                        break;
                    }
                }
            };

            let start = m.start();
            let end = m.end();

            let (previous_piece, removed_piece_id) = self.board.perform_move(m);

            let mut skip = self.board.is_in_check(player_color); // skip if move would put own king in check

            let mut reductions = 0;
            let mut gives_check = false;

            if !skip {
                let target_piece_id = m.piece_id();
                has_valid_moves = true;
                gives_check = self.board.is_in_check(-player_color);
                if removed_piece_id == EMPTY {
                    let own_moves_left = depth / 2;
                    if allow_reductions
                        && !gives_check
                        && evaluated_move_count > LMR_THRESHOLD
                        && !self.board.is_pawn_move_close_to_promotion(previous_piece, end, opponent_pieces) {
                        // Reduce search depth for late moves (i.e. after trying the most promising moves)
                        reductions = LMR_REDUCTIONS;
                        if self.hh.has_negative_history(player_color, depth, start, end) {
                            // Reduce more, if move has negative history or SEE score
                            reductions += 1;
                        }
                    } else if allow_futile_move_pruning && !gives_check && !m.is_promotion() {
                        if !is_in_check && (own_moves_left <= 1 || (self.hh.has_negative_history(player_color, depth, start, end) && self.board.see_score(-player_color, start, end, target_piece_id, EMPTY) < 0)) {
                            // Prune futile move
                            skip = true;
                            if prune_low_score > best_score {
                                best_score = prune_low_score; // remember score with added margin for cases when all moves are pruned
                            }
                        } else {
                            // Reduce futile move
                            reductions = FUTILE_MOVE_REDUCTIONS;
                        }
                    } else if self.hh.has_negative_history(player_color, depth, start, end) || self.board.see_score(-player_color, start, end, target_piece_id, EMPTY) < 0 {
                        // Reduce search depth for moves with negative history or negative SEE score
                        reductions = LOSING_MOVE_REDUCTIONS;
                    }
                } else if removed_piece_id < previous_piece.abs() as i8 && self.board.see_score(-player_color, start, end, target_piece_id, removed_piece_id) < 0 {
                    // Reduce search depth for capture moves with negative SEE score
                    reductions = 1;
                }
            }

            if skip {
                self.board.undo_move(m, previous_piece, removed_piece_id);
            } else {
                if removed_piece_id == EMPTY {
                    self.hh.update_played_moves(depth, player_color, start, end);
                }

                evaluated_move_count += 1;

                let a = if evaluated_move_count > 1 {
                    -(alpha + 1)
                } else {
                    -beta
                };
                let mut result = self.rec_find_best_move(a, -alpha, -player_color, depth - reductions - 1, ply + 1, false, gives_check);
                if result == CANCEL_SEARCH {
                    self.board.undo_move(m, previous_piece, removed_piece_id);
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                if -result > alpha && (-result < beta || reductions > 0) {
                    // Repeat search without reduction and with full window
                    result = self.rec_find_best_move(-beta, -alpha, -player_color, depth - 1, ply + 1, false, gives_check);
                    if result == CANCEL_SEARCH {
                        self.board.undo_move(m, previous_piece, removed_piece_id);
                        self.movegen.leave_ply();
                        return CANCEL_SEARCH;
                    }
                }

                let score = -result;
                self.board.undo_move(m, previous_piece, removed_piece_id);

                if score > best_score {
                    best_score = score;
                    best_move = m;

                    // Alpha-beta pruning
                    if best_score > alpha {
                        alpha = best_score;
                        score_type = ScoreType::Exact;
                    }

                    if alpha >= beta {
                        self.tt.write_entry(hash, depth, best_move.with_score(best_score), ScoreType::LowerBound);

                        if removed_piece_id == EMPTY {
                            self.hh.update(depth, ply, player_color, start, end, best_move);
                        }

                        self.movegen.leave_ply();
                        return alpha;
                    }
                }
            }
        }

        self.movegen.leave_ply();

        if !has_valid_moves {
            if is_in_check {
                // Check mate
                return WHITE_MATE_SCORE + ply;
            }

            // Stale mate
            return 0;
        }

        self.tt.write_entry(hash, depth, best_move.with_score(best_score), score_type);

        best_score
    }

    fn quiescence_search(&mut self, active_player: Color, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.node_count += 1;

        if self.board.is_engine_draw() {
            return 0;
        }

        let position_score = self.board.get_score() * active_player as i32;
        if ply >= MAX_DEPTH as i32 {
            return position_score;
        }

        if position_score >= beta {
            return beta;
        }

        // Prune nodes where the position score is already so far below alpha that it is very unlikely to be raised by any available move
        let prune_low_captures = position_score < alpha - self.board.options.get_qs_prune_margin();

        if alpha < position_score {
            alpha = position_score;
        }

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut threshold = alpha - position_score - self.board.options.get_qs_see_threshold();

        while let Some(m) = self.movegen.next_capture_move(&mut self.board) {
            let target_piece_id = m.piece_id();
            let start = m.start();
            let end = m.end();
            let previous_piece = self.board.get_item(start);
            let previous_piece_id = previous_piece.abs();

            let captured_piece_id = self.board.get_item(end).abs();

            if prune_low_captures && target_piece_id as i8 == previous_piece_id && captured_piece_id < R {
                continue;
            }


            // skip capture moves with a SEE score below the given threshold
            if self.board.see_score(-active_player, start, end, previous_piece_id, captured_piece_id) <= threshold {
                continue;
            }

            let (previous_piece, move_state) = self.board.perform_move(m);

            if self.board.is_in_check(active_player) {
                // Invalid move
                self.board.undo_move(m, previous_piece, move_state);
                continue;
            }

            let score = -self.quiescence_search(-active_player, -beta, -alpha, ply + 1);
            self.board.undo_move(m, previous_piece, move_state);

            if score >= beta {
                self.movegen.leave_ply();
                return beta;
            }

            if score > alpha {
                alpha = score;
                threshold = alpha - position_score - self.board.options.get_qs_see_threshold();
            }
        }

        self.movegen.leave_ply();
        alpha
    }

    // Updates the current position until it is quiet
    fn make_quiet_position(&mut self) -> bool {
        let _ = self.static_quiescence_search(MIN_SCORE, MAX_SCORE, 0) as i16;

        loop {
            let entry = self.tt.get_entry(self.board.get_hash());
            if entry == 0 {
                return true;
            }

            let mut m = get_untyped_move(entry);
            if m == NO_MOVE || !is_likely_valid_move(&self.board, self.board.active_player(), m) {
                return true;
            }

            m = m.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()));

            self.board.perform_move(m);
            if self.board.is_in_check(self.board.active_player()) {
                return false;
            }
        }
    }

    fn is_quiet_position(&mut self) -> bool {
        if self.board.is_in_check(WHITE) || self.board.is_in_check(BLACK) {
            return false;
        }

        self.movegen.enter_ply(self.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE);
        while let Some(m) = self.movegen.next_capture_move(&mut self.board)
        {
            let previous_piece = self.board.get_item(m.start());
            let previous_piece_id = previous_piece.abs();
            let captured_piece_id = self.board.get_item(m.end()).abs();

            // skip capture moves with a SEE score below the given threshold
            if self.board.see_score(-self.board.active_player(), m.start(), m.end(), previous_piece_id, captured_piece_id) > 0 {
                self.movegen.leave_ply();
                return false;
            }
        }

        self.movegen.leave_ply();
        true
    }

    fn is_quiet_pv(&mut self, m: Move, depth: i32) -> bool {
        if self.board.is_in_check(WHITE) || self.board.is_in_check(BLACK) {
            return false;
        }

        if depth == 0 {
            return true;
        }

        let (previous_piece, move_state) = self.board.perform_move(m);

        if move_state != EMPTY {
            self.board.undo_move(m, previous_piece, move_state);
            return false;
        }

        let entry = self.tt.get_entry(self.board.get_hash());
        let mut next_move = if entry != 0 {
            get_untyped_move(entry)
        } else {
            NO_MOVE
        };

        if next_move == NO_MOVE {
            self.board.undo_move(m, previous_piece, move_state);
            return false;
        }

        next_move = next_move.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()));

        let active_player = self.board.active_player();
        let is_valid_followup_move = next_move != NO_MOVE && is_likely_valid_move(&self.board, active_player, next_move);
        let is_quiet = if is_valid_followup_move {
            self.is_quiet_pv(next_move, depth - 1)
        } else {
            false
        };

        self.board.undo_move(m, previous_piece, move_state);

        is_quiet
    }

    fn static_quiescence_search(&mut self, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        let active_player = self.board.active_player();

        let position_score = self.board.get_static_score() as i32 * active_player as i32;

        if ply >= 60 as i32 {
            return position_score;
        }

        if position_score >= beta {
            return beta;
        }

        if alpha < position_score {
            alpha = position_score;
        }

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut best_move = NO_MOVE;

        while let Some(m) = self.movegen.next_capture_move(&mut self.board) {
            let start = m.start();
            let end = m.end();
            let previous_piece = self.board.get_item(start);
            let previous_piece_id = previous_piece.abs();
            let captured_piece_id = self.board.get_item(end).abs();

            // skip capture moves with a SEE score below the given threshold
            if captured_piece_id != EMPTY && self.board.see_score(-active_player, start, end, previous_piece_id, captured_piece_id) < 0 {
                continue;
            }

            let (_, move_state) = self.board.perform_move(m);

            if self.board.is_in_check(active_player) {
                // Invalid move
                self.board.undo_move(m, previous_piece, move_state);
                continue;
            }

            let score = -self.static_quiescence_search(-beta, -alpha, ply + 1);
            self.board.undo_move(m, previous_piece, move_state);

            if score >= beta {
                self.tt.write_entry(self.board.get_hash(), 60 - ply, m, ScoreType::LowerBound);
                self.movegen.leave_ply();
                return beta;
            }

            if score > alpha {
                best_move = m;
                alpha = score;
            }
        }

        if best_move != NO_MOVE {
            self.tt.write_entry(self.board.get_hash(), 60 - ply, best_move, ScoreType::Exact);
        }

        self.movegen.leave_ply();
        alpha
    }

    fn get_base_stats(&self, duration: Duration) -> String {
        let duration_micros = duration.as_micros();
        let nodes_per_second = if duration_micros > 0 {
            self.node_count as u128 * 1_000_000 / duration_micros
        } else {
            0
        };

        if nodes_per_second > 0 {
            format!(
                " nodes {} nps {} time {}",
                self.node_count,
                nodes_per_second,
                duration_micros / 1000
            )
        } else {
            format!(" nodes {} time {}", self.node_count, duration_micros / 1000)
        }
    }

    fn extract_pv(&mut self, m: Move, depth: i32) -> String {
        let uci_move = UCIMove::from_encoded_move(&self.board, m).to_uci();
        if depth == 0 {
            return uci_move;
        }

        let (previous_piece, move_state) = self.board.perform_move(m);

        let entry = self.tt.get_entry(self.board.get_hash());
        let next_move = if entry != 0 {
            let m = get_untyped_move(entry);
            m.with_typ(self.board.get_move_type(m.start(), m.end(), m.piece_id()))
        } else {
            NO_MOVE
        };

        let active_player = self.board.active_player();
        let is_valid_followup_move = next_move != NO_MOVE && is_likely_valid_move(&self.board, active_player, next_move);
        let followup_uci_moves = if is_valid_followup_move {
            format!(" {}", self.extract_pv(next_move, depth - 1))
        } else {
            String::from("")
        };

        self.board.undo_move(m, previous_piece, move_state);

        format!("{}{}", uci_move, followup_uci_moves)
    }
}

fn should_extend_timelimit(
    new_move: Move,
    new_score: i32,
    previous_move: Move,
    previous_score: i32,
    score_fluctuations: i32,
    fluctuation_count: i32,
    score_change_threshold: i32,
    score_fluctuation_threshold: i32
) -> bool {
    if previous_move == NO_MOVE || new_move == NO_MOVE {
        return false;
    }

    let avg_fluctuations = if fluctuation_count > 0 {
        score_fluctuations / fluctuation_count
    } else {
        0
    };

    !new_move.is_same_move(previous_move)
        || (new_score - previous_score).abs() >= score_change_threshold
        || avg_fluctuations >= score_fluctuation_threshold
}

fn get_score_info(score: i32) -> String {
    if score <= WHITE_MATE_SCORE + MAX_DEPTH as i32 {
        return format!("mate {}", (WHITE_MATE_SCORE - score - 1) / 2);
    } else if score >= BLACK_MATE_SCORE - MAX_DEPTH as i32 {
        return format!("mate {}", (BLACK_MATE_SCORE - score + 1) / 2);
    }

    format!("cp {}", score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::engine::Message;
    use crate::fen::write_fen;
    use crate::pieces::{K, R};
    use std::sync::mpsc;
    use crate::moves::NO_MOVE;

    #[test]
    fn finds_mate_in_one() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  K,  0,  0,  0,
           -R,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0, -R,
        ];

        let fen = to_fen(BLACK, &items);
        let (_, rx) = mpsc::channel::<Message>();
        let mut engine = Engine::new_from_fen(rx, &fen, 1);

        engine.timelimit_ms = 0;

        let m = engine.find_best_move(2, true);
        assert_ne!(NO_MOVE, m);

        engine.perform_move(m);

        let is_check_mate = engine.find_best_move(1, true) == NO_MOVE && engine.board.is_in_check(WHITE);
        assert!(is_check_mate);
    }

    #[test]
    fn finds_mate_in_two() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0, -K,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            R,  0,  0,  0,  K,  0,  0,  R,
        ];

        let fen = to_fen(WHITE, &items);
        let (_, rx) = mpsc::channel::<Message>();
        let mut engine = Engine::new_from_fen(rx, &fen, 1);

        engine.timelimit_ms = 0;

        let m1 = engine.find_best_move(3, true);
        engine.perform_move(m1);
        let m2 = engine.find_best_move(2, true);
        engine.perform_move(m2);
        let m3 = engine.find_best_move(1, true);
        engine.perform_move(m3);

        let is_check_mate = engine.find_best_move(1, true) == NO_MOVE && engine.board.is_in_check(BLACK);
        assert!(is_check_mate);
    }

    fn to_fen(active_player: Color, items: &[i8; 64]) -> String {
        write_fen(&Board::new(items, active_player, 0, None, 0, 1))
    }
}
