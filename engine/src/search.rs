/*
 * Velvet Chess Engine
 * Copyright (C) 2021 mhonert (https://github.com/mhonert)
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

use crate::colors::{Color, WHITE, BLACK};
use crate::engine::{LogLevel, Message};
use crate::pieces::{EMPTY, R, P};
use crate::scores::{MATE_SCORE, MIN_SCORE, MATED_SCORE, MAX_SCORE};
use crate::transposition_table::{get_depth, get_score_type, get_untyped_move, MAX_DEPTH, ScoreType, to_root_relative_score, from_root_relative_score, TranspositionTable};
use crate::uci_move::UCIMove;
use std::cmp::{max, min};
use std::time::{Duration, Instant};
use crate::moves::{Move, NO_MOVE};
use crate::move_gen::{NEGATIVE_HISTORY_SCORE, is_killer, MoveGenerator};
use LogLevel::Info;
use crate::time_management::{MAX_TIMELIMIT_MS, TIMEEXT_MULTIPLIER, TimeManager};
use crate::board::Board;
use std::sync::Arc;
use crate::history_heuristics::HistoryHeuristics;
use std::sync::atomic::{AtomicBool, Ordering, AtomicU64};
use std::sync::mpsc::{TryRecvError, Receiver, channel, Sender};
use std::thread;

pub const DEFAULT_SEARCH_THREADS: usize = 1;
pub const MAX_SEARCH_THREADS: usize = 256;

const CANCEL_SEARCH: i32 = i32::MAX - 1;

const LMR_THRESHOLD: i32 = 3;

const FUTILE_MOVE_REDUCTIONS: i32 = 2;
const LOSING_MOVE_REDUCTIONS: i32 = 2;

const QS_SEE_THRESHOLD: i32 = 104;
const QS_PRUNE_MARGIN: i32 = 650;

const TIME_SAFETY_MARGIN_MS: i32 = 20;

const INITIAL_ASPIRATION_WINDOW_SIZE: i32 = 16;
const INITIAL_ASPIRATION_WINDOW_STEP: i32 = 16;

#[derive(Clone)]
pub struct Search {
    pub board: Board,
    pub movegen: MoveGenerator,
    pub hh: HistoryHeuristics,
    pub tt: Arc<TranspositionTable>,

    log_level: LogLevel,
    limits: SearchLimits,
    time_mgr: TimeManager,

    cancel_possible: bool,
    last_log_time: Instant,
    next_check_node_count: u64,
    next_hh_age_node_count: u64,
    current_depth: i32,
    max_reached_depth: i32,

    local_node_count: u64,

    node_count: Arc<AtomicU64>,
    is_stopped: Arc<AtomicBool>,

    total_thread_count: usize,

    pondering: bool,
}

impl Search {
    pub fn new(is_stopped: Arc<AtomicBool>, node_count: Arc<AtomicU64>, log_level: LogLevel, limits: SearchLimits, tt: Arc<TranspositionTable>, board: Board, search_thread_count: usize, ponder: bool) -> Self {
        let hh = HistoryHeuristics::new();

        let time_mgr = TimeManager::new();

        Search {
            log_level,
            limits,
            tt,
            board,
            hh,
            time_mgr,
            movegen: MoveGenerator::new(),
            cancel_possible: false,
            local_node_count: 0,
            node_count,
            last_log_time: Instant::now(),
            next_check_node_count: 0,
            next_hh_age_node_count: 0,
            current_depth: 0,
            max_reached_depth: 0,
            is_stopped,

            total_thread_count: search_thread_count,
            pondering: ponder
        }
    }

    pub fn find_best_move(&mut self, rx: Option<&Receiver<Message>>, min_depth: i32, skipped_moves: &[Move]) -> (Move, PrincipalVariation) {
        self.time_mgr.reset(self.limits.time_limit_ms, self.limits.strict_time_limit);

        self.cancel_possible = false;
        self.node_count.store(0, Ordering::Relaxed);
        self.local_node_count = 0;

        self.next_check_node_count = min(self.limits.node_limit, 1000);
        self.next_hh_age_node_count = 1000000;

        let mut last_best_move: Move = NO_MOVE;

        self.hh.clear();

        let active_player = self.board.active_player();

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

        self.set_stopped(false);

        let helper_threads = self.start_helper_threads(skipped_moves);

        let mut pv = PrincipalVariation::default();

        helper_threads.start_search();

        let hash = self.board.get_hash();
        let tt_entry = self.tt.get_entry(hash);

        let mut score = self.board.eval();
        if tt_entry != 0 {
            let hash_move = self.movegen.sanitize_move(&self.board, active_player, get_untyped_move(tt_entry));

            if hash_move != NO_MOVE {
                score = to_root_relative_score(0, hash_move.score());
            }
        }

        let mut window_size = INITIAL_ASPIRATION_WINDOW_SIZE;
        let mut window_step = INITIAL_ASPIRATION_WINDOW_STEP;

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in 1..=self.limits.depth_limit {
            let iteration_start_time = Instant::now();
            self.current_depth = depth;
            self.max_reached_depth = 0;

            let mut iteration_cancelled = false;

            let mut local_pv = PrincipalVariation::default();
            let (move_num, mut best_move, current_pv, new_window_step) = self.root_search(rx, skipped_moves, window_step, window_size, score, depth, &mut local_pv);
            if new_window_step > window_step {
                window_step = new_window_step;
                window_size = new_window_step;
            } else if window_step > 16 {
                window_step /= 2;
                window_size /= 2;
            }

            let now = Instant::now();

            if best_move == NO_MOVE {
                iteration_cancelled = true;
            }

            if !iteration_cancelled {
                pv = local_pv.clone();
                self.cancel_possible = depth >= min_depth;
                let iteration_duration = now.duration_since(iteration_start_time);
                if !self.pondering && self.cancel_possible && !self.time_mgr.is_time_for_another_iteration(now, iteration_duration) {
                    if self.time_mgr.should_extend_timelimit() {
                        self.time_mgr.extend_timelimit();
                    } else {
                        iteration_cancelled = true;
                    }
                }
            }

            if best_move == NO_MOVE {
                best_move = last_best_move;
            } else if self.log(Info) {
                let seldepth = self.max_reached_depth;

                println!(
                    "info depth {} seldepth {} score {}{}{}",
                    depth,
                    seldepth,
                    get_score_info(best_move.score()),
                    self.get_base_stats(self.time_mgr.search_duration(now)),
                    current_pv.map(|pv| format!(" pv {}", pv)).unwrap_or_default()
                );
            }

            last_best_move = best_move;
            score = best_move.score();

            if iteration_cancelled || move_num <= 1 {
                // stop searching, if iteration has been cancelled or there is no valid move or only a single valid move
                break;
            }
        }

        helper_threads.stop_search();

        self.movegen.leave_ply();

        if let Some(r) = rx {
            while self.pondering && !self.is_stopped() {
                self.check_messages(r, true);
            }
        }

        self.set_stopped(true);

        (last_best_move, pv)
    }

    fn start_helper_threads(&mut self, skipped_moves: &[Move]) -> HelperThreads {
        let mut helper_threads = HelperThreads::new();

        for _ in 0..(self.total_thread_count - 1) {

            let node_count = self.node_count.clone();
            let tt = self.tt.clone();
            let board = self.board.clone();

            let search_stopped = Arc::new(AtomicBool::new(true));
            let stop_search = Arc::new(AtomicBool::new(false));
            let thread_terminated = Arc::new(AtomicBool::new(false));

            search_stopped.store(false, Ordering::Release);

            let (tx, rx) = channel::<HelperThreadMessage>();
            helper_threads.add(tx, stop_search.clone(), search_stopped.clone(), thread_terminated.clone());

            let skipped_moves = Vec::from(skipped_moves);
            thread::spawn(move || {
                let limits = SearchLimits::default();
                let sub_search = Search::new(stop_search.clone(), node_count, LogLevel::Error, limits, tt, board, 1, false);

                HelperThread::run(search_stopped, rx, &skipped_moves, sub_search);
                thread_terminated.store(true, Ordering::Release);
            });
        }

        helper_threads
    }

    fn root_search(&mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], window_step: i32, window_size: i32, score: i32, depth: i32, pv: &mut PrincipalVariation) -> (i32, Move, Option<String>, i32) {
        let mut alpha = if depth > 7 { score - window_size } else { MIN_SCORE };
        let mut beta = if depth > 7 { score + window_size } else { MAX_SCORE };

        let mut step = window_step;
        loop {
            pv.clear();

            let (move_num, best_move, current_pv) = self.bounded_root_search(rx, skipped_moves, alpha, beta, depth, pv);
            if best_move == NO_MOVE {
                return (move_num, best_move, current_pv, step);
            }

            let best_score = best_move.score();
            if best_score <= alpha {
                alpha = max(MIN_SCORE, alpha - step);
            } else if best_score >= beta {
                beta = min(MAX_SCORE, beta + step);
            } else {
                return (move_num, best_move, current_pv, step);
            }

            step *= 2;
        }
    }

    // Root search within the bounds of an aspiration window (alpha...beta)
    fn bounded_root_search(&mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], mut alpha: i32, beta: i32, depth: i32, pv: &mut PrincipalVariation) -> (i32, Move, Option<String>) {
        let mut move_num = 0;
        let mut a = -beta;
        let mut best_move: Move = NO_MOVE;
        let mut best_score = MIN_SCORE;

        let mut iteration_cancelled = false;
        let mut current_pv = None;

        let active_player = self.board.active_player();

        let mut reduction = 0;
        let mut tree_scale = 0;

        let is_multi_threaded = self.total_thread_count > 1;

        self.movegen.reset_root_moves();
        while let Some(m) = self.movegen.next_root_move(&self.hh, &mut self.board) {
            if skipped_moves.contains(&m.without_score()) {
                continue;
            }
            move_num += 1;

            if self.log(Info) && self.local_node_count > 2000000 {
                let now = Instant::now();
                if self.time_mgr.search_duration_ms(now) >= 1000 {
                    self.last_log_time = now;
                    let uci_move = UCIMove::from_encoded_move(&self.board, m).to_uci();
                    println!("info depth {} currmove {} currmovenumber {}", depth, uci_move, move_num);
                }
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(m);

            let capture_pos = if removed_piece_id != EMPTY { m.end() } else { -1 };

            let gives_check = self.board.is_in_check(-active_player);

            let mut local_pv = PrincipalVariation::default();

            let mut tree_size = self.local_node_count;
            // Use principal variation search
            let mut result = self.rec_find_best_move(rx, a, -alpha, depth - reduction - 1, 1, gives_check, capture_pos, &mut local_pv, m);
            if result == CANCEL_SEARCH {
                iteration_cancelled = true;
            } else if -result > alpha && a != -beta {
                // Repeat search if it falls outside the search window
                result = self.rec_find_best_move(rx, -beta, -alpha, depth - 1, 1, gives_check, capture_pos, &mut local_pv, m);
                if result == CANCEL_SEARCH {
                    iteration_cancelled = true;
                }
            }

            self.board.undo_move(m, previous_piece, removed_piece_id);

            if iteration_cancelled {
                break;
            }

            let score = -result;
            if score > best_score {
                best_score = score;
                best_move = m.with_score(score);

                if best_score <= alpha || best_score >= beta {
                    return (move_num, best_move, current_pv);
                }

                pv.update(best_move, &mut local_pv);
                current_pv = Some(self.pv_info(&pv.moves()));

                if best_score > alpha {
                    alpha = score;

                    self.time_mgr.update_best_move(best_move, depth);

                    if depth >= 7 && reduction == 0 {
                        reduction = 1;
                    } else {
                        reduction = 0;
                    }
                    a = -(alpha + 1);
                }
            }

            tree_size = (self.local_node_count - tree_size) << reduction;
            if move_num == 1 {
                tree_scale = max(13, 64 - tree_size.leading_zeros()) - 13;
            }
            self.movegen.update_root_move(m.with_score(min(MAX_SCORE, (tree_size >> tree_scale) as i32)));

        }

        self.movegen.reorder_root_moves(best_move, is_multi_threaded);

        (move_num, best_move, current_pv)
    }

    fn node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }

    fn inc_node_count(&self) {
        self.node_count.fetch_add(1, Ordering::Relaxed);
    }

    // Recursively calls itself with alternating player colors to
    // find the best possible move in response to the current board position.
    fn rec_find_best_move(&mut self, rx: Option<&Receiver<Message>>, mut alpha: i32, beta: i32, mut depth: i32, ply: i32, is_in_check: bool,
                          capture_pos: i32, pv: &mut PrincipalVariation, opponent_move: Move) -> i32 {

        self.max_reached_depth = max(ply, self.max_reached_depth);

        if let Some(rx) = rx {
            self.check_search_limits(rx)
        }

        if self.is_stopped() {
            return CANCEL_SEARCH;
        }

        if self.board.is_draw() {
            return 0;
        }

        if self.local_node_count >= self.next_hh_age_node_count {
            self.hh.age_entries();
            self.next_hh_age_node_count = self.local_node_count + 2000000;
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        // Prune, if even the best possible score cannot improve alpha (because a shorter mate has already been found)
        let best_possible_score = MATE_SCORE - ply - 1;
        if best_possible_score <= alpha {
            return best_possible_score;
        }

        // Prune, if worst possible score is already sufficient to reach beta
        let worst_possible_score = MATED_SCORE + ply + if is_in_check { 0 } else { 1 };
        if worst_possible_score >= beta {
            return worst_possible_score;
        }

        let mut pos_score: Option<i32> = None;

        let active_player = self.board.active_player();
        if is_in_check {
            // Extend search when in check
            depth = max(1, depth + 1);

        } else if !is_pv && depth > 0 && depth <= 3 && self.board.fast_eval() * active_player as i32 - (100 * depth) >= beta  {
            // Prune, if position is already so good, that it is unlikely for the opponent to counter it within the remaining search depth
            pos_score = pos_score.or_else(|| Some(self.board.eval() * active_player as i32));

            if pos_score.unwrap().abs() < MATE_SCORE - (2 * MAX_DEPTH as i32) && pos_score.unwrap() - (100 * depth) >= beta {
                return self.quiescence_search(active_player, alpha, beta, ply, pos_score, pv);
            }
        }

        // Quiescence search
        if depth <= 0 || ply >= (MAX_DEPTH - 16) as i32 {
            return self.quiescence_search(active_player, alpha, beta, ply, pos_score, pv);
        }

        self.local_node_count += 1;
        self.inc_node_count();

        // Check transposition table
        let hash = self.board.get_hash();
        let tt_entry = self.tt.get_entry(hash);

        let mut hash_move = NO_MOVE;

        if tt_entry != 0 {
            hash_move = self.movegen.sanitize_move(&self.board, active_player, get_untyped_move(tt_entry));

            if hash_move != NO_MOVE {
                let score = to_root_relative_score(ply, hash_move.score());
                match get_score_type(tt_entry) {
                    ScoreType::Exact => {
                        if get_depth(tt_entry) >= depth {
                            return score
                        }
                        pos_score = Some(score);
                    },

                    ScoreType::UpperBound => {
                        if score <= alpha && get_depth(tt_entry) >= depth {
                            return score;
                        }
                    },

                    ScoreType::LowerBound => {
                        if get_depth(tt_entry) >= depth {
                            alpha = max(alpha, score);
                            if alpha >= beta {
                                return score;
                            }
                        }
                        pos_score = Some(score);
                    }
                };
            }
        } else if depth > 7 {
            // Reduce nodes without hash move from transposition table
            depth -= 1;
        }

        // Null move pruning
        if !is_pv && depth > 3 && !is_in_check && !self.board.is_pawn_endgame() {
            pos_score = pos_score.or_else(|| Some(self.board.eval() * active_player as i32));
            if pos_score.unwrap() >= beta {
                let r = log2((depth * 3 - 3) as u32);
                self.board.perform_null_move();
                let result = self.rec_find_best_move(rx, -beta, -beta + 1, depth - r - 1, ply + 1, false, -1, &mut PrincipalVariation::default(), NO_MOVE);
                self.board.undo_null_move();
                if result == CANCEL_SEARCH {
                    return CANCEL_SEARCH;
                }
                if -result >= beta {
                    return if result.abs() < MATE_SCORE - 2 * MAX_DEPTH as i32 {
                        -result
                    } else {
                        beta
                    }
                }
            }
        }

        let (primary_killer, secondary_killer) = self.hh.get_killer_moves(ply);
        let counter_move = self.hh.get_counter_move(opponent_move);

        let mut best_score = worst_possible_score;
        let mut best_move = NO_MOVE;

        let mut score_type = ScoreType::UpperBound;
        let mut evaluated_move_count = 0;
        let mut has_valid_moves = false;

        let allow_reductions = depth > 2 && !is_in_check;

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        if !is_pv && depth <= 6 {
            let margin = (6 << depth) * 4 + 16;
            if self.board.fast_eval() * active_player as i32 + margin <= alpha {
                let prune_low_score = pos_score.unwrap_or_else(|| self.board.eval() * active_player as i32);
                allow_futile_move_pruning = prune_low_score.abs() < MATE_SCORE - 2 * MAX_DEPTH as i32 && prune_low_score + margin <= alpha;
            }
        }

        let opponent_pieces = self.board.get_all_piece_bitboard(-active_player);

        self.movegen.enter_ply(active_player, hash_move, primary_killer, secondary_killer, counter_move);

        let mut a = -beta;

        let occupied_bb = self.board.get_occupancy_bitboard();

        let previous_move_was_capture = capture_pos != -1;

        let hh_counter_scale = self.hh.calc_counter_scale(depth);

        while let Some(curr_move) = self.movegen.next_move(&self.hh, &mut self.board) {
            let start = curr_move.start();
            let end = curr_move.end();

            let (previous_piece, removed_piece_id) = self.board.perform_move(curr_move);

            let mut skip = self.board.is_in_check(active_player); // skip if move would put own king in check

            let mut reductions = 0;
            let mut gives_check = false;

            if !skip {
                let target_piece_id = curr_move.piece_id();
                has_valid_moves = true;
                gives_check = self.board.is_in_check(-active_player);

                if !is_pv && previous_move_was_capture && evaluated_move_count > 0 && capture_pos != curr_move.end() {
                    reductions = 1;
                }

                if removed_piece_id == EMPTY {
                    if allow_reductions
                        && !gives_check
                        && evaluated_move_count > LMR_THRESHOLD
                        && !self.board.is_pawn_move_close_to_promotion(previous_piece, end, opponent_pieces) {
                        reductions += if curr_move.score() == NEGATIVE_HISTORY_SCORE { 3 } else { 2 };
                    } else if allow_futile_move_pruning && !gives_check && !curr_move.is_queen_promotion() {
                        // Reduce futile move
                        reductions += FUTILE_MOVE_REDUCTIONS;
                    } else if !is_pv && (curr_move.score() == NEGATIVE_HISTORY_SCORE || self.board.has_negative_see(-active_player, start, end, target_piece_id, EMPTY, 0, occupied_bb)) {
                        // Reduce search depth for moves with negative history or negative SEE score
                        reductions += LOSING_MOVE_REDUCTIONS;
                    }

                    if !is_pv && allow_futile_move_pruning && evaluated_move_count > 0 && !is_in_check && !gives_check && reductions >= (depth - 1) {
                        // Prune futile move
                        skip = true;
                    } else if reductions > 0 && is_killer(curr_move) {
                        // Reduce killer moves less
                        reductions -= 1;
                    }
                } else if removed_piece_id < previous_piece.abs() as i8 {
                    skip = self.movegen.skip_bad_capture(curr_move, removed_piece_id, occupied_bb, &mut self.board)
                }
            }

            if skip {
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
            } else {
                let new_capture_pos = if removed_piece_id != EMPTY { end } else { - 1 };

                evaluated_move_count += 1;

                let mut local_pv = PrincipalVariation::default();

                let mut result = self.rec_find_best_move(rx, a, -alpha, depth - reductions - 1, ply + 1, gives_check, new_capture_pos, &mut local_pv, curr_move);
                if result == CANCEL_SEARCH {
                    self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                if -result > alpha && (reductions > 0 || (-result < beta && a != -beta)) {
                    // Repeat search without reduction and with full window
                    result = self.rec_find_best_move(rx, -beta, -alpha, depth - 1, ply + 1, gives_check, new_capture_pos, &mut local_pv, curr_move);
                    if result == CANCEL_SEARCH {
                        self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                        self.movegen.leave_ply();
                        return CANCEL_SEARCH;
                    }
                }

                let score = -result;
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);

                if score > best_score {
                    best_score = score;
                    best_move = curr_move;

                    // Alpha-beta pruning
                    if best_score >= beta {
                        self.tt.write_entry(hash, depth, best_move.with_score(from_root_relative_score(ply, best_score)), ScoreType::LowerBound);

                        if removed_piece_id == EMPTY {
                            self.hh.update(ply, active_player, opponent_move, best_move, hh_counter_scale);
                        }

                        self.movegen.leave_ply();
                        return best_score;
                    }

                    if best_score > alpha {
                        alpha = best_score;
                        score_type = ScoreType::Exact;
                        pv.update(best_move, &mut local_pv);
                    }
                } else if removed_piece_id == EMPTY {
                    self.hh.update_played_moves(active_player, curr_move, hh_counter_scale);
                }

                a = -(alpha + 1);
            }
        }

        self.movegen.leave_ply();

        if !has_valid_moves {
            return if is_in_check {
                MATED_SCORE + ply // Check mate
            } else {
                0 // Stale mate
            }
        }

        self.tt.write_entry(hash, depth, best_move.with_score(from_root_relative_score(ply, best_score)), score_type);

        best_score
    }

    fn check_search_limits(&mut self, rx: &Receiver<Message>) {
        if self.local_node_count >= self.next_check_node_count {
            self.next_check_node_count = if self.limits.node_limit != u64::MAX { self.limits.node_limit } else { self.local_node_count + 1000 };

            self.check_messages(rx, false);

            let now = Instant::now();
            if !self.pondering && self.cancel_possible && (self.node_count() >= self.limits.node_limit || self.time_mgr.is_timelimit_exceeded(now)) {
                // Cancel search if the node or time limit has been reached, but first check
                // whether the search time should be extended
                if !self.is_stopped() && self.time_mgr.should_extend_timelimit() {
                    self.time_mgr.extend_timelimit();
                } else {
                    self.set_stopped(true);
                }
            }

            if self.log(Info) && now.duration_since(self.last_log_time).as_millis() >= 1000 {
                self.last_log_time = now;
                let base_stats = self.get_base_stats(self.time_mgr.search_duration(now));
                println!("info depth {} seldepth {}{}", self.current_depth, self.max_reached_depth, base_stats);
            }
        }
    }

    pub fn quiescence_search(&mut self, active_player: Color, mut alpha: i32, beta: i32, ply: i32, pos_score: Option<i32>, pv: &mut PrincipalVariation) -> i32 {
        if self.is_stopped() {
            return CANCEL_SEARCH;
        }

        self.local_node_count += 1;
        self.inc_node_count();

        self.max_reached_depth = max(ply, self.max_reached_depth);

        if self.board.is_insufficient_material_draw() {
            return 0;
        }

        let position_score = pos_score.unwrap_or_else(|| self.board.eval() * active_player as i32);
        if ply >= MAX_DEPTH as i32 {
            return position_score;
        }

        if position_score >= beta {
            return beta;
        }

        // Prune nodes where the position score is already so far below alpha that it is very unlikely to be raised by any available move
        let prune_low_captures = position_score < alpha - QS_PRUNE_MARGIN;

        if alpha < position_score {
            alpha = position_score;
        }

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut threshold = (alpha - position_score - QS_SEE_THRESHOLD) as i16;

        let occupied_bb = self.board.get_occupancy_bitboard();

        while let Some(m) = self.movegen.next_capture_move(&mut self.board) {

            let end = m.end();
            let captured_piece_id = self.board.get_item(end).abs();
            let previous_piece_id;
            if !m.is_queen_promotion() {
                if prune_low_captures && captured_piece_id < R {
                    continue;
                }

                previous_piece_id = m.piece_id();
            } else {
                previous_piece_id = P;
            }

            let start = m.start();

            // skip capture moves with a SEE score below the given threshold
            if self.board.has_negative_see(-active_player, start, end, previous_piece_id, captured_piece_id, threshold, occupied_bb) {
                continue;
            }

            let (previous_piece, move_state) = self.board.perform_move(m);

            if self.board.is_in_check(active_player) {
                // Invalid move
                self.board.undo_move(m, previous_piece, move_state);
                continue;
            }

            let mut local_pv = PrincipalVariation::default();

            let score = -self.quiescence_search(-active_player, -beta, -alpha, ply + 1, None, &mut local_pv);
            self.board.undo_move(m, previous_piece, move_state);

            if score >= beta {
                self.movegen.leave_ply();
                return score;
            }

            if score > alpha {
                alpha = score;
                threshold = (alpha - position_score - QS_SEE_THRESHOLD) as i16;
                pv.update(m, &mut local_pv);
            }
        }

        self.movegen.leave_ply();
        alpha
    }

    fn get_base_stats(&self, duration: Duration) -> String {
        let node_count = self.node_count();
        let duration_micros = duration.as_micros();
        let nodes_per_second = if duration_micros > 0 {
            node_count as u128 * 1_000_000 / duration_micros
        } else {
            0
        };

        if nodes_per_second > 0 {
            format!(
                " nodes {} nps {} hashfull {} time {}",
                node_count,
                nodes_per_second,
                self.tt.hash_full(),
                duration_micros / 1000
            )
        } else {
            format!(" nodes {} time {}", node_count, duration_micros / 1000)
        }
    }

    fn pv_info(&mut self, pv: &[Move]) -> String {
        if let Some((m, rest_pv)) = pv.split_first() {
            let uci_move = UCIMove::from_encoded_move(&self.board, *m).to_uci();
            let (previous_piece, move_state) = self.board.perform_move(*m);

            let followup_moves = self.pv_info(rest_pv);

            self.board.undo_move(*m, previous_piece, move_state);
            format!("{} {}", uci_move, followup_moves)

        } else {
            String::new()
        }
    }

    fn log(&self, log_level: LogLevel) -> bool {
        self.log_level <= log_level
    }

    fn is_stopped(&self) -> bool {
        self.is_stopped.load(Ordering::Acquire)
    }

    fn set_stopped(&mut self, value: bool) {
        self.is_stopped.store(value, Ordering::Release);
    }

    fn check_messages(&mut self, rx: &Receiver<Message>, blocking: bool) {
        if let Some(msg) = self.receive_message(rx, blocking) {
            match msg {
                Message::IsReady => println!("readyok"),

                Message::Stop => {
                    self.pondering = false;
                    self.set_stopped(true);
                }

                Message::PonderHit => {
                    self.pondering = false;
                }

                _ => ()
            }
        }
    }

    fn receive_message(&mut self, rx: &Receiver<Message>, blocking: bool) -> Option<Message> {
        if blocking {
            match rx.recv() {
                Ok(msg) => Some(msg),
                Err(e) => {
                    self.uci_channel_error(e.to_string());
                    None
                }
            }
        } else {
            match rx.try_recv() {
                Ok(msg) => Some(msg),
                Err(e) => if matches!(e, TryRecvError::Empty) {
                    None
                } else {
                    self.uci_channel_error(e.to_string());
                    None
                }
            }
        }
    }

    fn uci_channel_error(&mut self, err_msg: String) {
        eprintln!("info search thread could not read from UCI thread and will be stopped: {}", err_msg);
        self.set_stopped(true);
        self.pondering = false;
    }

    pub fn set_node_limit(&mut self, node_limit: u64) {
        self.limits.node_limit = node_limit;
    }
}


#[derive(Copy, Clone, Debug)]
pub struct SearchLimits {
    node_limit: u64,
    depth_limit: i32,
    time_limit_ms: i32,
    strict_time_limit: bool,

    wtime: i32,
    btime: i32,
    winc: i32,
    binc: i32,
    move_time: i32,
    moves_to_go: i32,
}

impl SearchLimits {
    pub fn default() -> Self {
        SearchLimits {
            node_limit: u64::MAX,
            depth_limit: MAX_DEPTH as i32,
            time_limit_ms: i32::MAX,
            strict_time_limit: true,

            wtime: -1,
            btime: -1,
            winc: 0,
            binc: 0,
            move_time: i32::MAX,
            moves_to_go: 1,
        }
    }

    pub fn nodes(node_limit: u64) -> SearchLimits {
        let mut limits = SearchLimits::default();
        limits.node_limit = node_limit;

        limits
    }

    pub fn new(depth_limit: Option<i32>, node_limit: Option<u64>, wtime: Option<i32>, btime: Option<i32>, winc: Option<i32>, binc: Option<i32>, move_time: Option<i32>, moves_to_go: Option<i32>) -> Result<Self, &'static str> {
        let depth_limit = depth_limit.unwrap_or(MAX_DEPTH as i32);
        if depth_limit <= 0 {
            return Err("depth limit must be > 0");
        }

        let node_limit = node_limit.unwrap_or(u64::MAX);

        Ok(SearchLimits {
            depth_limit,
            node_limit,
            time_limit_ms: i32::MAX,
            strict_time_limit: true,

            wtime: wtime.unwrap_or(-1),
            btime: btime.unwrap_or(-1),
            winc: winc.unwrap_or(0),
            binc: binc.unwrap_or(0),
            move_time: move_time.unwrap_or(-1),
            moves_to_go: moves_to_go.unwrap_or(40)
        })
    }

    pub fn update(&mut self, active_player: Color) {
        let (time_left, inc) = match active_player {
            WHITE => (self.wtime, self.winc),
            BLACK => (self.btime, self.binc),
            _ => panic!("invalid player color: {}", active_player)
        };

        self.time_limit_ms = calc_time_limit(self.move_time, time_left, inc, self.moves_to_go);

        self.strict_time_limit = self.move_time > 0 || self.time_limit_ms == MAX_TIMELIMIT_MS
            || self.moves_to_go == 1 || (time_left - (TIMEEXT_MULTIPLIER * self.time_limit_ms) <= 20);
    }
}

fn calc_time_limit(movetime: i32, time_left: i32, time_increment: i32, moves_to_go: i32) -> i32 {
    if movetime == -1 && time_left == -1 {
        return MAX_TIMELIMIT_MS;
    }

    if movetime > 0 {
        return max(0, movetime - TIME_SAFETY_MARGIN_MS);
    }

    let time_for_move = time_left / max(1, moves_to_go);

    if time_for_move > time_left - TIME_SAFETY_MARGIN_MS {
        return max(0, time_left - TIME_SAFETY_MARGIN_MS)
    }

    let time_bonus = if moves_to_go > 1 { time_increment } else { 0 };
    if time_for_move + time_bonus > time_left - TIME_SAFETY_MARGIN_MS {
        time_for_move
    } else {
        time_for_move + time_bonus
    }
}

// Principal variation collects a sequence of best moves for a given position
#[derive(Clone, Default)]
pub struct PrincipalVariation(Vec<Move>);

impl PrincipalVariation {
    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn update(&mut self, best_move: Move, follow_up: &mut PrincipalVariation) {
        self.0.clear();
        self.0.push(best_move);
        self.0.append(&mut follow_up.0);
    }

    pub fn moves(&self) -> Vec<Move> {
        self.0.clone()
    }
}

enum HelperThreadMessage {
    Search,
    Terminate,
}

struct HelperThreads {
    threads: Vec<HelperThread>
}

impl HelperThreads {
    pub fn new() -> Self {
        HelperThreads{
            threads: Vec::new()
        }
    }

    pub fn add(&mut self, tx: Sender<HelperThreadMessage>, stop_search: Arc<AtomicBool>, search_stopped: Arc<AtomicBool>, thread_terminated: Arc<AtomicBool>) {
        self.threads.push(HelperThread{tx, stop_search, search_stopped, thread_terminated})
    }

    pub fn start_search(&self) {
        for t in self.threads.iter() {
            t.search();
        }
    }

    pub fn stop_search(&self) {
        for t in self.threads.iter() {
            t.stop();
        }

        for t in self.threads.iter() {
            t.wait_till_stopped();
        }
    }

    pub fn terminate(&mut self) {
        for t in self.threads.iter() {
            t.terminate();
        }

        for t in self.threads.iter() {
            t.wait_till_terminated();
        }

        self.threads.clear();
    }


}

impl Drop for HelperThreads {
    fn drop(&mut self) {
        self.terminate();
    }
}

struct HelperThread {
    tx: Sender<HelperThreadMessage>,
    stop_search: Arc<AtomicBool>,
    search_stopped: Arc<AtomicBool>,
    thread_terminated: Arc<AtomicBool>,
}

impl HelperThread {
    pub fn search(&self) {
        self.stop_search.store(false, Ordering::Release);
        self.search_stopped.store(false, Ordering::Release);
        self.tx.send(HelperThreadMessage::Search).unwrap();
    }

    pub fn stop(&self) {
        self.stop_search.store(true, Ordering::Release);
    }

    pub fn wait_till_stopped(&self) {
        while !self.search_stopped.load(Ordering::Acquire) {
            thread::yield_now();
        }
    }

    pub fn terminate(&self) {
        self.stop();
        self.tx.send(HelperThreadMessage::Terminate).unwrap();
    }

    pub fn wait_till_terminated(&self) {
        while !self.thread_terminated.load(Ordering::Acquire) {
            thread::yield_now();
        }
    }

    pub fn run(search_stopped: Arc<AtomicBool>, rx: Receiver<HelperThreadMessage>, skipped_moves: &[Move], mut sub_search: Search) {
        sub_search.movegen.enter_ply(sub_search.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

        loop {
            let msg = match rx.recv() {
                Ok(msg) => msg,
                Err(e) => {
                    eprintln!("Helper thread communication error: {}", e);
                    break;
                }
            };

            match msg {
                HelperThreadMessage::Search => {
                    let mut window_size = INITIAL_ASPIRATION_WINDOW_SIZE;
                    let mut window_step = INITIAL_ASPIRATION_WINDOW_STEP;
                    let mut score = 0;

                    for depth in 1..MAX_DEPTH {
                        let (move_count, best_move, _, new_window_step) = sub_search.root_search(None, skipped_moves, window_step, window_size, score, depth as i32, &mut PrincipalVariation::default());
                        if new_window_step > window_step {
                            window_step = new_window_step;
                            window_size = new_window_step;
                        } else if window_step > 16 {
                            window_step /= 2;
                            window_size /= 2;
                        }

                        if move_count <= 1 {
                            break;
                        }
                        if sub_search.is_stopped() {
                            break;
                        }

                        score = best_move.score();
                    }
                    search_stopped.store(true, Ordering::Release);
                }

                HelperThreadMessage::Terminate => break
            }
        }

        sub_search.movegen.leave_ply();
    }
}

fn get_score_info(score: i32) -> String {
    if score <= MATED_SCORE + MAX_DEPTH as i32 {
        return format!("mate {}", (MATED_SCORE - score - 1) / 2);
    } else if score >= MATE_SCORE - MAX_DEPTH as i32 {
        return format!("mate {}", (MATE_SCORE - score + 1) / 2);
    }

    format!("cp {}", score)
}

#[inline]
fn log2(i: u32) -> i32 {
    (32 - i.leading_zeros()) as i32 - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::pieces::{K, R};
    use crate::moves::NO_MOVE;
    use crate::colors::{BLACK, WHITE};
    use crate::fen::{write_fen, create_from_fen};

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

        let tt = TranspositionTable::new(1);
        let limits = SearchLimits::default();
        let mut board = create_from_fen(fen.as_str());

        let m = search(tt.clone(), limits, board.clone(), 2);
        assert_ne!(NO_MOVE, m);

        board.perform_move(m);

        let is_check_mate = search(tt, limits, board.clone(), 1) == NO_MOVE && board.is_in_check(WHITE);
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

        let tt = TranspositionTable::new(1);
        let limits = SearchLimits::default();
        let mut board = create_from_fen(fen.as_str());

        let m1 = search(tt.clone(), limits, board.clone(), 3);
        board.perform_move(m1);

        let m2 = search(tt.clone(), limits, board.clone(), 2);
        board.perform_move(m2);

        let m3 = search(tt.clone(), limits, board.clone(), 1);
        board.perform_move(m3);

        let is_check_mate = search(tt, limits, board.clone(), 1) == NO_MOVE && board.is_in_check(BLACK);
        assert!(is_check_mate);
    }

    fn search(tt: Arc<TranspositionTable>, limits: SearchLimits, board: Board, min_depth: i32) -> Move {
        let mut search = Search::new(Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, board, 1, false);
        let (m, _) = search.find_best_move(None, min_depth, &[]);
        m
    }

    fn to_fen(active_player: Color, items: &[i8; 64]) -> String {
        write_fen(&Board::new(items, active_player, 0, None, 0, 1))
    }

    #[test]
    fn calc_log2() {
        for i in 1..65536 {
            assert_eq!(log2(i), (i as f32).log2() as i32)
        }
    }
}
