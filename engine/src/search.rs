/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use crate::board::castling::CastlingRules;
use crate::board::{Board, StateEntry};
use crate::colors::Color;
use crate::engine::{LogLevel, Message};
use crate::history_heuristics::HistoryHeuristics;
use crate::move_gen::{is_killer, MoveGenerator, NEGATIVE_HISTORY_SCORE};
use crate::moves::{Move, NO_MOVE};
use crate::pieces::{EMPTY, P, R};
use crate::pos_history::PositionHistory;
use crate::scores::{sanitize_score, MATED_SCORE, MATE_SCORE, MAX_SCORE, MIN_SCORE};
use crate::time_management::{TimeManager, MAX_TIMELIMIT_MS, TIMEEXT_MULTIPLIER};
use crate::transposition_table::{
    from_root_relative_score, get_depth, get_score_type, get_untyped_move, to_root_relative_score, ScoreType,
    TranspositionTable, MAX_DEPTH,
};
use crate::uci_move::UCIMove;
use std::cmp::{max, min};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use LogLevel::Info;
use crate::bitboard::BitBoards;

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

    local_total_node_count: u64,
    local_node_count: u64,

    node_count: Arc<AtomicU64>,
    is_stopped: Arc<AtomicBool>,

    threads: HelperThreads,
    is_helper_thread: bool,

    pondering: bool,
}

impl Search {
    pub fn new(
        is_stopped: Arc<AtomicBool>, node_count: Arc<AtomicU64>, log_level: LogLevel, limits: SearchLimits,
        tt: Arc<TranspositionTable>, board: Board, is_helper_thread: bool,
    ) -> Self {
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
            local_total_node_count: 0,
            local_node_count: 0,
            node_count,
            last_log_time: Instant::now(),
            next_check_node_count: 0,
            next_hh_age_node_count: 0,
            current_depth: 0,
            max_reached_depth: 0,
            is_stopped,

            threads: HelperThreads::new(),
            is_helper_thread,

            pondering: false,
        }
    }

    pub fn resize_tt(&mut self, new_size_mb: i32) {
        // Remove all additional threads, which reference the transposition table
        let thread_count = self.threads.count();
        self.threads.resize(0, &self.node_count, &self.tt, &self.board, &self.is_stopped);

        // Resize transposition table
        Arc::get_mut(&mut self.tt).unwrap().resize(new_size_mb as u64);

        // Restart threads
        self.threads.resize(thread_count, &self.node_count, &self.tt, &self.board, &self.is_stopped);

        self.clear_tt();
    }

    pub fn reset_threads(&mut self, thread_count: i32) {
        self.threads.resize((thread_count - 1) as usize, &self.node_count, &self.tt, &self.board, &self.is_stopped);
    }

    pub fn clear_tt(&mut self) {
        self.threads.clear_tt();
        TranspositionTable::clear(&self.tt, 0, self.threads.count() + 1);
    }

    pub fn update(&mut self, board: &Board, limits: SearchLimits, ponder: bool) {
        self.pondering = ponder;
        self.board.reset(
            board.pos_history.clone(),
            board.bitboards,
            board.halfmove_count,
            board.state,
            board.castling_rules,
        );
        self.limits = limits;
    }

    pub fn update_limits(&mut self, limits: SearchLimits) {
        self.limits = limits;
    }

    pub fn reset(&mut self) {
        self.local_total_node_count = 0;
        self.local_node_count = 0;
        self.next_hh_age_node_count = 1000000;
        self.hh.clear();
    }

    pub fn find_best_move(
        &mut self, rx: Option<&Receiver<Message>>, min_depth: i32, skipped_moves: &[Move],
    ) -> (Move, PrincipalVariation) {
        self.reset();
        self.time_mgr.reset(self.limits.time_limit_ms, self.limits.strict_time_limit);

        self.last_log_time = Instant::now();

        self.cancel_possible = false;
        self.node_count.store(0, Ordering::Relaxed);

        self.next_check_node_count = min(self.limits.node_limit, 1000);

        let mut last_best_move: Move = NO_MOVE;

        let active_player = self.board.active_player();

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

        self.set_stopped(false);

        let mut pv = PrincipalVariation::default();

        self.threads.start_search(&self.board, skipped_moves);

        let mut window_size = INITIAL_ASPIRATION_WINDOW_SIZE;
        let mut window_step = INITIAL_ASPIRATION_WINDOW_STEP;

        let mut score = 0;

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in 1..=self.limits.depth_limit {
            let iteration_start_time = Instant::now();
            self.current_depth = depth;
            self.max_reached_depth = 0;

            let mut iteration_cancelled = false;

            let mut local_pv = PrincipalVariation::default();
            let (move_num, mut best_move, current_pv, new_window_step) =
                self.root_search(rx, skipped_moves, window_step, window_size, score, depth, &mut local_pv);
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
                if !self.pondering
                    && self.cancel_possible
                    && !self.time_mgr.is_time_for_another_iteration(now, iteration_duration)
                {
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

            if iteration_cancelled || move_num == 0 {
                // stop searching, if iteration has been cancelled or there is no valid move or only a single valid move
                break;
            }

            if depth == 1 && move_num == 1 {
                self.time_mgr.reduce_timelimit();
            }
        }

        self.movegen.leave_ply();

        if let Some(r) = rx {
            while self.pondering && !self.is_stopped() {
                self.check_messages(r, true);
            }
        }

        self.set_stopped(true);

        self.threads.wait_for_completion();

        (last_best_move, pv)
    }

    fn root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], window_step: i32, window_size: i32,
        score: i32, depth: i32, pv: &mut PrincipalVariation,
    ) -> (i32, Move, Option<String>, i32) {
        let mut alpha = if depth > 7 { score - window_size } else { MIN_SCORE };
        let mut beta = if depth > 7 { score + window_size } else { MAX_SCORE };

        let mut step = window_step;
        loop {
            pv.clear();

            let (move_num, best_move, current_pv) = self.bounded_root_search(rx, skipped_moves, alpha, beta, depth, pv);

            // Bulk update of global node count
            if self.local_node_count > 0 {
                self.node_count.fetch_add(self.local_node_count, Ordering::Relaxed);
                self.local_node_count = 0;
            }

            if best_move == NO_MOVE {
                return (move_num, best_move, current_pv, step);
            }

            let best_score = best_move.score();
            if best_score <= alpha {
                alpha = max(MIN_SCORE, alpha.saturating_sub(step));
            } else if best_score >= beta {
                beta = min(MAX_SCORE, beta.saturating_add(step));
            } else {
                return (move_num, best_move, current_pv, step);
            }

            step = min(MATE_SCORE / 2, step.saturating_mul(2));
        }
    }

    // Root search within the bounds of an aspiration window (alpha...beta)
    fn bounded_root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], mut alpha: i32, beta: i32, depth: i32,
        pv: &mut PrincipalVariation,
    ) -> (i32, Move, Option<String>) {
        let mut move_num = 0;
        let mut a = -beta;
        let mut best_move: Move = NO_MOVE;
        let mut best_score = MIN_SCORE;

        let mut iteration_cancelled = false;
        let mut current_pv = None;

        let active_player = self.board.active_player();

        let mut reduction = 0;
        let mut tree_scale = 0;

        self.movegen.reset_root_moves();
        while let Some(m) = self.movegen.next_root_move(&self.hh, &mut self.board) {
            if skipped_moves.contains(&m.without_score()) {
                continue;
            }
            move_num += 1;

            if self.log(Info) && self.local_total_node_count > 2000000 {
                let now = Instant::now();
                if self.time_mgr.search_duration_ms(now) >= 1000 {
                    self.last_log_time = now;
                    let uci_move = UCIMove::from_move(&self.board, m);
                    println!("info depth {} currmove {} currmovenumber {}", depth, uci_move, move_num);
                }
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(m);

            let capture_pos = if removed_piece_id != EMPTY { m.end() } else { -1 };

            let gives_check = self.board.is_in_check(active_player.flip());

            let mut local_pv = PrincipalVariation::default();

            let mut tree_size = self.local_total_node_count;

            // Use principal variation search
            self.inc_node_count();
            let mut result = self.rec_find_best_move(
                rx,
                a,
                -alpha,
                depth - reduction - 1,
                1,
                SearchFlags::new().check(gives_check),
                capture_pos,
                &mut local_pv,
                m,
                NO_MOVE,
            );
            if result == CANCEL_SEARCH {
                iteration_cancelled = true;
            } else if -result > alpha && a != -beta {
                // Repeat search if it falls outside the search window
                result = self.rec_find_best_move(
                    rx,
                    -beta,
                    -alpha,
                    depth - 1,
                    1,
                    SearchFlags::new().check(gives_check),
                    capture_pos,
                    &mut local_pv,
                    m,
                    NO_MOVE,
                );
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

                alpha = score;

                self.time_mgr.update_best_move(best_move, depth);

                if depth >= 7 && reduction == 0 {
                    reduction = 1;
                } else {
                    reduction = 0;
                }
                a = -(alpha + 1);
            }

            tree_size = (self.local_total_node_count - tree_size) << reduction;
            if move_num == 1 {
                tree_scale = max(13, 64 - tree_size.leading_zeros()) - 13;
            }
            self.movegen.update_root_move(m.with_score(min(MAX_SCORE, (tree_size >> tree_scale) as i32)));
        }

        self.movegen.reorder_root_moves(best_move, self.is_helper_thread);

        (move_num, best_move, current_pv)
    }

    fn node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }

    fn inc_node_count(&mut self) {
        self.local_total_node_count += 1;
        self.local_node_count += 1;
    }

    // Recursively calls itself with alternating player colors to
    // find the best possible move in response to the current board position.
    fn rec_find_best_move(
        &mut self, rx: Option<&Receiver<Message>>, mut alpha: i32, beta: i32, mut depth: i32, ply: i32,
        flags: SearchFlags, capture_pos: i32, pv: &mut PrincipalVariation, opponent_move: Move,
        excluded_singular_move: Move,
    ) -> i32 {
        self.max_reached_depth = max(ply, self.max_reached_depth);

        if let Some(rx) = rx {
            self.check_search_limits(rx)
        }

        if self.is_stopped() {
            return CANCEL_SEARCH;
        }

        if self.local_node_count > 1000 {
            // Bulk update of global node count
            self.node_count.fetch_add(self.local_node_count, Ordering::Relaxed);
            self.local_node_count = 0;
        }

        if self.board.is_draw() {
            return 0;
        }

        if self.local_total_node_count >= self.next_hh_age_node_count {
            self.hh.age_entries();
            self.next_hh_age_node_count = self.local_total_node_count + 2000000;
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        // Prune, if even the best possible score cannot improve alpha (because a shorter mate has already been found)
        let best_possible_score = MATE_SCORE - ply - 1;
        if best_possible_score <= alpha {
            return best_possible_score;
        }

        // Prune, if worst possible score is already sufficient to reach beta
        let worst_possible_score = MATED_SCORE + ply + if flags.in_check() { 0 } else { 1 };
        if worst_possible_score >= beta {
            return worst_possible_score;
        }

        let mut pos_score: Option<i32> = None;

        let active_player = self.board.active_player();
        if flags.in_check() {
            // Extend search when in check
            depth = max(1, depth + 1);
        }

        let hh_counter_scale = self.hh.calc_counter_scale(depth);

        let hash = self.board.get_hash();

        let mut hash_move = NO_MOVE;
        let mut hash_score = 0;

        let mut check_se = false;
        if excluded_singular_move == NO_MOVE {
            // Check transposition table
            let tt_entry = self.tt.get_entry(hash);
            if tt_entry != 0 {
                hash_move = self.movegen.sanitize_move(&self.board, active_player, get_untyped_move(tt_entry));

                if hash_move != NO_MOVE {
                    hash_score = to_root_relative_score(ply, sanitize_score(hash_move.score()));
                    let tt_depth = get_depth(tt_entry);
                    match get_score_type(tt_entry) {
                        ScoreType::Exact => {
                            if !is_pv && tt_depth >= depth {
                                return hash_score;
                            }
                            pos_score = Some(hash_score);
                            check_se = tt_depth >= depth - 3;
                        }

                        ScoreType::UpperBound => {
                            if hash_score <= alpha && tt_depth >= depth {
                                return hash_score;
                            }
                        }

                        ScoreType::LowerBound => {
                            if tt_depth >= depth && max(alpha, hash_score) >= beta {
                                if hash_move.is_quiet() {
                                    self.hh.update_killer_moves(ply, hash_move);
                                    self.hh.update_counter_move(opponent_move, hash_move);
                                }
                                return hash_score;
                            }
                            pos_score = Some(hash_score);
                            check_se = tt_depth >= depth - 3;
                        }
                    };

                    check_se = check_se
                        && !flags.in_singular_search()
                        && !flags.in_null_move_search()
                        && !flags.in_check()
                        && depth > 7
                        && capture_pos != hash_move.end();
                }
            } else if depth > 7 {
                // Reduce nodes without hash move from transposition table
                depth -= 1;
            }

            // Quiescence search
            if depth <= 0 || ply >= (MAX_DEPTH - 16) as i32 {
                return self.quiescence_search(rx, active_player, alpha, beta, ply, pos_score, pv);
            }

            if !is_pv && !flags.in_check() {
                if depth <= 3 {
                    // Jump directly to QS, if position is already so good, that it is unlikely for the opponent to counter it within the remaining search depth
                    pos_score = pos_score.or_else(|| Some(active_player.score(self.board.eval())));
                    let score = pos_score.unwrap();

                    if score.abs() < MATE_SCORE - (2 * MAX_DEPTH as i32) && score - (100 * depth) >= beta {
                        return self.quiescence_search(rx, active_player, alpha, beta, ply, pos_score, pv);
                    }
                } else if !self.board.is_pawn_endgame() {
                    // Null move pruning
                    pos_score = pos_score.or_else(|| Some(active_player.score(self.board.eval())));
                    if pos_score.unwrap() >= beta {
                        self.tt.prefetch(self.board.get_hash());
                        let r = log2((depth * 3 - 3) as u32);
                        self.board.perform_null_move();
                        let result = self.rec_find_best_move(
                            rx,
                            -beta,
                            -beta + 1,
                            depth - r - 1,
                            ply + 1,
                            flags.null_move_search(),
                            -1,
                            &mut PrincipalVariation::default(),
                            NO_MOVE,
                            NO_MOVE,
                        );
                        self.board.undo_null_move();
                        if result == CANCEL_SEARCH {
                            return CANCEL_SEARCH;
                        }
                        if -result >= beta {
                            return if result.abs() < MATE_SCORE - 2 * MAX_DEPTH as i32 { -result } else { beta };
                        }
                    }
                }
            }
        }

        let mut best_score = worst_possible_score;
        let mut best_move = NO_MOVE;

        let mut score_type = ScoreType::UpperBound;
        let mut evaluated_move_count = 0;
        let mut has_valid_moves = false;

        let allow_lmr = depth > 2;

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        if !is_pv && depth <= 6 && !flags.in_check() {
            let margin = (6 << depth) * 4 + 16;
            let prune_low_score = pos_score.unwrap_or_else(|| active_player.score(self.board.eval()));
            allow_futile_move_pruning =
                prune_low_score.abs() < MATE_SCORE - 2 * MAX_DEPTH as i32 && prune_low_score + margin <= alpha;
        }

        let (primary_killer, secondary_killer) = self.hh.get_killer_moves(ply);
        let counter_move = self.hh.get_counter_move(opponent_move);
        self.movegen.enter_ply(active_player, hash_move, primary_killer, secondary_killer, counter_move);

        let occupied_bb = self.board.get_occupancy_bitboard();

        let previous_move_was_capture = capture_pos != -1;

        let mut is_singular = false;

        let mut a = -beta;
        while let Some(curr_move) = self.movegen.next_move(&self.hh, &mut self.board) {
            if curr_move.is_same_move(excluded_singular_move) {
                continue;
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(curr_move);
            let gives_check = self.board.is_in_check(active_player.flip());

            // Check, if the hash move is singular and should be extended
            let mut se_extension = 0;
            if check_se && !gives_check && curr_move.is_same_move(hash_move) {
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                let se_beta = hash_score - (5 + depth / 2);
                let result = self.rec_find_best_move(
                    rx,
                    se_beta - 1,
                    se_beta,
                    depth / 2,
                    ply,
                    flags.singular_search(),
                    capture_pos,
                    &mut PrincipalVariation::default(),
                    opponent_move,
                    hash_move,
                );

                if result == CANCEL_SEARCH {
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                self.board.perform_move(curr_move);

                if result < se_beta {
                    se_extension = 1;
                    is_singular = true;
                }
            };

            let start = curr_move.start();
            let end = curr_move.end();

            let mut skip = self.board.is_in_check(active_player); // skip if move would put own king in check

            let mut reductions = 0;

            if !skip {
                let target_piece_id = curr_move.piece_id();
                has_valid_moves = true;

                if !is_pv && previous_move_was_capture && evaluated_move_count > 0 && capture_pos != curr_move.end() {
                    reductions = 1;
                }

                if se_extension == 0 {
                    if removed_piece_id == EMPTY {
                        if allow_lmr && evaluated_move_count > LMR_THRESHOLD {
                            reductions += if is_pv { 1 } else { 2 };
                            if curr_move.score() == NEGATIVE_HISTORY_SCORE {
                                reductions += 1;
                            }

                            if evaluated_move_count >= 6 && excluded_singular_move != NO_MOVE {
                                reductions += 1;
                            }
                        } else if allow_futile_move_pruning && !gives_check && !curr_move.is_queen_promotion() {
                            // Reduce futile move
                            reductions += FUTILE_MOVE_REDUCTIONS;
                        } else if !is_pv
                            && (curr_move.score() == NEGATIVE_HISTORY_SCORE
                                || self.board.has_negative_see(
                                    active_player.flip(),
                                    start,
                                    end,
                                    target_piece_id,
                                    EMPTY,
                                    0,
                                    occupied_bb,
                                ))
                        {
                            // Reduce search depth for moves with negative history or negative SEE score
                            reductions += LOSING_MOVE_REDUCTIONS;
                            if evaluated_move_count > 0 && depth <= 3 {
                                skip = true;
                            }
                        }

                        if is_singular {
                            reductions += 1;
                        }

                        if allow_futile_move_pruning
                            && evaluated_move_count > 0
                            && !gives_check
                            && reductions >= (depth - 1)
                        {
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
            }

            if skip {
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
            } else {
                self.tt.prefetch(self.board.get_hash());

                let new_capture_pos = if removed_piece_id != EMPTY { end } else { -1 };

                evaluated_move_count += 1;

                let mut local_pv = PrincipalVariation::default();

                self.inc_node_count();
                let mut result = self.rec_find_best_move(
                    rx,
                    a,
                    -alpha,
                    depth + se_extension - reductions - 1,
                    ply + 1,
                    flags.check(gives_check),
                    new_capture_pos,
                    &mut local_pv,
                    curr_move,
                    NO_MOVE,
                );
                if result == CANCEL_SEARCH {
                    self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                if -result > alpha && (reductions > 0 || (-result < beta && a != -beta)) {
                    // Repeat search without reduction and with full window
                    result = self.rec_find_best_move(
                        rx,
                        -beta,
                        -alpha,
                        depth + se_extension - 1,
                        ply + 1,
                        flags.check(gives_check),
                        new_capture_pos,
                        &mut local_pv,
                        curr_move,
                        NO_MOVE,
                    );
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
                        if excluded_singular_move == NO_MOVE {
                            self.tt.write_entry(
                                hash,
                                depth,
                                best_move.with_score(from_root_relative_score(ply, best_score)),
                                ScoreType::LowerBound,
                            );
                        }

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
            return if excluded_singular_move != NO_MOVE {
                alpha
            } else if flags.in_check() {
                MATED_SCORE + ply // Check mate
            } else {
                0 // Stale mate
            };
        }

        if excluded_singular_move == NO_MOVE {
            self.tt.write_entry(
                hash,
                depth,
                best_move.with_score(from_root_relative_score(ply, best_score)),
                score_type,
            );
        }

        best_score
    }

    fn check_search_limits(&mut self, rx: &Receiver<Message>) {
        if self.local_total_node_count >= self.next_check_node_count {
            self.next_check_node_count = if self.limits.node_limit != u64::MAX {
                self.limits.node_limit
            } else {
                self.local_total_node_count + 1000
            };

            self.check_messages(rx, false);

            let now = Instant::now();
            if !self.pondering
                && self.cancel_possible
                && (self.node_count() >= self.limits.node_limit || self.time_mgr.is_timelimit_exceeded(now))
            {
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

    pub fn quiescence_search(
        &mut self, rx: Option<&Receiver<Message>>, active_player: Color, mut alpha: i32, beta: i32, ply: i32,
        pos_score: Option<i32>, pv: &mut PrincipalVariation,
    ) -> i32 {
        if self.is_stopped() {
            return CANCEL_SEARCH;
        }

        self.max_reached_depth = max(ply, self.max_reached_depth);

        if self.board.is_insufficient_material_draw() {
            return 0;
        }

        let position_score = pos_score.unwrap_or_else(|| active_player.score(self.board.eval()));
        if ply >= MAX_DEPTH as i32 {
            return position_score;
        }

        if position_score >= beta {
            return position_score;
        }

        // Prune nodes where the position score is already so far below alpha that it is very unlikely to be raised by any available move
        let prune_low_captures = position_score < alpha - QS_PRUNE_MARGIN;

        let mut best_score = position_score;
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
            if self.board.has_negative_see(
                active_player.flip(),
                start,
                end,
                previous_piece_id,
                captured_piece_id,
                threshold,
                occupied_bb,
            ) {
                continue;
            }

            let (previous_piece, move_state) = self.board.perform_move(m);

            if self.board.is_in_check(active_player) {
                // Invalid move
                self.board.undo_move(m, previous_piece, move_state);
                continue;
            }

            let mut local_pv = PrincipalVariation::default();

            self.inc_node_count();
            let score = -self.quiescence_search(rx, active_player.flip(), -beta, -alpha, ply + 1, None, &mut local_pv);
            self.board.undo_move(m, previous_piece, move_state);

            if score <= best_score {
                // No improvement
                continue;
            }

            best_score = score;
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
        best_score
    }

    fn get_base_stats(&self, duration: Duration) -> String {
        let node_count = self.node_count();
        let duration_micros = duration.as_micros();
        let nodes_per_second = if duration_micros > 0 { node_count as u128 * 1_000_000 / duration_micros } else { 0 };

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
            let uci_move = UCIMove::from_move(&self.board, *m);
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

                _ => (),
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
                Err(e) => {
                    if matches!(e, TryRecvError::Empty) {
                        None
                    } else {
                        self.uci_channel_error(e.to_string());
                        None
                    }
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

    pub fn new(
        depth_limit: Option<i32>, node_limit: Option<u64>, wtime: Option<i32>, btime: Option<i32>, winc: Option<i32>,
        binc: Option<i32>, move_time: Option<i32>, moves_to_go: Option<i32>,
    ) -> Result<Self, &'static str> {
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
            moves_to_go: moves_to_go.unwrap_or(40),
        })
    }

    pub fn update(&mut self, active_player: Color) {
        let (time_left, inc) = if active_player.is_white() { (self.wtime, self.winc) } else { (self.btime, self.binc) };

        self.time_limit_ms = calc_time_limit(self.move_time, time_left, inc, self.moves_to_go);

        self.strict_time_limit = self.move_time > 0
            || self.time_limit_ms == MAX_TIMELIMIT_MS
            || self.moves_to_go == 1
            || (time_left - (TIMEEXT_MULTIPLIER * self.time_limit_ms) <= 20);
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
        return max(0, time_left - TIME_SAFETY_MARGIN_MS);
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

    pub fn add(&mut self, best_move: Move) {
        self.0.push(best_move);
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

enum ToThreadMessage {
    Search {
        pos_history: PositionHistory,
        bitboards: BitBoards,
        halfmove_count: u16,
        state: StateEntry,
        castling_rules: CastlingRules,
        skipped_moves: Vec<Move>,
    },
    ClearTT {
        thread_no: usize,
        total_threads: usize,
    },
    Terminate,
}

type FromThreadMessage = ();

struct HelperThreads {
    threads: Vec<HelperThread>,
}

impl HelperThreads {
    pub fn new() -> Self {
        HelperThreads { threads: Vec::new() }
    }

    pub fn resize(
        &mut self, target_count: usize, node_count: &Arc<AtomicU64>, tt: &Arc<TranspositionTable>, board: &Board,
        is_stopped: &Arc<AtomicBool>,
    ) {
        if target_count < self.threads.len() {
            self.threads.drain(target_count..).for_each(|t| {
                t.terminate();
                t.handle.join().unwrap();
            });
            return;
        }

        let additional_count = target_count - self.threads.len();
        for _ in 0..additional_count {
            let (to_tx, to_rx) = channel::<ToThreadMessage>();
            let (from_tx, from_rx) = channel::<FromThreadMessage>();

            let node_count = node_count.clone();
            let tt = tt.clone();
            let board = board.clone();
            let is_stopped = is_stopped.clone();

            let handle = thread::spawn(move || {
                let limits = SearchLimits::default();
                let sub_search = Search::new(is_stopped, node_count, LogLevel::Error, limits, tt, board, true);
                HelperThread::run(to_rx, from_tx, sub_search);
            });

            self.threads.push(HelperThread { handle, to_tx, from_rx });
        }
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    pub fn start_search(&self, board: &Board, skipped_moves: &[Move]) {
        for t in self.threads.iter() {
            t.search(board, skipped_moves);
        }
    }

    pub fn clear_tt(&self) {
        let total_count = self.threads.len() + 1;
        for (i, t) in self.threads.iter().enumerate() {
            t.clear_tt(i + 1, total_count);
        }

        self.wait_for_completion();
    }

    pub fn wait_for_completion(&self) {
        for t in self.threads.iter() {
            t.wait_for_completion();
        }
    }

    pub fn terminate(&mut self) {
        for t in self.threads.iter() {
            t.terminate();
        }

        while let Some(t) = self.threads.pop() {
            t.handle.join().unwrap();
        }
    }
}

impl Drop for HelperThreads {
    fn drop(&mut self) {
        self.terminate();
    }
}

struct HelperThread {
    handle: JoinHandle<()>,
    to_tx: Sender<ToThreadMessage>,
    from_rx: Receiver<FromThreadMessage>,
}

impl HelperThread {
    pub fn search(&self, board: &Board, skipped_moves: &[Move]) {
        match self.to_tx.send(ToThreadMessage::Search {
            pos_history: board.pos_history.clone(),
            bitboards: board.bitboards,
            halfmove_count: board.halfmove_count,
            state: board.state,
            castling_rules: board.castling_rules,
            skipped_moves: Vec::from(skipped_moves),
        }) {
            Ok(_) => {}
            Err(e) => {
                println!("Could not send message: {}", e);
                panic!("Could not send message!");
            }
        };
    }

    pub fn terminate(&self) {
        self.to_tx.send(ToThreadMessage::Terminate).unwrap();
    }

    pub fn clear_tt(&self, thread_no: usize, total_threads: usize) {
        self.to_tx.send(ToThreadMessage::ClearTT { thread_no, total_threads }).unwrap();
    }

    pub fn wait_for_completion(&self) {
        match self.from_rx.recv() {
            Ok(_) => (),
            Err(e) => {
                eprintln!("Channel communication error while waiting for helper thread: {}", e);
            }
        };
    }

    pub fn run(rx: Receiver<ToThreadMessage>, tx: Sender<FromThreadMessage>, mut sub_search: Search) {
        loop {
            let msg = match rx.recv() {
                Ok(msg) => msg,
                Err(e) => {
                    eprintln!("Helper thread communication error: {}", e);
                    break;
                }
            };

            match msg {
                ToThreadMessage::Search {
                    pos_history,
                    bitboards,
                    halfmove_count,
                    state,
                    castling_rules,
                    skipped_moves,
                } => {
                    let mut window_size = INITIAL_ASPIRATION_WINDOW_SIZE;
                    let mut window_step = INITIAL_ASPIRATION_WINDOW_STEP;
                    let mut score = 0;

                    sub_search.reset();
                    sub_search.board.reset(pos_history, bitboards, halfmove_count, state, castling_rules);

                    sub_search.movegen.enter_ply(sub_search.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

                    for depth in 1..MAX_DEPTH {
                        let (move_count, best_move, _, new_window_step) = sub_search.root_search(
                            None,
                            &skipped_moves,
                            window_step,
                            window_size,
                            score,
                            depth as i32,
                            &mut PrincipalVariation::default(),
                        );
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

                    sub_search.movegen.leave_ply();

                    tx.send(()).unwrap();
                }

                ToThreadMessage::ClearTT { thread_no, total_threads } => {
                    sub_search.tt.clear(thread_no, total_threads);
                    tx.send(()).unwrap();
                }

                ToThreadMessage::Terminate => break,
            }
        }
    }
}

pub struct SearchFlags(u8);

impl SearchFlags {
    const IN_CHECK: u8 = 0b0001;
    const IN_SINGULAR_SEARCH: u8 = 0b0010;
    const IN_NULL_MOVE_SEARCH: u8 = 0b0100;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn in_check(&self) -> bool {
        self.0 & SearchFlags::IN_CHECK != 0
    }

    pub fn check(&self, gives_check: bool) -> Self {
        if gives_check {
            SearchFlags(self.0 | SearchFlags::IN_CHECK)
        } else {
            SearchFlags(self.0 & !SearchFlags::IN_CHECK)
        }
    }

    pub fn in_singular_search(&self) -> bool {
        self.0 & SearchFlags::IN_SINGULAR_SEARCH != 0
    }

    pub fn singular_search(&self) -> Self {
        SearchFlags(self.0 | SearchFlags::IN_SINGULAR_SEARCH)
    }

    pub fn in_null_move_search(&self) -> bool {
        self.0 & SearchFlags::IN_NULL_MOVE_SEARCH != 0
    }

    pub fn null_move_search(&self) -> Self {
        SearchFlags(self.0 | SearchFlags::IN_NULL_MOVE_SEARCH)
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
    use crate::board::castling::{CastlingRules, CastlingState};
    use crate::board::Board;
    use crate::colors::{BLACK, WHITE};
    use crate::fen::{create_from_fen, write_fen};
    use crate::moves::NO_MOVE;
    use crate::pieces::{K, R};

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
        let mut search = Search::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
            LogLevel::Error,
            limits,
            tt,
            board,
            false,
        );
        let (m, _) = search.find_best_move(None, min_depth, &[]);
        m
    }

    fn to_fen(active_player: Color, items: &[i8; 64]) -> String {
        write_fen(&Board::new(items, active_player, CastlingState::default(), None, 0, 1, CastlingRules::default()))
    }

    #[test]
    fn calc_log2() {
        for i in 1..65536 {
            assert_eq!(log2(i), (i as f32).log2() as i32)
        }
    }
}
