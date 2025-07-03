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

use crate::bitboard::{BitBoards, is_passed_pawn};
use crate::board::castling::CastlingRules;
use crate::board::{Board, StateEntry};
use crate::colors::{Color, WHITE};
use crate::engine::{LogLevel, Message};
use crate::history_heuristics::{EMPTY_HISTORY, HistoryHeuristics, MIN_HISTORY_SCORE};
use crate::move_gen::{is_killer, NEGATIVE_HISTORY_SCORE, QUIET_BASE_SCORE, is_valid_move};
use crate::moves::{Move, MoveType, NO_MOVE};
use crate::pieces::{EMPTY, P};
use crate::pos_history::PositionHistory;
use crate::scores::{mate_in, sanitize_score, MATED_SCORE, MATE_SCORE, MAX_SCORE, MIN_SCORE, is_mate_or_mated_score, MAX_EVAL, MIN_EVAL, is_eval_score, clock_scaled_eval};
use crate::time_management::{SearchLimits, TimeManager};
use crate::transposition_table::{ScoreType, TranspositionTable, MAX_DEPTH, get_tt_move, get_depth, get_score_type};
use crate::uci_move::UCIMove;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use LogLevel::Info;
use crate::{next_ply};
use crate::nn::io::FastHasher;
use crate::params::{lmr_idx, DerivedArrayParams, SingleParams};
use crate::search_context::{SearchContext};
use crate::syzygy::{DEFAULT_TB_PROBE_DEPTH, ProbeTB};
use crate::syzygy::tb::{TBResult};


pub const DEFAULT_SEARCH_THREADS: usize = 1;
pub const MAX_SEARCH_THREADS: usize = 4096;

const LMR_THRESHOLD: i16 = 1;

const FUTILE_MOVE_REDUCTIONS: i32 = 2;
const NEG_HISTORY_REDUCTIONS: i32 = 2;
const NEG_SEE_REDUCTIONS: i32 = 2;

const INITIAL_ASPIRATION_WINDOW_SIZE: i16 = 16;

type MoveSet = HashSet<Move, BuildHasherDefault<FastHasher>>;

pub struct Search {
    pub board: Board,
    pub hh: HistoryHeuristics,
    pub tt: Arc<TranspositionTable>,
    ctx: SearchContext,

    log_level: LogLevel,
    limits: SearchLimits,
    time_mgr: TimeManager,

    cancel_possible: bool,
    last_log_time: Instant,
    next_check_node_count: u64,
    current_depth: i32,
    pub max_reached_depth: usize,

    local_total_node_count: u64,
    local_tb_hits: u64,
    local_node_count: u64,
    multi_pv_count: usize,
    tb_probe_depth: i32,
    tb_probe_root: bool,
    is_tb_root: bool,

    node_count: Arc<AtomicU64>,
    is_stopped: Arc<AtomicBool>,
    tb_hits: Arc<AtomicU64>,

    threads: HelperThreads,
    is_helper_thread: bool,

    player_pov: Color,

    pondering: bool,
    is_strength_limited: bool,

    params: SingleParams,
    derived_params: DerivedArrayParams,
}

impl Search {
    pub fn new(
        is_stopped: Arc<AtomicBool>, node_count: Arc<AtomicU64>, tb_hits: Arc<AtomicU64>, log_level: LogLevel, limits: SearchLimits,
        tt: Arc<TranspositionTable>, board: Board, is_helper_thread: bool,
    ) -> Self {
        let hh = HistoryHeuristics::default();

        let time_mgr = TimeManager::new();
        
        let params = SingleParams::default();
        let derived_params = DerivedArrayParams::new(&params);

        Search {
            log_level,
            limits,
            tt,
            ctx: SearchContext::default(),
            board,
            hh,
            time_mgr,
            cancel_possible: false,
            local_total_node_count: 0,
            local_node_count: 0,
            local_tb_hits: 0,
            node_count,
            tb_hits,
            last_log_time: Instant::now(),
            next_check_node_count: 0,
            current_depth: 0,
            max_reached_depth: 0,
            is_stopped,
            multi_pv_count: 1,
            tb_probe_depth: DEFAULT_TB_PROBE_DEPTH,
            tb_probe_root: true,
            is_tb_root: false,

            player_pov: WHITE,

            threads: HelperThreads::new(),
            is_helper_thread,

            pondering: false,
            is_strength_limited: false,

            params,
            derived_params,
        }
    }

    pub fn set_param(&mut self, name: &str, value: i16) -> bool {
        if let Some(updated) = self.params.set_param(name, value) {
            if updated {
                self.derived_params.update(&self.params);
            }
            
            return true;
        }
        false
    }

    pub fn set_params(&mut self, params: SingleParams) {
        self.params = params;
    }

    pub fn resize_tt(&mut self, new_size_mb: i32) {
        // Remove all additional threads, which reference the transposition table
        let thread_count = self.threads.count();
        self.threads.resize(0, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped, self.params);

        // Resize transposition table
        Arc::get_mut(&mut self.tt).unwrap().resize(new_size_mb as u64);

        // Restart threads
        self.threads.resize(thread_count, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped, self.params);

        self.clear_tt();
    }

    pub fn adjust_thread_count(&mut self, thread_count: i32) {
        self.threads.resize((thread_count - 1) as usize, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped, self.params);
    }

    pub fn clear_tt(&mut self) {
        self.threads.clear_tt();
        TranspositionTable::clear(&self.tt, 0, self.threads.count() + 1);
    }
    
    pub fn reset_threads(&mut self) {
        self.threads.reset();
    }

    pub fn set_multi_pv_count(&mut self, count: i32) {
        self.multi_pv_count = count as usize;
    }

    pub fn set_tb_probe_depth(&mut self, depth: i32) {
        self.tb_probe_depth = depth;
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
        self.time_mgr.reset(self.limits);
    }

    pub fn update_limits(&mut self, limits: SearchLimits) {
        self.limits = limits;
    }

    pub fn reset(&mut self) {
        self.local_total_node_count = 0;
        self.local_node_count = 0;
        self.local_tb_hits = 0;
    }

    pub fn find_move_with_limited_strength(
        &mut self, rx: Option<&Receiver<Message>>, simulate_thinking_time: bool, skipped_moves: &[Move],
    ) -> (Move, PrincipalVariation) {
        self.ctx.set_root_move_randomization(true);
        self.limits.set_strict_time_limit(true);
        self.is_strength_limited = true;

        if self.hh.is_empty() {
            // Fill history tables for the very first move
            let limits = self.limits;

            self.limits.set_node_limit(limits.node_limit() * 4);
            let _ = self.find_best_move(rx, skipped_moves);
            self.limits = limits;
        }

        let mut all_skipped = Vec::from(skipped_moves);
        let base_candidate = self.find_best_move(rx, &all_skipped);
        if self.ctx.root_move_count() == 0 {
            self.set_stopped(true);
            return (NO_MOVE, PrincipalVariation::default());
        }

        let active_player = self.board.active_player();
        let eval = next_ply!(self.ctx, self.qs::<false>(active_player, MIN_SCORE, MAX_SCORE, self.board.is_in_check(active_player))).unwrap_or(MATED_SCORE);
        if eval >= 2000 {
            self.limits.set_node_limit(self.limits.node_limit().max(200) * 4);
        }
        
        self.ctx.prepare_moves(active_player, NO_MOVE, EMPTY_HISTORY);
        self.ctx.reset_root_moves();

        all_skipped.push(base_candidate.0);
        while let Some(m) = self.ctx.next_root_move(&self.hh, &mut self.board) {
            if self.board.has_negative_see(active_player.flip(), m.start() as usize, m.end() as usize, m.move_type().piece_id(), self.board.get_item(m.end() as usize), self.board.occupancy_bb()) {
                all_skipped.push(m);
            }
        }

        let (base_qs_score, base_tactical) = self.get_qs_score(base_candidate.0);

        let alt_candidate = self.find_best_move(rx, &all_skipped);
        let (alt_qs_score, alt_tactical) = if alt_candidate.0 != NO_MOVE { self.get_qs_score(alt_candidate.0) } else { (MIN_SCORE, false) };

        let (m, pv) =
            if (base_tactical || alt_tactical) && alt_qs_score > base_qs_score + 75 {
                alt_candidate
            } else {
                base_candidate
            };

        let mut result = AnalysisResult::new();
        result.update_result(self.max_reached_depth as i32, self.max_reached_depth as i32, m, Some(self.pv_info(&pv.0)), pv.clone());
        result.print(self.board.halfmove_clock(), None, self.multi_pv_count, self.get_base_stats(self.time_mgr.search_duration(Instant::now())));

        if simulate_thinking_time && self.limits.has_time_limit() {
            if let Some(r) = rx {
                if self.ctx.root_move_count() == 1 || self.board.fullmove_count() <= 4 {
                    self.time_mgr.reduce_timelimit();
                }
                self.set_stopped(false);
                println!("info string Simulate thinking time for {} seconds ...", (self.time_mgr.remaining_time_ms(Instant::now()) + 500) / 1000);
                while self.time_mgr.remaining_time_ms(Instant::now()) > 100 && !self.is_stopped() {
                    self.check_messages(r, false);
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }

        result.print(self.board.halfmove_clock(), None, self.multi_pv_count, self.get_base_stats(self.time_mgr.search_duration(Instant::now())));

        self.set_stopped(true);
        (m, pv.clone())
    }

    fn get_qs_score(&mut self, m: Move) -> (i16, bool) {
        self.local_total_node_count = 0;
        let (previous_piece, removed_piece_id) = self.board.perform_move(m);
        let active_player = self.board.active_player();
        let gives_check = self.board.is_in_check(active_player);
        let mut pv = PrincipalVariation::default();
        let qs_score = self.quiescence_search(active_player, MIN_SCORE, MAX_SCORE, gives_check).unwrap_or(MATED_SCORE);

        self.ctx.update_next_ply_entry(m, gives_check);
        self.set_stopped(false);
        let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(None, MIN_SCORE, MAX_SCORE, 1, &mut pv, false, NO_MOVE)) else {
            self.board.undo_move(m, previous_piece, removed_piece_id);
            return (qs_score, false);
        };

        let score = -result;
        let has_captures = pv.moves().iter().any(|m| m.is_capture());
        self.board.undo_move(m, previous_piece, removed_piece_id);
        
        if is_mate_or_mated_score(score) {
            (score, true)
        } else {
            (qs_score, has_captures)
        }
    }

    pub fn find_best_move_with_full_strength(&mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move]) -> (Move, PrincipalVariation) {
        self.ctx.set_root_move_randomization(false);
        self.is_strength_limited = false;
        self.find_best_move(rx, skipped_moves)
    }

    fn find_best_move(&mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move]) -> (Move, PrincipalVariation) {
        self.reset();

        self.board.pos_history.mark_root(self.board.halfmove_clock());

        self.last_log_time = Instant::now();

        self.cancel_possible = false;
        self.node_count.store(0, Ordering::Relaxed);
        self.tb_hits.store(0, Ordering::Relaxed);

        self.next_check_node_count = self.limits.node_limit().min(1000);

        let active_player = self.board.active_player();

        let (tt_move, _, _) = self.get_tt_move(0);
        self.ctx.prepare_moves(active_player, tt_move, EMPTY_HISTORY);
        self.ctx.reset_root_moves();

        self.set_stopped(false);

        let mut analysis_result = AnalysisResult::new();

        let mut skipped_moves = Vec::from(skipped_moves);

        self.is_tb_root = false;
        // Probe tablebases
        let tb_result = if self.tb_probe_root {
            if let Some((tb_result, mut tb_skip_moves)) = self.board.probe_root() {
                self.local_tb_hits += 1;
                self.is_tb_root = true;
                skipped_moves.append(&mut tb_skip_moves);
                Some(tb_result)
            } else {
                None
            }
        } else {
            None
        };

        self.player_pov = self.board.active_player();
        self.threads.start_search(&self.board, &skipped_moves, self.multi_pv_count, self.tb_probe_depth, self.is_tb_root);

        let mut multi_pv_state = vec![self.board.clock_scaled_eval(); self.multi_pv_count];

        let mut pv = PrincipalVariation::default();

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in 1..=self.limits.depth_limit() {
            self.max_reached_depth = 0;
            self.current_depth = depth;
            let iteration_start_time = Instant::now();

            let mut iteration_cancelled = false;

            let mut local_skipped_moves = skipped_moves.clone();
            for multi_pv_num in 1..=self.multi_pv_count {
                let score = multi_pv_state[multi_pv_num - 1];

                pv.clear();
                let (cancelled, best_move, pv_uci_moves) =
                    self.root_search(rx, &local_skipped_moves, score, depth, &mut pv);

                if cancelled {
                    iteration_cancelled = true;
                }

                if best_move == NO_MOVE {
                    break;
                }

                if best_move != NO_MOVE {
                    analysis_result.update_result(depth, self.max_reached_depth as i32, best_move, pv_uci_moves, pv.clone());

                    let now = Instant::now();
                    let iteration_duration = now.duration_since(iteration_start_time);
                    if !self.pondering
                        && self.cancel_possible
                        && !self.time_mgr.is_time_for_another_iteration(now, iteration_duration)
                        && !self.time_mgr.try_extend_timelimit()
                    {
                        iteration_cancelled = true;
                    }

                    if let Some(mate_distance) = mate_in(best_move.score()) {
                        if mate_distance.abs() <= self.limits.mate_limit() {
                            iteration_cancelled = true;
                        }
                    }

                    local_skipped_moves.push(best_move.without_score());
                    multi_pv_state[multi_pv_num - 1] = best_move.score();
                }

                if iteration_cancelled {
                    break;
                }

                if depth == 1 && multi_pv_num == 1 && self.ctx.root_move_count() == 1 {
                    self.time_mgr.reduce_timelimit();
                }
            }

            analysis_result.finish_iteration();

            if self.log(Info) {
                analysis_result
                    .print(self.board.halfmove_clock(), tb_result, self.multi_pv_count, self.get_base_stats(self.time_mgr.search_duration(Instant::now())))
            }

            if iteration_cancelled {
                break;
            }
        }

        if let Some(r) = rx {
            while (self.limits.is_infinite() || self.pondering) && !self.is_stopped() {
                self.check_messages(r, true);
            }
        }

        self.set_stopped(true);

        self.threads.wait_for_completion();

        (analysis_result.get_best_move(), analysis_result.get_best_pv())
    }

    fn root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], score: i16, mut depth: i32, pv: &mut PrincipalVariation) -> (bool, Move, Option<String>) {

        let aspiration_window_size = calc_aspiration_window(0, 0, score, score);
        let mut alpha = if depth > 7 { score.saturating_sub(aspiration_window_size) } else { MIN_SCORE };
        let mut beta = if depth > 7 { score.saturating_add(aspiration_window_size) } else { MAX_SCORE };

        self.ctx.set_eval(if self.board.is_in_check(self.board.active_player()) { MIN_SCORE } else { self.board.clock_scaled_eval() });

        let original_depth = depth;
        let mut attempt = 0;
        let initial_score = score;
        let mut step = 0;

        let mut fail_high_move = NO_MOVE;
        let mut fail_high_pv_str = None;
        let mut fail_high_pv: Option<PrincipalVariation> = None;
        loop {
            pv.clear();

            let (cancelled, best_move, current_pv) =
                self.bounded_root_search(rx, skipped_moves, alpha, beta, depth, pv);

            // Bulk update of global node count
            if self.local_node_count > 0 {
                self.node_count.fetch_add(self.local_node_count, Ordering::Relaxed);
                self.local_node_count = 0;
            }
            if self.local_tb_hits > 0 {
                self.tb_hits.fetch_add(self.local_tb_hits, Ordering::Relaxed);
                self.local_tb_hits = 0;
            }

            if cancelled || best_move == NO_MOVE {
                if let Some(fail_high_pv) = fail_high_pv.take() {
                    pv.clone_from(&fail_high_pv);
                }
                return (cancelled, fail_high_move, fail_high_pv_str);
            }

            let best_score = best_move.score();
            if best_score <= alpha {
                if !self.is_helper_thread {
                    fail_high_move = NO_MOVE;
                    fail_high_pv_str = None;
                    fail_high_pv = None;
                }
                attempt += 1;
                depth = original_depth;
                beta = (alpha + beta) / 2;
                step = calc_aspiration_window(attempt, step, initial_score, best_score);
                alpha = best_score.saturating_sub(step).clamp(MIN_SCORE, MAX_SCORE);
            } else if best_score >= beta {
                if !self.is_helper_thread {
                    fail_high_move = best_move;
                    fail_high_pv_str = current_pv.clone();
                    fail_high_pv = Some(pv.clone());
                }
                attempt += 1;
                step = calc_aspiration_window(attempt, step, initial_score, best_score);
                beta = best_score.saturating_add(step).clamp(MIN_SCORE, MAX_SCORE);
                if !is_mate_or_mated_score(best_score) {
                    depth = (depth - 1).max(original_depth - 5);
                }
            } else {
                return (false, best_move, current_pv);
            }
        }
    }

    // Root search within the bounds of an aspiration window (alpha...beta)
    fn bounded_root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], mut alpha: i16, beta: i16, depth: i32, pv: &mut PrincipalVariation
    ) -> (bool, Move, Option<String>) {
        let mut move_num = 0;
        let mut a = -beta;
        let mut best_move: Move = NO_MOVE;
        let mut best_score = MIN_SCORE;

        let mut current_pv = None;

        let active_player = self.board.active_player();

        let mut tree_scale = 0;

        self.ctx.reset_root_moves();
        let mut local_pv = PrincipalVariation::default();
        while let Some(m) = self.ctx.next_root_move(&self.hh, &mut self.board) {
            if skipped_moves.contains(&m.without_score()) {
                continue;
            }
            move_num += 1;

            if self.log(Info) && self.local_total_node_count > 2000000 {
                let now = Instant::now();
                if self.time_mgr.search_duration_ms(now) >= 1000 {
                    self.last_log_time = now;
                    let uci_move = UCIMove::from_move(&self.board, m);
                    println!("info currmove {} currmovenumber {}", uci_move, move_num);
                }
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(m);

            let gives_check = self.board.is_in_check(active_player.flip());

            let mut tree_size = self.local_total_node_count;

            // Use principal variation search
            self.ctx.update_next_ply_entry(m, gives_check);

            local_pv.clear();
            let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, a, -alpha, depth - 1, &mut local_pv, false, NO_MOVE)) else {
                self.board.undo_move(m, previous_piece, removed_piece_id);
                return (true, NO_MOVE, None);
            };
            let mut score = -result;

            if score > alpha && a != -beta {
                // Repeat search if it falls outside the search window
                local_pv.clear();
                let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, -beta, -alpha, depth - 1, &mut local_pv, false, NO_MOVE)) else {
                    self.board.undo_move(m, previous_piece, removed_piece_id);
                    return (true, NO_MOVE, None);
                };
                score = -result;
            }

            self.board.undo_move(m, previous_piece, removed_piece_id);

            if score > best_score {
                best_score = score;
                best_move = m.with_score(score);
                self.cancel_possible = true;

                if !self.is_helper_thread {
                    self.time_mgr.update_best_move(best_move, depth);
                    pv.update(best_move, &mut local_pv);
                    current_pv = Some(self.pv_info(&pv.moves()));
                }

                if best_score <= alpha || best_score >= beta {
                    self.ctx.reorder_root_moves(best_move, false);
                    return (false, best_move, current_pv);
                }

                alpha = score;

                a = -(alpha + 1);
            }

            tree_size = self.local_total_node_count - tree_size;
            if move_num == 1 {
                tree_scale = 13.max(64 - tree_size.leading_zeros()) - 13;
            }
            self.ctx.update_root_move(m.with_score((MAX_SCORE as i32).min((tree_size >> tree_scale) as i32) as i16));
        }

        self.ctx.reorder_root_moves(best_move, self.is_helper_thread);

        (false, best_move, current_pv)
    }

    pub fn node_count(&self) -> u64 {
        self.node_count.load(Ordering::Relaxed)
    }

    fn tb_hits(&self) -> u64 {
        self.tb_hits.load(Ordering::Relaxed)
    }

    fn inc_node_count(&mut self) {
        self.local_total_node_count += 1;
        self.local_node_count += 1;
    }
    
    // Recursively calls itself with alternating player colors to
    // find the best possible move in response to the current board position.
    fn rec_find_best_move(
        &mut self, rx: Option<&Receiver<Message>>, mut alpha: i16, mut beta: i16, mut depth: i32,
        pv: &mut PrincipalVariation, in_se_search: bool, se_move: Move
    ) -> Option<i16> {
        let in_check = self.ctx.in_check();
        let active_player = self.board.active_player();

        // Quiescence search
        if depth <= 0 || self.ctx.max_search_depth_reached() {
            return self.quiescence_search(active_player, alpha, beta, in_check);
        }

        self.max_reached_depth = self.ctx.ply().max(self.max_reached_depth);
        self.inc_node_count();

        if let Some(rx) = rx {
            self.check_search_limits(Some(rx))
        }

        if self.is_stopped() {
            return None;
        }

        if self.local_node_count > 1000 {
            // Bulk update of global node count
            self.node_count.fetch_add(self.local_node_count, Ordering::Relaxed);
            self.local_node_count = 0;
        }
        
        if let Some(score) = self.check_draw(active_player, in_check) {
            return Some(score);
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        if alpha < 0 && self.board.has_upcoming_repetition() {
            alpha = self.effective_draw_score();
            if alpha >= beta {
                return Some(alpha);
            }
        }

        // Mate distance pruning
        let mut best_possible_score = MATE_SCORE - (self.ctx.ply() as i16 + 1);
        let mut worst_possible_score = MATED_SCORE + (self.ctx.ply() as i16);

        alpha = alpha.max(worst_possible_score);
        beta = beta.min(best_possible_score);
        if alpha >= beta {
            return Some(alpha);
        }

        if in_check && se_move == NO_MOVE {
            // Extend search when in check
            depth = (depth + 1).max(1);
        }

        let hash = self.board.get_hash();

        let mut tt_move = NO_MOVE;
        let mut tt_score = 0;


        let mut check_se = false;

        let mut best_score = worst_possible_score;
        let mut best_move = NO_MOVE;

        let move_history = self.ctx.move_history();

        let mut tt_score_type = ScoreType::Exact;
        let mut tt_depth = 0;
        if se_move == NO_MOVE {
            // Check transposition table
            let mut is_tt_hit = false;
            if let Some((tt_entry, matching_clock)) = self.tt.get_entry(hash, self.board.halfmove_clock()) {
                tt_move = get_tt_move(tt_entry, self.ctx.ply());
                tt_score = tt_move.score();

                let is_tb_move = if self.is_tb_move(tt_move) {
                    tt_move = NO_MOVE;
                    true
                } else if is_valid_move(&self.board, active_player, tt_move) {
                    false
                } else {
                    tt_move = NO_MOVE;
                    false
                };

                if tt_move != NO_MOVE || is_tb_move {
                    is_tt_hit = true;
                    tt_depth = get_depth(tt_entry);
                    tt_score_type = get_score_type(tt_entry);

                    let tt_score_is_upper_bound = matches!(tt_score_type, ScoreType::UpperBound);

                    if matching_clock || is_tb_move || is_mate_or_mated_score(tt_score) {
                        if tt_depth >= depth && match tt_score_type {
                            ScoreType::Exact => !is_pv || depth <= 0,
                            ScoreType::UpperBound => (!is_pv || depth <= 0) && tt_score <= alpha,
                            ScoreType::LowerBound => (!is_pv || depth <= 0) && tt_score >= beta
                        } {
                            if !tt_score_is_upper_bound && tt_move != NO_MOVE && !tt_move.is_capture() {
                                self.hh.update(self.ctx.ply(), active_player, move_history, tt_move, true);
                            }
                            return Some(tt_score);
                        }
                    } else if tt_depth >= depth && (!is_pv || depth <= 0) {
                        match tt_score_type {
                            ScoreType::Exact => {
                                if tt_score <= alpha {
                                    return Some(tt_score);
                                }

                                let adj_tt_score = clock_scaled_eval(self.board.halfmove_clock(), tt_score);
                                if adj_tt_score >= beta {
                                    return Some(adj_tt_score);
                                }
                            },
                            ScoreType::UpperBound => {
                                if tt_score <= alpha {
                                    return Some(tt_score);
                                }
                            },
                            ScoreType::LowerBound => {
                                let adj_tt_score = clock_scaled_eval(self.board.halfmove_clock(), tt_score);
                                if adj_tt_score >= beta {
                                    return Some(adj_tt_score);
                                }
                            }
                        }
                    }

                    check_se = !in_se_search
                        && tt_move != NO_MOVE
                        && !in_check
                        && !tt_score_is_upper_bound
                        && depth >= 6
                        && !is_mate_or_mated_score(tt_score)
                        && tt_depth >= depth - 3;
                }
            }

            // Probe tablebases
            if !self.is_tb_root && depth.max(0) >= self.tb_probe_depth {
                if let Some(tb_result) = self.board.probe_wdl() {
                    self.local_tb_hits += 1;

                    match tb_result {
                        TBResult::Draw => {
                            self.tt.write_entry(hash, self.ctx.ply(), MAX_DEPTH as i32, self.tb_move(), 0, ScoreType::Exact, 100);
                            return Some(0);
                        },
                        TBResult::Win => {
                            worst_possible_score = 400 - (self.ctx.ply() as i16 / 4);
                            if !is_pv && worst_possible_score >= beta {
                                return Some(worst_possible_score);
                            }
                        },
                        TBResult::Loss => {
                            best_possible_score = -400 + (self.ctx.ply() as i16 / 4);
                            if !is_pv && best_possible_score <= alpha {
                                return Some(best_possible_score);
                            }
                        },
                        TBResult::CursedWin => {
                            self.tt.write_entry(hash, self.ctx.ply(), MAX_DEPTH as i32, self.tb_move(), 1, ScoreType::Exact, 100);
                            return Some(0);
                        },
                        TBResult::BlessedLoss => {
                            self.tt.write_entry(hash, self.ctx.ply(), MAX_DEPTH as i32, self.tb_move(), -1, ScoreType::Exact, 100);
                            return Some(0);
                        }
                    }
                }
            }

            if !is_tt_hit && depth > 3 {
                // Reduce nodes without move from transposition table
                depth -= 1;
            }
        }

        self.ctx.set_eval(if in_check { MIN_SCORE } else {
            let corr_eval = self.hh.corr_eval(active_player, self.board.piece_hashes(), self.ctx.move_history_hash());
            self.tt.get_or_calc_eval(self.board.get_hash(), self.board.halfmove_clock(), || self.board.eval(), corr_eval)
        });
        let improving = self.ctx.is_improving();

        let mut ref_score = self.ctx.eval();
        if tt_move != NO_MOVE {
            ref_score = match tt_score_type {
                ScoreType::Exact => tt_score,
                ScoreType::UpperBound => ref_score.min(tt_score),
                ScoreType::LowerBound => ref_score.max(tt_score)
            }
        }
        ref_score = clamp_score(ref_score, worst_possible_score, best_possible_score);

        let unreduced_depth = depth;
        if !is_pv && !in_check {
            if is(self.params.rfp_enabled()) && depth <= 8 && !is_mate_or_mated_score(beta) && !is_mate_or_mated_score(ref_score) {
                // Reverse futility pruning
                let margin = self.params.rfp_margin_multiplier() * (depth as i16 - i16::from(improving));
                if ref_score - margin >= beta {
                    return Some(sanitize_score(beta + (ref_score - beta) / 2));
                }
            }

            if is(self.params.razoring_enabled()) && !improving && depth <= 4 && !is_mate_or_mated_score(alpha) && !is_mate_or_mated_score(ref_score) && ref_score + (1 << (depth - 1)) * self.params.razor_margin_multiplier() <= alpha {
                // Razoring
                let result = self.quiescence_search(active_player, alpha, beta, in_check)?;
                let score = clamp_score(result, worst_possible_score, best_possible_score);
                if score <= alpha {
                    return Some(score);
                }
            }

            if is(self.params.nmp_enabled()) && se_move == NO_MOVE && ref_score >= beta {
                let reduced_depth = depth - self.null_move_reduction(depth);
                if !(matches!(tt_score_type, ScoreType::UpperBound) && tt_depth >= reduced_depth) && self.board.has_non_pawns(active_player) {
                    // Null move pruning
                    self.board.perform_null_move();
                    self.tt.prefetch(self.board.get_hash());

                    self.ctx.update_next_ply_entry(NO_MOVE, false);

                    let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, -beta, -beta + 1, reduced_depth, &mut PrincipalVariation::default(), in_se_search, NO_MOVE)) else {
                        self.board.undo_null_move();
                        return None;
                    };

                    self.board.undo_null_move();
                    let score = clamp_score(-result, worst_possible_score, best_possible_score);
                    if score >= beta {
                        if is_mate_or_mated_score(score) {
                            return Some(beta);
                        } else if reduced_depth >= 12 {
                            depth = reduced_depth; // verify null move result with reduced regular search
                        } else {
                            return Some(score);
                        }
                    }
                }
            }
            
            // ProbCut
            let prob_cut_beta = beta + self.params.prob_cut_margin();
            let prob_cut_depth = depth - self.params.prob_cut_depth() as i32;
            if is(self.params.prob_cut_enabled()) && se_move == NO_MOVE && tt_move != NO_MOVE && !matches!(tt_score_type, ScoreType::UpperBound) && tt_score >= prob_cut_beta && prob_cut_depth > 0 && is_eval_score(beta) {
                let (previous_piece, removed_piece_id) = self.board.perform_move(tt_move);
                self.tt.prefetch(self.board.get_hash());
                if !self.board.is_left_in_check(active_player, false, tt_move) {
                    let gives_check = self.board.is_in_check(active_player.flip());
                    self.ctx.update_next_ply_entry(tt_move, gives_check);
            
                    let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, -prob_cut_beta, -prob_cut_beta + 1, prob_cut_depth, &mut PrincipalVariation::default(), in_se_search, NO_MOVE)) else {
                        self.board.undo_move(tt_move, previous_piece, removed_piece_id);
                        return None;
                    };
                    let score = clamp_score(-result, worst_possible_score, best_possible_score);
            
                    if score >= prob_cut_beta {
                        self.board.undo_move(tt_move, previous_piece, removed_piece_id);
                        return Some(score);
                    }
                }
                self.board.undo_move(tt_move, previous_piece, removed_piece_id);
            }
        }
        
        let mut evaluated_move_count = 0;
        let mut quiet_move_count = 0;

        let allow_lmr = depth > 2;

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        if is(self.params.fp_enabled()) && !is_pv && !improving && depth <= 6 && !in_check {
            let margin = (self.params.fp_margin_multiplier() << depth) + self.params.fp_base_margin();
            allow_futile_move_pruning = ref_score + margin <= alpha;
        }

        self.ctx.prepare_moves(active_player, tt_move, move_history);

        let occupied_bb = self.board.occupancy_bb();

        let mut is_singular = false;

        let has_non_pawns = self.board.has_non_pawns(active_player);
        let allow_lmp = !is_pv && !in_check && has_non_pawns && depth <= self.params.lmp_max_depth() as i32;

        self.hh.clear_killers(self.ctx.ply() + 1);
        self.ctx.clear_cutoff_count();

        let mut score_type = ScoreType::UpperBound;
        let mut a = -beta;
        
        while let Some(curr_move) = self.ctx.next_move(&self.hh, &self.board) {
            if se_move == curr_move {
                continue;
            }

            if allow_lmp && !curr_move.is_capture()
                && !is_killer(curr_move)
                && !curr_move.is_queen_promotion() && quiet_move_count > self.params.lmp(improving, depth) as i16 {
                continue;
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(curr_move);

            self.tt.prefetch(self.board.get_hash());

            let gives_check = self.board.is_in_check(active_player.flip());

            // Check, if the transposition table move is singular and should be extended
            let mut se_extension = 0;
            if is(self.params.se_enabled()) && check_se && !gives_check && curr_move == tt_move {
                let se_beta = sanitize_score(tt_score - depth as i16);
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                let result = self.rec_find_best_move(rx, se_beta - 1, se_beta, depth / 2, &mut PrincipalVariation::default(), true, curr_move).unwrap_or(MATED_SCORE);

                if result < se_beta {
                    is_singular = true;
                    if !is_pv && result + self.params.se_double_ext_margin() < se_beta && self.ctx.double_extensions() < self.params.se_double_ext_limit() {
                        self.ctx.inc_double_extensions();
                        se_extension = 2;
                    } else {
                        se_extension = 1;
                    }
                } else if se_beta >= beta {
                    // Multi-Cut Pruning
                    if !curr_move.is_capture() {
                        self.hh.update(self.ctx.ply(), active_player, move_history, curr_move, curr_move.score() > QUIET_BASE_SCORE);
                    }

                    return Some(clamp_score(se_beta, worst_possible_score, best_possible_score));
                } else if tt_score <= alpha || tt_score >= beta {
                    se_extension = -1;
                }

                self.ctx.prepare_moves(active_player, tt_move, move_history);
                self.ctx.next_move(&self.hh, &self.board);

                self.board.perform_move(curr_move);
            };

            let start = curr_move.start();
            let end = curr_move.end();

            let mut skip = self.board.is_left_in_check(active_player, in_check, curr_move); // skip if move would put own king in check

            let mut reductions = 0;
            if !skip && evaluated_move_count > 0 {
                if !is_pv && self.ctx.next_ply_cutoff_count() > 3 {
                    reductions += 1;
                }

                let target_piece_id = curr_move.move_type().piece_id();

                if curr_move.is_capture() {
                    if !in_check && !gives_check && self.ctx.is_bad_capture_move() && !improving {
                        reductions += 2;
                    }

                } else {
                    if allow_lmr && quiet_move_count > LMR_THRESHOLD && !curr_move.is_queen_promotion()  {
                        reductions += self.derived_params.lmr(lmr_idx(quiet_move_count)) as i32 + i32::from(!is_pv);

                        if is_singular || tt_move.is_capture() || tt_move.is_queen_promotion() {
                            reductions += 1;
                        }

                        let history_diff = (curr_move.score() - QUIET_BASE_SCORE) / -MIN_HISTORY_SCORE;
                        if !is_pv && !improving && history_diff < 0  {
                            reductions -= history_diff as i32;
                        } else if is_pv && improving && history_diff > 0 && reductions > 0 {
                            reductions -= 1;
                        }
                        
                        if curr_move.score() < QUIET_BASE_SCORE && self.board.has_negative_see(active_player.flip(), start as usize, end as usize, target_piece_id, EMPTY, occupied_bb) {
                            reductions += 1;
                        } 
                        
                        if curr_move.score() >= QUIET_BASE_SCORE && target_piece_id == P && is_passed_pawn(end as usize, active_player, self.board.get_bitboard(active_player.flip().piece(P))) {
                            reductions -= 1;
                        }

                    } else if allow_futile_move_pruning && !gives_check && !curr_move.is_queen_promotion() {
                        // Reduce futile move
                        reductions += FUTILE_MOVE_REDUCTIONS;
                        if target_piece_id == P && is_passed_pawn(end as usize, active_player, self.board.get_bitboard(active_player.flip().piece(P))) {
                            reductions -= 1;
                        }

                    } else if curr_move.score() <= NEGATIVE_HISTORY_SCORE {
                        reductions += NEG_HISTORY_REDUCTIONS;

                    } else if curr_move.score() < QUIET_BASE_SCORE
                        && self.board.has_negative_see(active_player.flip(), start as usize, end as usize, target_piece_id, EMPTY, occupied_bb)
                    {
                        // Reduce search depth for moves with negative SEE score
                        reductions += NEG_SEE_REDUCTIONS;
                    }

                    quiet_move_count += 1;

                    if allow_futile_move_pruning
                        && has_non_pawns
                        && !gives_check
                        && !curr_move.is_queen_promotion()
                        && reductions >= (depth - 1)
                    {
                        // Prune futile move
                        skip = true;
                    } else if reductions > 0 && is_killer(curr_move) {
                        // Reduce killer moves less
                        reductions -= 1;
                    }
                }
            }

            if skip {
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
            } else {
                evaluated_move_count += 1;
                if curr_move == tt_move && curr_move.is_quiet() {
                    quiet_move_count += 1;
                }

                let mut local_pv = PrincipalVariation::default();

                self.ctx.update_next_ply_entry(curr_move, gives_check);

                let reduced_depth = depth + se_extension - reductions - 1;
                let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, a, -alpha, reduced_depth, &mut local_pv, in_se_search, NO_MOVE)) else {
                    self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                    return None;
                };

                let mut score = clamp_score(-result, worst_possible_score, best_possible_score);
                if score > alpha {
                    let full_depth = unreduced_depth + se_extension - 1;
                    if score > alpha && (a != -beta || full_depth > reduced_depth) {
                        // Repeat search without reduction and with full window
                        local_pv.clear();
                        let Some(result) = next_ply!(self.ctx, self.rec_find_best_move(rx, -beta, -alpha, full_depth, &mut local_pv, in_se_search, NO_MOVE)) else {
                            self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                            return None;
                        };
                        score = clamp_score(-result, worst_possible_score, best_possible_score);
                    }
                }

                self.board.undo_move(curr_move, previous_piece, removed_piece_id);

                if score > best_score {
                    best_score = score;
                    best_move = curr_move;

                    // Alpha-beta pruning
                    if best_score >= beta {
                        depth = unreduced_depth;
                        if se_move == NO_MOVE {
                            self.tt.write_entry(hash, self.ctx.ply(), depth, best_move, best_score, ScoreType::LowerBound, self.board.halfmove_clock());
                            if !(in_check || best_move.is_capture() || is_mate_or_mated_score(best_score) || best_score <= self.ctx.eval()) {
                                self.hh.update_corr_histories(active_player, depth, self.board.piece_hashes(), self.ctx.move_history_hash(), best_score - self.ctx.eval());
                            }
                        }

                        if !curr_move.is_capture() {
                            self.hh.update(self.ctx.ply(), active_player, move_history, best_move, curr_move.score() > QUIET_BASE_SCORE)
                        }

                        if is_pv && !self.is_helper_thread {
                            pv.update(best_move, &mut local_pv);
                        }
                        
                        self.ctx.inc_cutoff_count();

                        return Some(best_score);
                    }

                    if best_score > alpha {
                        depth = unreduced_depth;
                        alpha = best_score;
                        score_type = ScoreType::Exact;
                        if is_pv && !self.is_helper_thread {
                            pv.update(best_move, &mut local_pv);
                        }
                    }
                } else if !curr_move.is_capture() {
                    self.hh.update_played_moves(active_player, move_history, curr_move);
                }

                a = -(alpha + 1);
            }
        }

        if evaluated_move_count == 0 {
            return if se_move != NO_MOVE {
                Some(alpha)
            } else if in_check {
                Some(MATED_SCORE + self.ctx.ply() as i16) // Check mate
            } else {
                Some(self.effective_draw_score())
            }
        }

        best_score = best_score.clamp(worst_possible_score, best_possible_score);

        if se_move == NO_MOVE {
            self.tt.write_entry(hash, self.ctx.ply(), depth, best_move, best_score, score_type, self.board.halfmove_clock());

            if !(in_check || best_move.is_capture() || is_mate_or_mated_score(best_score) || matches!(score_type, ScoreType::UpperBound) && best_score >= self.ctx.eval()) {
                self.hh.update_corr_histories(active_player, depth, self.board.piece_hashes(), self.ctx.move_history_hash(), best_score - self.ctx.eval());
            }
        }

        Some(best_score)
    }

    fn check_search_limits(&mut self, rx: Option<&Receiver<Message>>) {
        if self.local_total_node_count >= self.next_check_node_count && self.current_depth > 1 {
            self.next_check_node_count = if self.limits.node_limit() != u64::MAX {
                self.limits.node_limit()
            } else {
                self.local_total_node_count + 1000
            };

            if let Some(rx) = rx {
                self.check_messages(rx, false);
            }

            let now = Instant::now();
            if !self.pondering
                && self.cancel_possible
                && (self.local_total_node_count >= self.limits.node_limit() || self.time_mgr.is_timelimit_exceeded(now))
            {
                // Cancel search if the node or time limit has been reached, but first check
                // whether the search time should be extended
                if !self.is_stopped() && !self.time_mgr.try_extend_timelimit() {
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

    pub fn quiescence_search(&mut self, active_player: Color, alpha: i16, beta: i16, in_check: bool) -> Option<i16> {
        if self.is_helper_thread {
            self.qs::<true>(active_player, alpha, beta, in_check).map(sanitize_score)
        } else {
            self.qs::<false>(active_player, alpha, beta, in_check).map(sanitize_score)
        }
    }

    fn qs<const HELPER_THREAD: bool>(&mut self, active_player: Color, mut alpha: i16, beta: i16, in_check: bool) -> Option<i16> {
        self.max_reached_depth = self.ctx.ply().max(self.max_reached_depth);
        self.inc_node_count();

        if !HELPER_THREAD {
            self.check_search_limits(None)
        }

        if self.is_stopped() {
            return None;
        }

        if let Some(score) = self.check_draw(active_player, in_check) {
            return Some(score);
        }

        if alpha < 0 && self.board.has_upcoming_repetition() {
            alpha = self.effective_draw_score();
            if alpha >= beta {
                return Some(alpha);
            }
        }

        let mut position_score = if in_check { MATED_SCORE + self.ctx.ply() as i16 } else {
            let corr_eval = self.hh.corr_eval(active_player, self.board.piece_hashes(), self.ctx.move_history_hash());
            self.tt.get_or_calc_eval(self.board.get_hash(), self.board.halfmove_clock(), || self.board.eval(), corr_eval)
        };

        if self.ctx.max_qs_depth_reached() || (self.is_strength_limited && self.local_total_node_count >= self.limits.node_limit()) {
            return Some(position_score);
        }

        let opp_player = active_player.flip();
        let mut search_evasions = in_check;

        let (tt_move, score_type, is_tb_entry) = self.get_tt_move(self.ctx.ply());
        if tt_move != NO_MOVE || is_tb_entry {
            let tt_score = tt_move.score();
            match score_type {
                ScoreType::Exact => {
                    return Some(tt_score);
                },
                ScoreType::UpperBound => {
                    if tt_score <= alpha {
                        return Some(tt_score);
                    }
                    if !in_check {
                        position_score = position_score.min(tt_score);
                    }
                },
                ScoreType::LowerBound => {
                    if tt_score >= beta {
                        return Some(tt_score);
                    }
                    alpha = alpha.max(tt_score);
                    if !in_check {
                        position_score = position_score.max(tt_score);
                    }
                }
            }
        }
        
        if position_score > alpha {
            if position_score >= beta {
                return Some(position_score);
            }
            alpha = position_score;
        }

        let mut best_score = position_score;

        if tt_move != NO_MOVE {
            if let Some(score) = self.check_qs_move::<HELPER_THREAD>(active_player, best_score, alpha, beta, in_check, opp_player, tt_move) {
                if score > best_score {
                    search_evasions = false;
                    best_score = score;
                    alpha = alpha.max(score);
                    if alpha >= beta {
                        return Some(alpha);
                    }
                }
            }
        }

        self.ctx.prepare_moves(active_player, NO_MOVE, self.ctx.move_history());
        if search_evasions {
            while let Some(m) = self.ctx.next_move(&self.hh, &self.board) {
                if m == tt_move {
                    continue;
                }
                if let Some(score) = self.check_qs_move::<HELPER_THREAD>(active_player, best_score, alpha, beta,  in_check, opp_player, m) {
                    if score > best_score {
                        best_score = score;
                        alpha = alpha.max(score);
                        if alpha >= beta {
                            self.tt.write_entry(self.board.get_hash(), self.ctx.ply(), 0, m, alpha, ScoreType::LowerBound, self.board.halfmove_clock());
                            return Some(alpha);
                        }
                        break;
                    }
                }
            }
        }

        self.ctx.generate_qs_captures(&self.board);
        let mut best_move = NO_MOVE;
        while let Some(m) = self.ctx.next_good_capture_move(&self.board) {
            if m == tt_move {
                continue;
            }
            if let Some(score) = self.check_qs_move::<HELPER_THREAD>(active_player, best_score, alpha, beta, in_check, opp_player, m) {
                if score > best_score {
                    best_score = score;
                    best_move = m;
                    alpha = alpha.max(score);
                    if alpha >= beta {
                        self.tt.write_entry(self.board.get_hash(), self.ctx.ply(), 0, m, alpha, ScoreType::LowerBound, self.board.halfmove_clock());
                        return Some(alpha);
                    }
                }
            }
        }
        if best_move != NO_MOVE {
            let score_type = if best_score == alpha { ScoreType::Exact } else { ScoreType::UpperBound };
            self.tt.write_entry(self.board.get_hash(), self.ctx.ply(), 0, best_move, alpha, score_type, self.board.halfmove_clock());
        }

        Some(best_score)
    }

    fn check_qs_move<const HELPER_THREAD: bool>(&mut self, active_player: Color, best_score: i16, alpha: i16, beta: i16, in_check: bool, opp_player: Color, m: Move) -> Option<i16> {
        let (previous_piece, captured_piece_id) = self.board.perform_move(m);
        self.tt.prefetch(self.board.get_hash());
        if self.board.is_left_in_check(active_player, in_check, m) {
            self.board.undo_move(m, previous_piece, captured_piece_id);
            return None;
        }

        let gives_check = self.board.is_in_check(opp_player);
        self.ctx.update_next_ply_entry(m, gives_check);
        let Some(result) = next_ply!(self.ctx, self.qs::<HELPER_THREAD>(opp_player, -beta, -alpha, gives_check)) else {
            self.board.undo_move(m, previous_piece, captured_piece_id);
            return None;
        };
        let score = -result;
        self.board.undo_move(m, previous_piece, captured_piece_id);

        if score > best_score {
            Some(score)
        } else {
            None
        }
    }

    fn check_draw(&mut self, active_player: Color, in_check: bool) -> Option<i16> {
        if self.board.is_insufficient_material_draw() || self.board.is_repetition_draw() {
            return Some(self.effective_draw_score());
        }

        if self.board.is_fifty_move_draw() {
            if in_check && !self.ctx.has_any_legal_move(active_player, &self.hh, &mut self.board) {
                return Some(MATED_SCORE + self.ctx.ply() as i16); // Check mate
            }
            return Some(self.effective_draw_score());
        }

        None
    }

    fn get_tt_move(&self, ply: usize) -> (Move, ScoreType, bool) {
        if let Some((entry, _)) = self.tt.get_entry(self.board.get_hash(), self.board.halfmove_clock()) {
            let tt_move = get_tt_move(entry, ply);
            let active_player = self.board.active_player();
            if self.is_tb_move(tt_move) {
                return (NO_MOVE, ScoreType::Exact, true);
            } else if is_valid_move(&self.board, active_player, tt_move) {
                return (tt_move, get_score_type(entry), false);
            }
        }

        (NO_MOVE, ScoreType::LowerBound, false)
    }

    pub fn determine_skipped_moves(&mut self, search_moves: Vec<String>) -> Vec<Move> {
        let mut search_moves_set = MoveSet::default();
        for uci_move in search_moves.iter() {
            if let Some(m) = UCIMove::from_uci(uci_move) {
                search_moves_set.insert(m.to_move(&self.board));
            }
        }

        self.ctx.reset_root_moves();
        let mut skipped_moves = Vec::new();
        while let Some(m) = self.ctx.next_root_move(&self.hh, &mut self.board) {
            if !search_moves_set.contains(&m) {
                skipped_moves.push(m.without_score());
            }
        }

        skipped_moves
    }

    fn get_base_stats(&self, duration: Duration) -> String {
        let node_count = self.node_count();
        let tb_hits = self.tb_hits();
        let duration_micros = duration.as_micros();
        let nodes_per_second = if duration_micros > 0 { node_count as u128 * 1_000_000 / duration_micros } else { 0 };

        if nodes_per_second > 0 {
            format!(
                " nodes {} nps {} tbhits {} hashfull {} time {}",
                node_count,
                nodes_per_second,
                tb_hits,
                self.tt.hash_full(),
                duration_micros / 1000
            )
        } else {
            format!(" nodes {} time {}", node_count, duration_micros / 1000)
        }
    }

    fn pv_info(&mut self, pv: &[Move]) -> String {
        if let Some((&m, rest_pv)) = pv.split_first() {
            let uci_move = UCIMove::from_move(&self.board, m);
            let (previous_piece, move_state) = self.board.perform_move(m);

            let followup_moves = self.pv_info(rest_pv);

            self.board.undo_move(m, previous_piece, move_state);
            format!("{} {}", uci_move, followup_moves)
        } else {
            String::new()
        }
    }

    fn log(&self, log_level: LogLevel) -> bool {
        self.log_level <= log_level
    }

    fn is_stopped(&self) -> bool {
        self.is_stopped.load(Ordering::Relaxed)
    }

    pub fn set_stopped(&mut self, value: bool) {
        self.is_stopped.store(value, Ordering::Relaxed);
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
        self.limits.set_node_limit(node_limit);
    }

    pub fn set_tb_probe_root(&mut self, value: bool) {
        self.tb_probe_root = value;
    }

    fn effective_draw_score(&self) -> i16 {
        -1 + (self.local_total_node_count & 2) as i16
    }

    fn tb_move(&self) -> Move {
        let active_player = self.board.active_player();
        Move::new(MoveType::TableBaseMarker, self.board.king_pos(active_player), self.board.king_pos(active_player.flip()))
    }

    fn is_tb_move(&self, m: Move) -> bool {
        if !matches!(m.move_type(), MoveType::TableBaseMarker) {
            return false;
        }

        let active_player = self.board.active_player();
        m.start() == self.board.king_pos(active_player) && m.end() == self.board.king_pos(active_player.flip())
    }

    fn null_move_reduction(&self, depth: i32) -> i32 {
        (self.params.nmp_base() as i32 + (depth * 256 * 256) / self.params.nmp_divider() as i32) / 256
    }
    
    pub fn set_expected_best_move(&mut self, m: Move) {
        self.time_mgr.set_expected_best_move(m);
    }
}

fn is(value: i16) -> bool {
    value != 0
}

fn clamp_score(score: i16, worst_possible_score: i16, best_possible_score: i16) -> i16 {
    if worst_possible_score < best_possible_score {
        score.clamp(worst_possible_score, best_possible_score)
    } else {
        score.clamp(best_possible_score, worst_possible_score)
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
        multi_pv_count: usize,
        tb_probe_depth: i32,
        is_tb_root: bool,
    },
    Reset,
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
        &mut self, target_count: usize, node_count: &Arc<AtomicU64>, tb_hits: &Arc<AtomicU64>, tt: &Arc<TranspositionTable>, board: &Board,
        is_stopped: &Arc<AtomicBool>, params: SingleParams,
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
            let tb_hits = tb_hits.clone();
            let tt = tt.clone();
            let board = board.clone();
            let is_stopped = is_stopped.clone();

            let handle = thread::spawn(move || {
                let limits = SearchLimits::default();
                let mut sub_search = Search::new(is_stopped, node_count, tb_hits, LogLevel::Error, limits, tt, board, true);
                sub_search.set_params(params);

                HelperThread::run(to_rx, from_tx, sub_search);
            });

            self.threads.push(HelperThread { handle, to_tx, from_rx });
        }
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    pub fn start_search(&self, board: &Board, skipped_moves: &[Move], multi_pv_count: usize, tb_probe_depth: i32, is_tb_root: bool) {
        for t in self.threads.iter() {
            t.search(board, skipped_moves, multi_pv_count, tb_probe_depth, is_tb_root);
        }
    }

    pub fn clear_tt(&self) {
        let total_count = self.threads.len() + 1;
        for (i, t) in self.threads.iter().enumerate() {
            t.clear_tt(i + 1, total_count);
        }

        self.wait_for_completion();
    }
    
    pub fn reset(&self) {
        for t in self.threads.iter() {
            t.reset();
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

#[derive(Clone)]
struct AnalysisResult {
    entries: Vec<AnalysisEntry>,
}

impl AnalysisResult {
    fn new() -> Self {
        AnalysisResult { entries: Vec::new() }
    }

    pub fn update_result(
        &mut self, depth: i32, sel_depth: i32, best_move: Move, pv_info: Option<String>, pv: PrincipalVariation,
    ) {
        for entry in self.entries.iter_mut() {
            if entry.best_move == best_move {
                entry.best_move = best_move;
                entry.depth = depth;
                entry.sel_depth = sel_depth;
                entry.pv_info = pv_info;
                entry.pv = pv;
                return;
            }
        }

        self.entries.push(AnalysisEntry { best_move, depth, sel_depth, pv_info, pv })
    }

    pub fn finish_iteration(&mut self) {
        self.entries.sort_by_key(|entry| Reverse((entry.depth, entry.best_move.score())));
    }

    pub fn get_best_move(&self) -> Move {
        if self.entries.is_empty() {
            NO_MOVE
        } else {
            self.entries[0].best_move
        }
    }

    pub fn get_best_pv(&self) -> PrincipalVariation {
        if self.entries.is_empty() {
            PrincipalVariation::default()
        } else {
            self.entries[0].pv.clone()
        }
    }

    pub fn print(&self, halfmove_clock: u8, tb_result: Option<TBResult>, max_moves: usize, base_stats: String) {
        for (i, entry) in self.entries.iter().take(max_moves).enumerate() {
            let score = adjust_score(halfmove_clock, tb_result,entry.best_move.score());
            println!(
                "info depth {} seldepth {} multipv {} score {}{}{}",
                entry.depth,
                entry.sel_depth,
                i + 1,
                get_score_info(score),
                base_stats,
                entry.pv_info.clone().map(|pv| format!(" pv {}", pv)).unwrap_or_default()
            );
        }
    }
}

fn adjust_score(halfmove_clock: u8, tb_result: Option<TBResult>, score: i16) -> i16 {
    if let Some(result) = tb_result {
        let divider = halfmove_clock as i16 + 1;
        match result {
            TBResult::Draw => 0,
            TBResult::CursedWin => score.clamp(0, 100) / divider,
            TBResult::BlessedLoss => score.clamp(-100, 0) / divider,
            TBResult::Win => (MAX_EVAL - halfmove_clock as i16).max(score),
            TBResult::Loss => (MIN_EVAL + halfmove_clock as i16).min(score)
        }
    } else {
        score
    }
}


#[derive(Clone)]
struct AnalysisEntry {
    depth: i32,
    sel_depth: i32,
    best_move: Move,
    pv_info: Option<String>,
    pv: PrincipalVariation,
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
    pub fn search(&self, board: &Board, skipped_moves: &[Move], multi_pv_count: usize, tb_probe_depth: i32, is_tb_root: bool) {
        match self.to_tx.send(ToThreadMessage::Search {
            pos_history: board.pos_history.clone(),
            bitboards: board.bitboards,
            halfmove_count: board.halfmove_count,
            state: board.state,
            castling_rules: board.castling_rules,
            skipped_moves: Vec::from(skipped_moves),
            multi_pv_count,
            tb_probe_depth,
            is_tb_root,
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
    
    pub fn reset(&self) {
        self.to_tx.send(ToThreadMessage::Reset).unwrap();
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
                    multi_pv_count,
                    tb_probe_depth,
                    is_tb_root

                } => {
                    sub_search.reset();
                    sub_search.board.reset(pos_history, bitboards, halfmove_count, state, castling_rules);
                    sub_search.set_tb_probe_depth(tb_probe_depth);
                    sub_search.player_pov = sub_search.board.active_player();
                    sub_search.is_tb_root = is_tb_root;
                    
                    let active_player = sub_search.board.active_player();
                    let (tt_move, _, _) = sub_search.get_tt_move(0);
                    sub_search.ctx.prepare_moves(active_player, tt_move, EMPTY_HISTORY);

                    let mut multi_pv_state = vec![sub_search.board.clock_scaled_eval(); multi_pv_count];

                    let mut found_moves = false;
                    let mut pv = PrincipalVariation::default();
                    for depth in 1..MAX_DEPTH {
                        sub_search.current_depth = depth as i32;
                        let mut local_skipped_moves = skipped_moves.clone();
                        for multi_pv_num in 1..=multi_pv_count {
                            let score= multi_pv_state[multi_pv_num - 1];
                            pv.clear();
                            let (_, best_move, _) = sub_search.root_search(
                                None,
                                &local_skipped_moves,
                                score,
                                depth as i32,
                                &mut pv,
                            );

                            if sub_search.ctx.root_move_count() == 0 {
                                break;
                            } else {
                                found_moves = true;
                            }

                            if sub_search.is_stopped() {
                                break;
                            }

                            multi_pv_state[multi_pv_num - 1] = best_move.score();
                            local_skipped_moves.push(best_move.without_score());
                        }

                        if sub_search.is_stopped() || !found_moves {
                            break;
                        }
                    }

                    tx.send(()).unwrap();
                }

                ToThreadMessage::ClearTT { thread_no, total_threads } => {
                    sub_search.tt.clear(thread_no, total_threads);
                    tx.send(()).unwrap();
                }
                
                ToThreadMessage::Reset => {
                    sub_search.hh.clear();
                    tx.send(()).unwrap();
                }

                ToThreadMessage::Terminate => break,
            }
        }
    }
}

fn get_score_info(score: i16) -> String {
    if !is_mate_or_mated_score(score) {
        return format!("cp {}", score);
    }

    if score < 0 {
        format!("mate {}", (MATED_SCORE - score - 1) / 2)
    } else {
        format!("mate {}", (MATE_SCORE - score + 1) / 2)
    }
}

fn calc_aspiration_window(attempt: usize, prev_step: i16, prev_score: i16, curr_score: i16) -> i16 {
    if is_mate_or_mated_score(curr_score) && is_mate_or_mated_score(prev_score) {
        return MAX_SCORE;
    } else if attempt == 0 {
        return INITIAL_ASPIRATION_WINDOW_SIZE;
    }

    let delta = (curr_score - prev_score).abs() as i32;

    (((delta as f32).max(prev_step as f32) * (4.0f32 / 3.0f32).powi(attempt as i32)) as i32).clamp(MIN_SCORE as i32, MAX_SCORE as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::castling::{CastlingRules, CastlingState};
    use crate::board::Board;
    use crate::colors::{BLACK, WHITE};
    use crate::fen::{create_from_fen, write_fen};
    use crate::init::init;
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

        let m = search(tt.clone(), limits, board.clone());
        assert_ne!(NO_MOVE, m);

        board.perform_move(m);

        let is_check_mate = search(tt, limits, board.clone()) == NO_MOVE && board.is_in_check(WHITE);
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
        let limits = SearchLimits::nodes(10000);
        let mut board = create_from_fen(fen.as_str());

        let m1 = search(tt.clone(), limits, board.clone());
        board.perform_move(m1);

        let m2 = search(tt.clone(), limits, board.clone());
        board.perform_move(m2);

        let m3 = search(tt.clone(), limits, board.clone());
        board.perform_move(m3);

        let is_check_mate = search(tt, limits, board.clone()) == NO_MOVE && board.is_in_check(BLACK);
        assert!(is_check_mate);
    }

    #[test]
    fn encode_tb_move() {
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
        let board = create_from_fen(fen.as_str());

        let search = new_search(tt, limits, board);
        let tb_move = search.tb_move();
        assert!(search.is_tb_move(tb_move));
    }

    fn search(tt: Arc<TranspositionTable>, limits: SearchLimits, board: Board) -> Move {
        let mut search = new_search(tt, limits, board.clone());
        search.update(&board, limits, false);
        let (m, _) = search.find_best_move(None, &[]);
        m
    }

    fn new_search(tt: Arc<TranspositionTable>, limits: SearchLimits, board: Board) -> Search {
        init();

        Search::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            LogLevel::Error,
            limits,
            tt,
            board,
            false,
        )
    }

    fn to_fen(active_player: Color, items: &[i8; 64]) -> String {
        write_fen(&Board::new(items, active_player, CastlingState::default(), None, 0, 1, CastlingRules::default()))
    }
}
