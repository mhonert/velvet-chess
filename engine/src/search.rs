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

use crate::bitboard::{BitBoards, is_passed_pawn};
use crate::board::castling::CastlingRules;
use crate::board::{Board, StateEntry};
use crate::colors::{Color, WHITE};
use crate::engine::{LogLevel, Message};
use crate::history_heuristics::{HistoryHeuristics};
use crate::move_gen::{is_killer, MoveGenerator, NEGATIVE_HISTORY_SCORE, QUIET_BASE_SCORE, is_valid_move, MoveSet};
use crate::moves::{Move, MoveType, NO_MOVE, TTPackedMove, UnpackedMove};
use crate::pieces::{EMPTY, P};
use crate::pos_history::PositionHistory;
use crate::scores::{mate_in, sanitize_score, MATED_SCORE, MATE_SCORE, MAX_SCORE, MIN_SCORE, is_mate_or_mated_score, is_eval_score, MAX_EVAL, MIN_EVAL};
use crate::time_management::{SearchLimits, TimeManager};
use crate::transposition_table::{from_root_relative_score, ScoreType, TranspositionTable, MAX_DEPTH, get_hash_move, to_root_relative_score, get_depth, get_score_type, MAX_GENERATION, update_generation};
use crate::uci_move::UCIMove;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use LogLevel::Info;
use crate::params;
use crate::syzygy::{DEFAULT_TB_PROBE_DEPTH, ProbeTB};
use crate::syzygy::tb::{TBResult};

pub const DEFAULT_SEARCH_THREADS: usize = 1;
pub const MAX_SEARCH_THREADS: usize = 512;

const CANCEL_SEARCH: i16 = i16::MAX - 1;

const LMR_THRESHOLD: i16 = 1;

const FUTILE_MOVE_REDUCTIONS: i32 = 2;
const NEG_HISTORY_REDUCTIONS: i32 = 2;
const NEG_SEE_REDUCTIONS: i32 = 2;

const INITIAL_ASPIRATION_WINDOW_SIZE: i16 = 16;
const INITIAL_ASPIRATION_WINDOW_STEP: i16 = 16;

const MAX_LMR_MOVES: usize = 64;

const LMR: [i32; MAX_LMR_MOVES] = calc_late_move_reductions();

const fn calc_late_move_reductions() -> [i32; MAX_LMR_MOVES] {
    let mut lmr = [0i32; MAX_LMR_MOVES];
    let mut moves = 0;
    while moves < MAX_LMR_MOVES {
        lmr[moves] = log2(1 + moves as u32 / 4);
        moves += 1;
    }

    lmr
}

fn lmp_threshold(improving: bool, depth: i32) -> i16 {
    if improving {
        params::lmp_improving(depth as usize)
    } else {
        params::lmp_not_improving(depth as usize)
    }
}

#[derive(Copy, Clone, Default)]
struct SearchInfo {
    eval: i16,
    in_check: bool,
    capture_pos: i8,
    opp_move: Move,
    in_singular_move_search: bool,
}

impl SearchInfo {
    pub fn set_eval(&mut self, eval: i16) {
        self.eval = eval;
    }

    pub fn set_capture_pos(&mut self, pos: i8) -> &mut Self {
        self.capture_pos = pos;
        self
    }

    pub fn set_opp_move(&mut self, m: Move) -> &mut Self {
        self.opp_move = m;
        self
    }

    pub fn set_in_singular_move_search(&mut self, in_singular_move_search: bool) -> &mut Self {
        self.in_singular_move_search = in_singular_move_search;
        self
    }

    pub fn set_in_check(&mut self, gives_check: bool) -> &mut Self {
        self.in_check = gives_check;
        self
    }

    pub fn eval(&self) -> i16 {
        self.eval
    }

    pub fn in_check(&self) -> bool {
        self.in_check
    }

    pub fn is_recapture(&self, target: i8) -> bool {
        self.capture_pos == target
    }

    pub fn opp_played_capture(&self) -> bool {
        self.capture_pos != -1
    }

    pub fn opp_move(&self) -> Move {
        self.opp_move
    }

    pub fn opp_played_null_move(&self) -> bool {
        self.opp_move == NO_MOVE
    }

    pub fn in_singular_move_search(&self, excluded_singular_move: Move) -> bool {
        self.in_singular_move_search || excluded_singular_move != NO_MOVE
    }
}

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
    current_depth: i32,
    max_reached_depth: usize,

    local_total_node_count: u64,
    local_tb_hits: u64,
    local_node_count: u64,
    multi_pv_count: usize,
    tb_probe_depth: i32,
    tb_probe_root: bool,

    node_count: Arc<AtomicU64>,
    is_stopped: Arc<AtomicBool>,
    tb_hits: Arc<AtomicU64>,

    threads: HelperThreads,
    is_helper_thread: bool,

    draw_score: i16,
    player_pov: Color,

    pondering: bool,

    tt_gen: u16,

    infos: [SearchInfo; MAX_DEPTH + 1],
}

impl Search {
    pub fn new(
        is_stopped: Arc<AtomicBool>, node_count: Arc<AtomicU64>, tb_hits: Arc<AtomicU64>, log_level: LogLevel, limits: SearchLimits,
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

            draw_score: 0,
            player_pov: WHITE,

            threads: HelperThreads::new(),
            is_helper_thread,

            pondering: false,

            tt_gen: 0,

            infos: [SearchInfo::default(); MAX_DEPTH + 1],
        }
    }

    pub fn resize_tt(&mut self, new_size_mb: i32) {
        // Remove all additional threads, which reference the transposition table
        let thread_count = self.threads.count();
        self.threads.resize(0, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped);

        // Resize transposition table
        Arc::get_mut(&mut self.tt).unwrap().resize(new_size_mb as u64);

        // Restart threads
        self.threads.resize(thread_count, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped);

        self.clear_tt();
    }

    pub fn reset_threads(&mut self, thread_count: i32) {
        self.threads.resize((thread_count - 1) as usize, &self.node_count, &self.tb_hits, &self.tt, &self.board, &self.is_stopped);
    }

    pub fn clear_tt(&mut self) {
        self.threads.clear_tt();
        TranspositionTable::clear(&self.tt, 0, self.threads.count() + 1);
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
    }

    pub fn update_limits(&mut self, limits: SearchLimits) {
        self.limits = limits;
    }

    pub fn reset(&mut self) {
        self.local_total_node_count = 0;
        self.local_node_count = 0;
        self.local_tb_hits = 0;
    }

    pub fn find_best_move(
        &mut self, rx: Option<&Receiver<Message>>, min_depth: i32, skipped_moves: &[Move],
    ) -> (Move, PrincipalVariation) {
        self.reset();
        self.time_mgr.reset(self.limits);

        self.board.pos_history.mark_root();

        self.last_log_time = Instant::now();

        self.cancel_possible = false;
        self.node_count.store(0, Ordering::Relaxed);
        self.tb_hits.store(0, Ordering::Relaxed);

        self.next_check_node_count = self.limits.node_limit().min(1000);

        self.tt_gen = self.board.fullmove_count() % (MAX_GENERATION + 1);

        let mut last_best_move: Move = NO_MOVE;

        let active_player = self.board.active_player();

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE);

        self.set_stopped(false);

        let mut pv = PrincipalVariation::default();

        let mut analysis_result = AnalysisResult::new();

        let mut skipped_moves = Vec::from(skipped_moves);

        // Probe tablebases
        let (draw_score, tb_result) = if self.tb_probe_root {
            if let Some((tb_result, mut tb_skip_moves)) = self.board.probe_root() {
                self.local_tb_hits += 1;
                // Adjust draw score based upon the tablebase result to prevent
                // accidental draws in winning TB positions when the eval is <= 0
                let draw_score = match tb_result {
                    TBResult::Loss => MATE_SCORE,
                    TBResult::BlessedLoss => 1,
                    TBResult::Draw => 0,
                    TBResult::CursedWin => -1,
                    TBResult::Win => MATED_SCORE,
                };
                skipped_moves.append(&mut tb_skip_moves);
                (draw_score, Some(tb_result))
            } else {
                (0, None)
            }
        } else {
            (0, None)
        };

        self.player_pov = self.board.active_player();
        self.draw_score = draw_score;
        self.threads.start_search(&self.board, &skipped_moves, self.multi_pv_count, self.tb_probe_depth, draw_score);

        let mut multi_pv_state = vec![
            (
                self.board.eval(),
                INITIAL_ASPIRATION_WINDOW_SIZE,
                INITIAL_ASPIRATION_WINDOW_STEP
            );
            self.multi_pv_count
        ];

        // Use iterative deepening, i.e. increase the search depth after each iteration
        for depth in 1..=self.limits.depth_limit() {
            self.max_reached_depth = 0;
            self.current_depth = depth;
            let iteration_start_time = Instant::now();

            let mut iteration_cancelled = false;

            let mut local_skipped_moves = skipped_moves.clone();
            for multi_pv_num in 1..=self.multi_pv_count {
                let mut local_pv = PrincipalVariation::default();
                let (score, mut window_step, mut window_size) = multi_pv_state[multi_pv_num - 1];

                let (cancelled, move_num, best_move, current_pv, new_window_step) =
                    self.root_search(rx, &local_skipped_moves, window_step, window_size, score, depth, &mut local_pv);
                if new_window_step > window_step {
                    window_step = new_window_step;
                    window_size = new_window_step;
                } else if window_step > 16 {
                    window_step /= 2;
                    window_size /= 2;
                }

                if cancelled {
                    iteration_cancelled = true;
                }

                if best_move == NO_MOVE {
                    break;
                }

                if !iteration_cancelled {
                    analysis_result.update_result(depth, self.max_reached_depth as i32, best_move, current_pv, local_pv);

                    let now = Instant::now();
                    self.cancel_possible = depth >= min_depth;
                    let iteration_duration = now.duration_since(iteration_start_time);
                    if !self.pondering
                        && self.cancel_possible
                        && !self.time_mgr.is_time_for_another_iteration(now, iteration_duration)
                        && !self.time_mgr.try_extend_timelimit()
                    {
                        iteration_cancelled = true;
                    }
                }

                if let Some(mate_distance) = mate_in(best_move.score()) {
                    if mate_distance <= self.limits.mate_limit() {
                        iteration_cancelled = true;
                    }
                }

                local_skipped_moves.push(best_move.without_score());
                multi_pv_state[multi_pv_num - 1] = (best_move.score(), window_size, window_step);

                if iteration_cancelled {
                    break;
                }

                if depth == 1 && multi_pv_num == 1 && move_num == 1 {
                    self.time_mgr.reduce_timelimit();
                }
            }

            analysis_result.finish_iteration();

            if self.log(Info) {
                analysis_result
                    .print(self.board.halfmove_clock(), tb_result, self.multi_pv_count, self.get_base_stats(self.time_mgr.search_duration(Instant::now())))
            }

            pv = analysis_result.get_best_pv();
            last_best_move = analysis_result.get_best_move();

            if iteration_cancelled {
                break;
            }
        }

        self.movegen.leave_ply();

        if let Some(r) = rx {
            while (self.limits.is_infinite() || self.pondering) && !self.is_stopped() {
                self.check_messages(r, true);
            }
        }

        self.set_stopped(true);

        self.threads.wait_for_completion();

        (last_best_move, pv)
    }

    fn root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], window_step: i16, window_size: i16,
        score: i16, depth: i32, pv: &mut PrincipalVariation) -> (bool, i32, Move, Option<String>, i16) {
        let mut alpha = if depth > 7 { score - window_size } else { MIN_SCORE };
        let mut beta = if depth > 7 { score + window_size } else { MAX_SCORE };

        self.infos[0].set_eval(self.board.eval());

        let mut step = window_step;
        loop {
            pv.clear();

            let (cancelled, move_num, best_move, current_pv) =
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

            if best_move == NO_MOVE || cancelled {
                return (cancelled, move_num, best_move, current_pv, step);
            }

            let best_score = best_move.score();
            if best_score <= alpha {
                alpha = MIN_SCORE.max(alpha.saturating_sub(step));
            } else if best_score >= beta {
                beta = MAX_SCORE.min(beta.saturating_add(step));
            } else {
                return (false, move_num, best_move, current_pv, step);
            }

            step = (MATE_SCORE / 2).min(step.saturating_mul(2));
        }
    }

    // Root search within the bounds of an aspiration window (alpha...beta)
    fn bounded_root_search(
        &mut self, rx: Option<&Receiver<Message>>, skipped_moves: &[Move], mut alpha: i16, beta: i16, depth: i32, pv: &mut PrincipalVariation
    ) -> (bool, i32, Move, Option<String>) {
        let mut move_num = 0;
        let mut a = -beta;
        let mut best_move: Move = NO_MOVE;
        let mut best_score = MIN_SCORE;

        let mut iteration_cancelled = false;
        let mut current_pv = None;

        let active_player = self.board.active_player();

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
                    let uci_move = UCIMove::from_move(&self.board, m.unpack());
                    println!("info currmove {} currmovenumber {}", uci_move, move_num);
                }
            }

            let upm = m.unpack();
            let (previous_piece, removed_piece_id) = self.board.perform_move(upm);

            let capture_pos = if removed_piece_id != EMPTY { upm.end } else { -1 };

            let gives_check = self.board.is_in_check(active_player.flip());

            let mut local_pv = PrincipalVariation::default();

            let mut tree_size = self.local_total_node_count;

            // Use principal variation search
            self.inc_node_count();
            self.infos[1]
                .set_capture_pos(capture_pos)
                .set_opp_move(m)
                .set_in_singular_move_search(false)
                .set_in_check(gives_check);

            let mut result = self.rec_find_best_move(rx, a, -alpha, 1, depth - 1, &mut local_pv, NO_MOVE);
            if result == CANCEL_SEARCH {
                iteration_cancelled = true;
            } else if -result > alpha && a != -beta {
                // Repeat search if it falls outside the search window
                result = self.rec_find_best_move(rx, -beta, -alpha, 1, depth - 1, &mut local_pv, NO_MOVE);
                if result == CANCEL_SEARCH {
                    iteration_cancelled = true;
                }
            }

            self.board.undo_move(upm, previous_piece, removed_piece_id);

            if iteration_cancelled {
                break;
            }

            let score = -result;
            if score > best_score {
                best_score = score;
                best_move = m.with_score(score);

                if best_score <= alpha || best_score >= beta {
                    return (false, move_num, best_move, current_pv);
                }

                pv.update(best_move, &mut local_pv);
                current_pv = Some(self.pv_info(&pv.moves()));

                alpha = score;

                self.time_mgr.update_best_move(best_move, depth);

                a = -(alpha + 1);
            }

            tree_size = self.local_total_node_count - tree_size;
            if move_num == 1 {
                tree_scale = 13.max(64 - tree_size.leading_zeros()) - 13;
            }
            self.movegen.update_root_move(m.with_score((MAX_SCORE as i32).min((tree_size >> tree_scale) as i32) as i16));
        }

        self.movegen.reorder_root_moves(best_move, self.is_helper_thread);

        (iteration_cancelled, move_num, best_move, current_pv)
    }

    fn node_count(&self) -> u64 {
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
        &mut self, rx: Option<&Receiver<Message>>, mut alpha: i16, beta: i16, ply: usize, mut depth: i32,
        pv: &mut PrincipalVariation, excluded_singular_move: Move,
    ) -> i16 {
        self.max_reached_depth = ply.max(self.max_reached_depth);

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
            return self.effective_draw_score();
        }

        let is_pv = (alpha + 1) < beta; // in a principal variation search, non-PV nodes are searched with a zero-window

        // Prune, if even the best possible score cannot improve alpha (because a shorter mate has already been found)
        let mut best_possible_score = MATE_SCORE - (ply as i16 + 1);
        if best_possible_score <= alpha {
            return best_possible_score;
        }

        let in_check = self.infos[ply].in_check();
        // Prune, if worst possible score is already sufficient to reach beta
        let mut worst_possible_score = MATED_SCORE + ply as i16 + if in_check { 0 } else { 1 };
        if worst_possible_score >= beta {
            return worst_possible_score;
        }

        let active_player = self.board.active_player();
        if in_check {
            // Extend search when in check
            depth = (depth + 1).max(1);
        }

        let mut pos_score = None;

        let hash = self.board.get_hash();

        let mut hash_move = NO_MOVE;
        let mut hash_score = 0;

        let mut check_se = false;

        let mut skip_null_move = false;

        let mut best_score = worst_possible_score;
        let mut best_move = NO_MOVE;
        let mut score_type = ScoreType::UpperBound;
        let mut is_tb_move = false;

        if excluded_singular_move == NO_MOVE {
            // Check transposition table
            let (tt_entry, tt_slot) = self.tt.get_entry(hash);
            if tt_entry != 0 {
                let packed_tt_move = get_hash_move(tt_entry);
                hash_move = packed_tt_move.unpack(active_player, &self.board.bitboards);
                let upm = hash_move.unpack();
                hash_score = upm.score;

                is_tb_move = if self.is_tb_move(upm) {
                    hash_move = NO_MOVE;
                    true
                } else if is_valid_move(&self.board, active_player, upm) {
                    false
                } else {
                    hash_move = NO_MOVE;
                    false
                };

                if hash_move != NO_MOVE || is_tb_move {
                    hash_score = to_root_relative_score(ply as i32, sanitize_score(hash_score));
                    let tt_depth = get_depth(tt_entry);
                    match get_score_type(tt_entry) {
                        ScoreType::Exact => {
                            if !is_pv && tt_depth >= depth {
                                if let Some(slot) = tt_slot { update_generation(tt_entry, slot, self.tt_gen) }
                                return hash_score;
                            }
                            pos_score = Some(hash_score);
                            check_se = tt_depth >= depth - 3;
                        }

                        ScoreType::UpperBound => {
                            if !is_pv && hash_score <= alpha && tt_depth >= depth {
                                if let Some(slot) = tt_slot { update_generation(tt_entry, slot, self.tt_gen) }
                                return hash_score;
                            }
                            skip_null_move = tt_depth >= depth - null_move_reduction(depth);
                        }

                        ScoreType::LowerBound => {
                            if !is_pv && tt_depth >= depth && hash_score.max(alpha) >= beta {
                                if hash_move != NO_MOVE && !hash_move.is_capture() {
                                    self.hh.update_killer_moves(ply, hash_move);
                                    self.hh.update_counter_move(self.infos[ply].opp_move, hash_move);
                                }
                                if let Some(slot) = tt_slot { update_generation(tt_entry, slot, self.tt_gen) }
                                return hash_score;
                            }
                            pos_score = Some(hash_score);
                            check_se = tt_depth >= depth - 3;
                        }
                    };

                    check_se = check_se
                        && hash_move != NO_MOVE
                        && !self.infos[ply].in_singular_move_search(excluded_singular_move)
                        && !self.in_null_move_search(ply)
                        && !in_check
                        && depth > 7
                        && !self.infos[ply].is_recapture(upm.end);
                }
            }

            if !is_tb_move {
                if hash_move == NO_MOVE && depth > 7 {
                    // Reduce nodes without hash move from transposition table
                    depth -= 1;
                }

                // Probe tablebases
                if depth.max(0) >= self.tb_probe_depth {
                    if let Some(tb_result) = self.board.probe_wdl() {
                        self.local_tb_hits += 1;

                        match tb_result {
                            TBResult::Draw => {
                                self.tt.write_entry(hash, self.tt_gen, depth, self.tb_move(0), ScoreType::Exact);
                                return 0;
                            },
                            TBResult::Win => {
                                worst_possible_score = 400;
                            }
                            TBResult::Loss => {
                                best_possible_score = -400;
                            },
                            TBResult::CursedWin => {
                                self.tt.write_entry(hash, self.tt_gen, depth, self.tb_move(1), ScoreType::Exact);
                                return 1;
                            },
                            TBResult::BlessedLoss => {
                                self.tt.write_entry(hash, self.tt_gen, depth, self.tb_move(-1), ScoreType::Exact);
                                return -1;
                            }
                        }
                    }
                }
            }
        }

        // Quiescence search
        if depth <= 0 || ply >= (MAX_DEPTH - 16) {
            return clamp_score(self.quiescence_search(active_player, alpha, beta, ply, pos_score), worst_possible_score, best_possible_score);
        }

        self.infos[ply].set_eval(self.board.eval());
        let improving = self.is_improving(ply);

        if !is_pv && !in_check {
            if depth <= 2 {
                // Jump directly to QS, if position is already so good, that it is unlikely for the opponent to counter it within the remaining search depth
                pos_score = pos_score.or_else(|| Some(self.infos[ply].eval()));
                let score = pos_score.unwrap();

                let margin = if improving {
                    (params::rfp_margin_multiplier_improving() << depth) + params::rfp_base_margin_improving()
                } else {
                    (params::rfp_margin_multiplier_not_improving() << depth) + params::rfp_base_margin_not_improving()
                };
                if !is_mate_or_mated_score(score) && score - margin >= beta {
                    return clamp_score(self.quiescence_search(active_player, alpha, beta, ply, pos_score), worst_possible_score, best_possible_score);
                }
            } else if !skip_null_move {
                // Null move pruning
                pos_score = pos_score.or_else(|| Some(self.infos[ply].eval()));
                if pos_score.unwrap() >= beta && self.board.has_non_pawns(active_player) {
                    self.board.perform_null_move();
                    self.tt.prefetch(self.board.get_hash());

                    let in_se_search = self.infos[ply].in_singular_move_search(excluded_singular_move);
                    self.infos[ply + 1]
                        .set_opp_move(NO_MOVE)
                        .set_in_singular_move_search(in_se_search)
                        .set_in_check(false)
                        .set_capture_pos(-1);

                    let result = self.rec_find_best_move(rx, -beta, -beta + 1, ply + 1, depth - null_move_reduction(depth), &mut PrincipalVariation::default(), NO_MOVE);
                    self.board.undo_null_move();
                    if result == CANCEL_SEARCH {
                        return CANCEL_SEARCH;
                    }
                    let score = clamp_score(-result, worst_possible_score, best_possible_score);
                    if score >= beta {
                        return if is_eval_score(score) { score } else { beta };
                    }
                }
            }
        }

        let mut evaluated_move_count = 0;
        let mut quiet_move_count = 0;
        let mut has_valid_moves = false;

        let allow_lmr = depth > 2;

        // Futile move pruning
        let mut allow_futile_move_pruning = false;
        if !is_pv && depth <= 6 && !in_check && self.current_depth >= 8 {
            let margin = (params::fp_margin_multiplier() << depth) + params::fp_base_margin();
            pos_score = pos_score.or_else(|| Some(self.infos[ply].eval()));
            let static_score = pos_score.unwrap();
            allow_futile_move_pruning = !is_mate_or_mated_score(static_score) && static_score + margin <= alpha;

            if depth == 1 && static_score + params::razor_margin_multiplier() <= alpha {
                let score = clamp_score(self.quiescence_search(active_player, alpha, beta, ply, pos_score), worst_possible_score, best_possible_score);
                if score <= alpha {
                    return score;
                }
            }
        }

        self.movegen.enter_ply(active_player, hash_move, self.infos[ply - 1].opp_move(), self.infos[ply].opp_move());

        let occupied_bb = self.board.occupancy_bb();

        let mut is_singular = false;

        let allow_lmp = !is_pv && !in_check && depth <= 2 && self.current_depth >= 7;

        self.hh.clear_killers(ply + 1);

        let mut a = -beta;
        while let Some(packed_curr_move) = self.movegen.next_move(ply, &self.hh, &mut self.board) {
            if excluded_singular_move == packed_curr_move {
                continue;
            }

            let curr_move = packed_curr_move.unpack();

            if allow_lmp && !curr_move.is_capture()
                && !is_killer(curr_move)
                && !curr_move.is_queen_promotion() && quiet_move_count > lmp_threshold(improving, depth) {
                continue;
            }

            let (previous_piece, removed_piece_id) = self.board.perform_move(curr_move);
            let gives_check = self.board.is_in_check(active_player.flip());

            // Check, if the hash move is singular and should be extended
            let mut se_extension = 0;
            if check_se && !gives_check && packed_curr_move == hash_move {
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                let se_beta = sanitize_score(hash_score - (5 + depth as i16 / 2));
                let result = self.rec_find_best_move(rx, se_beta - 1, se_beta, ply, depth / 2, &mut PrincipalVariation::default(), packed_curr_move);

                if result == CANCEL_SEARCH {
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                if result < se_beta {
                    se_extension = 1;
                    is_singular = true;
                } else if se_beta >= beta {
                    // Multi-Cut Pruning
                    self.movegen.leave_ply();
                    return clamp_score(se_beta, worst_possible_score, best_possible_score);
                }

                self.board.perform_move(curr_move);
            };

            let start = curr_move.start;
            let end = curr_move.end;

            let mut skip = self.board.is_in_check(active_player); // skip if move would put own king in check

            let mut reductions = 0;

            if !skip {
                let target_piece_id = curr_move.move_type.piece_id();
                has_valid_moves = true;

                if !is_pv && self.infos[ply].opp_played_capture() && !self.infos[ply].is_recapture(curr_move.end) && evaluated_move_count > 0 && !curr_move.is_queen_promotion() {
                    reductions += 1;
                }

                if se_extension == 0 && removed_piece_id == EMPTY {

                    if allow_lmr && quiet_move_count > LMR_THRESHOLD && !curr_move.is_queen_promotion()  {
                        reductions += unsafe { *LMR.get_unchecked((quiet_move_count as usize).min(MAX_LMR_MOVES - 1)) } + i32::from(!is_pv);
                        if !improving && curr_move.score <= NEGATIVE_HISTORY_SCORE {
                            reductions += 1;
                        }

                    } else if allow_futile_move_pruning && !gives_check && !curr_move.is_queen_promotion() {
                        // Reduce futile move
                        reductions += FUTILE_MOVE_REDUCTIONS;
                        if target_piece_id == P && is_passed_pawn(end as usize, active_player, self.board.get_bitboard(active_player.flip().piece(P))) {
                            reductions -= 1;
                        }

                    } else if curr_move.score <= NEGATIVE_HISTORY_SCORE {
                        reductions += NEG_HISTORY_REDUCTIONS;

                    } else if curr_move.score <= QUIET_BASE_SCORE
                        && self.board.has_negative_see( active_player.flip(), start as usize, end as usize, target_piece_id, EMPTY, 0, occupied_bb)
                    {
                        // Reduce search depth for moves with negative SEE score
                        reductions += NEG_SEE_REDUCTIONS;
                        if curr_move.score < QUIET_BASE_SCORE && evaluated_move_count > 0 && depth <= 3 {
                            skip = true;
                        }
                    }

                    if is_singular {
                        reductions += 1;
                    }

                    quiet_move_count += 1;

                    if allow_futile_move_pruning
                        && evaluated_move_count > 0
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
                self.tt.prefetch(self.board.get_hash());

                let new_capture_pos = if removed_piece_id != EMPTY { end } else { -1 };

                evaluated_move_count += 1;

                let mut local_pv = PrincipalVariation::default();

                self.inc_node_count();
                let in_se_search = self.infos[ply].in_singular_move_search(excluded_singular_move);
                self.infos[ply + 1]
                    .set_opp_move(packed_curr_move)
                    .set_capture_pos(new_capture_pos)
                    .set_in_check(gives_check)
                    .set_in_singular_move_search(in_se_search);

                let mut result = self.rec_find_best_move(rx, a, -alpha, ply + 1, depth + se_extension - reductions - 1, &mut local_pv, NO_MOVE);
                if result == CANCEL_SEARCH {
                    self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                    self.movegen.leave_ply();
                    return CANCEL_SEARCH;
                }

                if -result > alpha && (reductions > 0 || (-result < beta && a != -beta)) {
                    // Repeat search without reduction and with full window
                    result = self.rec_find_best_move(rx, -beta, -alpha, ply + 1, depth + se_extension - 1, &mut local_pv, NO_MOVE);
                    if result == CANCEL_SEARCH {
                        self.board.undo_move(curr_move, previous_piece, removed_piece_id);
                        self.movegen.leave_ply();
                        return CANCEL_SEARCH;
                    }
                }

                let score = clamp_score(-result, worst_possible_score, best_possible_score);
                self.board.undo_move(curr_move, previous_piece, removed_piece_id);

                if score > best_score {
                    best_score = score;
                    best_move = packed_curr_move;

                    // Alpha-beta pruning
                    if best_score >= beta {
                        if excluded_singular_move == NO_MOVE {
                            let tt_move = best_move.with_score(from_root_relative_score(ply as i32, best_score)).to_tt_packed_move(active_player, &self.board.bitboards);
                            self.tt.write_entry(
                                hash,
                                self.tt_gen,
                                depth,
                                tt_move,
                                ScoreType::LowerBound,
                            );
                        }

                        if removed_piece_id == EMPTY {
                            self.hh.update(ply, active_player, self.infos[ply - 1].opp_move(), self.infos[ply].opp_move(), best_move);
                        }

                        self.movegen.leave_ply();
                        return best_score;
                    }

                    if best_score > alpha {
                        alpha = best_score;
                        score_type = ScoreType::Exact;
                        if is_pv {
                            pv.update(best_move, &mut local_pv);
                        }
                    }
                } else if removed_piece_id == EMPTY {
                    self.hh.update_played_moves(active_player, self.infos[ply - 1].opp_move(), self.infos[ply].opp_move(), packed_curr_move);
                }

                a = -(alpha + 1);
            }
        }

        self.movegen.leave_ply();

        if !has_valid_moves {
            return if excluded_singular_move != NO_MOVE {
                alpha
            } else if in_check {
                MATED_SCORE + ply as i16 // Check mate
            } else {
                self.effective_draw_score()
            }
        }

        best_score = best_score.min(best_possible_score);

        if excluded_singular_move == NO_MOVE {
            let tt_move = best_move.with_score(from_root_relative_score(ply as i32, best_score)).to_tt_packed_move(active_player, &self.board.bitboards);

            self.tt.write_entry(
                hash,
                self.tt_gen,
                depth,
                tt_move,
                score_type,
            );
        }

        best_score
    }

    fn check_search_limits(&mut self, rx: &Receiver<Message>) {
        if self.local_total_node_count >= self.next_check_node_count {
            self.next_check_node_count = if self.limits.node_limit() != u64::MAX {
                self.limits.node_limit()
            } else {
                self.local_total_node_count + 1000
            };

            self.check_messages(rx, false);

            let now = Instant::now();
            if !self.pondering
                && self.cancel_possible
                && (self.node_count() >= self.limits.node_limit() || self.time_mgr.is_timelimit_exceeded(now))
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

    pub fn quiescence_search(&mut self, active_player: Color, mut alpha: i16, beta: i16, ply: usize, pos_score: Option<i16>) -> i16 {
        self.max_reached_depth = ply.max(self.max_reached_depth);

        if ply >= MAX_DEPTH {
            return beta;
        }

        let position_score = pos_score.unwrap_or_else(|| { self.board.eval() });
        if position_score > alpha {
            if position_score >= beta {
                return position_score;
            }
            alpha = position_score;
        }

        self.movegen.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE);
        self.movegen.generate_qs_captures(&mut self.board);

        let mut threshold = alpha - position_score - params::qs_see_threshold();
        let mut best_score = position_score;

        let opp_player = active_player.flip();
        while let Some(m) = self.movegen.next_good_capture_move(&mut self.board, threshold) {
            let upm = m.unpack();
            let (previous_piece, captured_piece_id) = self.board.perform_move(upm);
            if self.board.is_in_check(active_player) {
                self.board.undo_move(upm, previous_piece, captured_piece_id);
                continue;
            }
            self.inc_node_count();

            let score = if self.board.is_insufficient_material_draw() {
                -self.effective_draw_score()
            } else {
                -self.quiescence_search(opp_player, -beta, -alpha, ply + 1, None)
            };
            self.board.undo_move(upm, previous_piece, captured_piece_id);

            if score > best_score {
                best_score = score;
                if best_score > alpha {
                    if best_score >= beta {
                        break;
                    }

                    alpha = best_score;
                    threshold = alpha - position_score - params::qs_see_threshold();
                }
            }
        }

        self.movegen.leave_ply();
        best_score
    }

    pub fn determine_skipped_moves(&mut self, search_moves: Vec<String>) -> Vec<Move> {
        let mut search_moves_set = MoveSet::default();
        for uci_move in search_moves.iter() {
            if let Some(m) = UCIMove::from_uci(uci_move) {
                search_moves_set.insert(m.to_move(&self.board));
            }
        }

        self.movegen.reset_root_moves();
        let mut skipped_moves = Vec::new();
        while let Some(m) = self.movegen.next_root_move(&self.hh, &mut self.board) {
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
        if let Some((m, rest_pv)) = pv.split_first() {
            let upm = m.unpack();
            let uci_move = UCIMove::from_move(&self.board, upm);
            let (previous_piece, move_state) = self.board.perform_move(upm);

            let followup_moves = self.pv_info(rest_pv);

            self.board.undo_move(upm, previous_piece, move_state);
            format!("{} {}", uci_move, followup_moves)
        } else {
            self.pv_info_from_tt()
        }
    }

    fn pv_info_from_tt(&mut self) -> String {
        if self.board.is_draw() {
            return String::new();
        }

        let (entry, _) = self.tt.get_entry(self.board.get_hash());
        let active_player = self.board.active_player();
        let packed_tt_move = get_hash_move(entry);
        if packed_tt_move.is_no_move() {
            return String::new();
        }
        let tt_move = packed_tt_move.unpack(active_player, &self.board.bitboards);
        let upm = tt_move.unpack();
        if !is_valid_move(&self.board, active_player, upm) {
            return String::new();
        }

        let uci_move = UCIMove::from_move(&self.board, upm);
        let (previous_piece, move_state) = self.board.perform_move(upm);

        let followup_moves = self.pv_info_from_tt();

        self.board.undo_move(upm, previous_piece, move_state);
        format!("{} {}", uci_move, followup_moves)
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
        if self.board.active_player().0 == self.player_pov.0 {
            self.draw_score
        } else {
            -self.draw_score
        }
    }

    fn in_null_move_search(&self, ply: usize) -> bool {
        self.infos.iter().skip(1).take(ply).any(|s| s.opp_played_null_move())
    }

    fn is_improving(&self, ply: usize) -> bool {
        if ply <= 1 || self.infos[ply].in_check {
            return false;
        }
        let prev_eval = self.infos[ply - 2].eval();
        let curr_eval = self.infos[ply].eval();

        curr_eval > prev_eval
    }

    fn tb_move(&self, score: i16) -> TTPackedMove {
        let active_player = self.board.active_player();
        let m = Move::new(MoveType::TableBaseMarker, 0, self.board.king_pos(active_player)).with_initial_score(score);
        TTPackedMove::new(m.to_u32())
    }

    #[inline]
    fn is_tb_move(&self, m: UnpackedMove) -> bool {
        if !matches!(m.move_type, MoveType::TableBaseMarker) {
            return false;
        }

        let active_player = self.board.active_player();
        m.end == self.board.king_pos(active_player)
    }
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
        draw_score: i16,
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
        &mut self, target_count: usize, node_count: &Arc<AtomicU64>, tb_hits: &Arc<AtomicU64>, tt: &Arc<TranspositionTable>, board: &Board,
        is_stopped: &Arc<AtomicBool>
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
                let sub_search = Search::new(is_stopped, node_count, tb_hits, LogLevel::Error, limits, tt, board, true);

                HelperThread::run(to_rx, from_tx, sub_search);
            });

            self.threads.push(HelperThread { handle, to_tx, from_rx });
        }
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    pub fn start_search(&self, board: &Board, skipped_moves: &[Move], multi_pv_count: usize, tb_probe_depth: i32, draw_score: i16) {
        for t in self.threads.iter() {
            t.search(board, skipped_moves, multi_pv_count, tb_probe_depth, draw_score);
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
    pub fn search(&self, board: &Board, skipped_moves: &[Move], multi_pv_count: usize, tb_probe_depth: i32, draw_score: i16) {
        match self.to_tx.send(ToThreadMessage::Search {
            pos_history: board.pos_history.clone(),
            bitboards: board.bitboards,
            halfmove_count: board.halfmove_count,
            state: board.state,
            castling_rules: board.castling_rules,
            skipped_moves: Vec::from(skipped_moves),
            multi_pv_count,
            tb_probe_depth,
            draw_score
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
                    multi_pv_count,
                    tb_probe_depth,
                    draw_score
                } => {
                    sub_search.reset();
                    sub_search.board.reset(pos_history, bitboards, halfmove_count, state, castling_rules);
                    sub_search.set_tb_probe_depth(tb_probe_depth);
                    sub_search.draw_score = draw_score;
                    sub_search.player_pov = sub_search.board.active_player();
                    sub_search.tt_gen = sub_search.board.fullmove_count() % (MAX_GENERATION + 1);

                    sub_search.movegen.enter_ply(sub_search.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE);

                    let mut multi_pv_state = vec![
                        (
                            sub_search.board.eval(),
                            INITIAL_ASPIRATION_WINDOW_SIZE,
                            INITIAL_ASPIRATION_WINDOW_STEP
                        );
                        multi_pv_count
                    ];

                    let mut found_moves = false;
                    for depth in 1..MAX_DEPTH {
                        sub_search.current_depth = depth as i32;
                        let mut local_skipped_moves = skipped_moves.clone();
                        for multi_pv_num in 1..=multi_pv_count {
                            let (score, mut window_size, mut window_step) = multi_pv_state[multi_pv_num - 1];
                            let (_, move_count, best_move, _, new_window_step) = sub_search.root_search(
                                None,
                                &local_skipped_moves,
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

                            if move_count == 0 {
                                break;
                            } else {
                                found_moves = true;
                            }

                            if sub_search.is_stopped() {
                                break;
                            }

                            multi_pv_state[multi_pv_num - 1] = (best_move.score(), window_size, window_step);
                            local_skipped_moves.push(best_move.without_score());
                        }

                        if sub_search.is_stopped() || !found_moves {
                            break;
                        }
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

#[inline]
fn null_move_reduction(depth: i32) -> i32 {
    log2((depth * 3 - 4) as u32) + 1
}

#[inline]
const fn log2(i: u32) -> i32 {
    (32 - i.leading_zeros()) as i32 - 1
}

#[cfg(test)]
mod tests {
    use std::sync::Once;
    use super::*;
    use crate::board::castling::{CastlingRules, CastlingState};
    use crate::board::Board;
    use crate::colors::{BLACK, WHITE};
    use crate::fen::{create_from_fen, write_fen};
    use crate::init::init;
    use crate::moves::NO_MOVE;
    use crate::pieces::{K, R};

    static INIT: Once = Once::new();

    pub fn initialize() {
        INIT.call_once(|| {
            init();
        })
    }

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

        board.perform_move(m.unpack());

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
        board.perform_move(m1.unpack());

        let m2 = search(tt.clone(), limits, board.clone(), 2);
        board.perform_move(m2.unpack());

        let m3 = search(tt.clone(), limits, board.clone(), 1);
        board.perform_move(m3.unpack());

        let is_check_mate = search(tt, limits, board.clone(), 1) == NO_MOVE && board.is_in_check(BLACK);
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
        let tb_move = search.tb_move(123);
        assert!(search.is_tb_move(tb_move.unpack(search.board.active_player(), &search.board.bitboards).unpack()));
    }

    fn search(tt: Arc<TranspositionTable>, limits: SearchLimits, board: Board, min_depth: i32) -> Move {
        let mut search = new_search(tt, limits, board);
        let (m, _) = search.find_best_move(None, min_depth, &[]);
        m
    }

    fn new_search(tt: Arc<TranspositionTable>, limits: SearchLimits, board: Board) -> Search {
        initialize();

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

    #[test]
    fn calc_log2() {
        for i in 1..65536 {
            assert_eq!(log2(i), (i as f32).log2() as i32)
        }
    }
}
