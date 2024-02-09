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

use crate::board::Board;
use crate::fen::{create_from_fen, read_fen, write_fen, START_POS};
use crate::history_heuristics::HistoryHeuristics;
use crate::moves::{Move, NO_MOVE};
use crate::nn::init_nn_params;
use crate::perft::perft;
use crate::search::{Search, DEFAULT_SEARCH_THREADS};
use crate::time_management::SearchLimits;
use crate::transposition_table::{TranspositionTable, DEFAULT_SIZE_MB};
use crate::uci_move::UCIMove;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Instant};
use crate::{params, syzygy};
use crate::search_context::SearchContext;

pub enum Message {
    ClearHash,
    Fen,
    Go(SearchLimits, bool, Option<Vec<String>>),
    IsReady,
    NewGame,
    Perft(i32),
    Profile,
    Quit,
    SetPosition(String, Vec<UCIMove>),
    SetParam(String, i16),
    SetThreadCount(i32),
    SetTableBasePath(String),
    SetTableBaseProbeDepth(i32),
    SetMultiPV(i32),
    SetTranspositionTableSize(i32),
    Stop,
    PonderHit,
}

#[repr(u8)]
#[derive(PartialOrd, PartialEq, Copy, Clone)]
pub enum LogLevel {
    Debug,
    Info,
    Error,
}

pub struct Engine {
    rx: Receiver<Message>,
    board: Board,
    log_level: LogLevel,
    initialized: bool,
    new_thread_count: Option<i32>,
    current_thread_count: i32,
    new_tt_size: Option<i32>,
    current_tt_size: i32,
    new_tb_path: Option<String>,
    search: Search,
}

pub fn spawn_engine_thread() -> Sender<Message> {
    let (tx, rx) = mpsc::channel::<Message>();

    thread::spawn(move || {
        let mut engine = Engine::new(rx);
        engine.start_loop();
    });

    tx
}

impl Engine {
    pub fn new_from_fen(rx: Receiver<Message>, fen: &str, tt_size_mb: u64) -> Self {
        let mut board = create_from_fen(fen);
        board.reset_nn_eval();

        let search = Search::new(
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            LogLevel::Info,
            SearchLimits::default(),
            TranspositionTable::new(tt_size_mb),
            board.clone(),
            false,
        );

        Engine {
            rx,
            board,
            log_level: LogLevel::Info,
            initialized: false,
            new_thread_count: None,
            current_thread_count: DEFAULT_SEARCH_THREADS as i32,
            new_tt_size: None,
            current_tt_size: DEFAULT_SIZE_MB as i32,
            search,
            new_tb_path: None
        }
    }

    pub fn new(rx: Receiver<Message>) -> Self {
        Engine::new_from_fen(rx, START_POS, DEFAULT_SIZE_MB)
    }

    pub fn set_log_level(&mut self, log_level: LogLevel) {
        self.log_level = log_level;
    }

    fn start_loop(&mut self) {
        loop {
            match self.rx.recv() {
                Ok(msg) => {
                    if !self.handle_message(msg) {
                        return;
                    }
                }
                Err(err) => {
                    println!("Engine communication error: {:?}", err);
                    return;
                }
            }
        }
    }

    fn handle_message(&mut self, msg: Message) -> bool {
        match msg {
            Message::NewGame => self.reset(),

            Message::SetPosition(fen, moves) => self.set_position(fen, moves),

            Message::SetTranspositionTableSize(size_mb) => {
                if size_mb != self.current_tt_size {
                    self.new_tt_size = Some(size_mb);
                    if self.initialized {
                        self.update_tt_size();
                    }
                }
            }

            Message::SetThreadCount(count) => {
                if count != self.current_thread_count {
                    self.new_thread_count = Some(count);
                    if self.initialized {
                        self.update_thread_count();
                    }
                }
            }

            Message::SetTableBasePath(path) => {
                self.new_tb_path = Some(path.trim().to_string());
                if self.initialized {
                    self.update_tb();
                }
            }

            Message::SetTableBaseProbeDepth(depth) => {
                self.search.set_tb_probe_depth(depth);
            }

            Message::SetMultiPV(count) => {
                self.search.set_multi_pv_count(count);
            }

            Message::Perft(depth) => self.perft(depth),

            Message::IsReady => self.check_readiness(),

            Message::Go(limits, ponder, search_moves) => self.go(limits, ponder, search_moves),

            Message::Fen => println!("{}", write_fen(&self.board)),

            Message::Profile => {
                self.profile();
                return false;
            }

            Message::Quit => {
                return false;
            }

            Message::Stop => (),

            Message::PonderHit => println!("info Received 'ponderhit' outside ongoing search"),

            Message::ClearHash => self.search.clear_tt(),

            Message::SetParam(name, value) => {
                params::set(name, value);
            }
        }

        true
    }

    fn go(&mut self, limits: SearchLimits, ponder: bool, search_moves: Option<Vec<String>>) {
        let (m, ponder_m) = self.search(limits, ponder, search_moves);
        if m == NO_MOVE {
            println!("bestmove 0000");
            return;
        }

        let move_info = UCIMove::from_move(&self.board, m);

        if ponder_m != NO_MOVE {
            println!("bestmove {} ponder {}", move_info, self.encode_ponder_move(m, ponder_m));
        } else {
            println!("bestmove {}", move_info);
        };
    }

    fn encode_ponder_move(&mut self, own_move: Move, ponder_move: Move) -> String {
        let (own_piece, removed_piece) = self.board.perform_move(own_move);
        let ponder_move_uci = UCIMove::from_move(&self.board, ponder_move);
        self.board.undo_move(own_move, own_piece, removed_piece);

        ponder_move_uci
    }

    fn search(&mut self, mut limits: SearchLimits, ponder: bool, search_moves: Option<Vec<String>>) -> (Move, Move) {
        init_nn_params();
        let skipped_moves = if let Some(search_moves) = search_moves {
            self.search.determine_skipped_moves(search_moves)
        } else {
            vec![]
        };

        limits.update(self.board.active_player());

        self.search.update(&self.board, limits, ponder);

        let (m, pv) = self.search.find_best_move(Some(&self.rx), 3, &skipped_moves);
        let ponder_m = *pv.moves().get(1).unwrap_or(&NO_MOVE);
        (m, ponder_m)
    }

    fn check_readiness(&mut self) {
        // Peform postponed initializations
        init_nn_params();
        self.update_thread_count();
        self.update_tt_size();
        self.update_tb();

        self.initialized = true;
        println!("readyok");
    }

    fn update_thread_count(&mut self) {
        if let Some(count) = self.new_thread_count {
            self.search.reset_threads(count);
            self.current_thread_count = count;
            self.new_thread_count = None;
        }
    }

    fn update_tt_size(&mut self) {
        if let Some(new_tt_size) = self.new_tt_size {
            self.search.resize_tt(new_tt_size);
            self.current_tt_size = new_tt_size;
            self.new_tt_size = None;
        }
    }

    fn update_tb(&mut self) {
        if let Some(path) = self.new_tb_path.clone() {
            if !syzygy::tb::init(path.clone()) {
                eprintln!("could not initialize tablebases using path: {}", path);
            } else {
                let count = syzygy::tb::max_piece_count();
                if count == 0 {
                    println!("debug no tablebases found");
                } else {
                    println!("debug found {}-men tablebases", syzygy::tb::max_piece_count());
                }

            }
            self.new_tb_path = None;
        }
    }

    pub fn set_position(&mut self, fen: String, moves: Vec<UCIMove>) {
        match read_fen(&mut self.board, &fen) {
            Ok(_) => (),
            Err(err) => println!("position cmd: {}", err),
        }

        for m in moves {
            self.board.perform_move(m.to_move(&self.board));
        }

        self.board.reset_nn_eval();
    }

    pub fn reset(&mut self) {
        self.search.clear_tt();
        self.search.hh.clear();
    }

    fn perft(&mut self, depth: i32) {
        let start = Instant::now();

        let mut ctx = SearchContext::default();
        let hh = HistoryHeuristics::default();
        let nodes = perft(&mut ctx, &hh, &mut self.board, depth);

        let duration = start.elapsed();

        println!("Nodes: {}", nodes);
        println!("Duration: {:?}", duration);

        let duration_micro = duration.as_micros();
        if duration_micro > 0 {
            let nodes_per_sec = nodes * 1_000_000 / duration_micro as u64;
            println!("Nodes per second: {}", nodes_per_sec);
        }
    }
    pub fn profile(&mut self) {
        println!("Profiling ...");
        self.go(SearchLimits::nodes(100_000), false, None);
    }
}
