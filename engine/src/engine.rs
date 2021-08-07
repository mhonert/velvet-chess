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

use crate::board::Board;
use crate::colors::{WHITE};
use crate::fen::{create_from_fen, read_fen, write_fen, START_POS};
use crate::history_heuristics::HistoryHeuristics;
use crate::perft::perft;
use crate::search::Search;
use crate::transposition_table::{TranspositionTable, DEFAULT_SIZE_MB, MAX_DEPTH};
use crate::uci_move::UCIMove;
use std::cmp::max;
use std::process::exit;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Instant, SystemTime};
use crate::moves::{NO_MOVE, Move};
use crate::move_gen::MoveGenerator;
use crate::time_management::{TimeManager, MAX_TIMELIMIT_MS, TIMEEXT_MULTIPLIER};

pub enum Message {
    NewGame,
    SetPosition(String, Vec<UCIMove>),
    SetTranspositionTableSize(i32),
    Go {
        depth: i32,
        wtime: i32,
        btime: i32,
        winc: i32,
        binc: i32,
        movetime: i32,
        movestogo: i32,
        nodes: u64
    },
    Perft(i32),
    IsReady,
    Stop,
    Fen,
    Profile,
    SetOption(String, i32),
    SetArrayOption(String, i32, i32),
    Quit,
}

#[repr(u8)]
#[derive(PartialOrd, PartialEq)]
pub enum LogLevel {
    Debug,
    Info,
    Error
}

pub struct Engine {
    pub rx: Receiver<Message>,
    pub board: Board,
    pub movegen: MoveGenerator,
    pub hh: HistoryHeuristics,
    pub tt: TranspositionTable,
    pub time_mgr: TimeManager,

    pub node_limit: u64,
    pub depth_limit: i32,

    pub cancel_possible: bool,
    pub node_count: u64,
    pub last_log_time: Instant,
    pub next_check_node_count: u64,
    pub current_depth: i32,
    pub max_reached_depth: i32,

    pub is_stopped: bool,

    pub check_stop_cmd: bool,

    log_level: LogLevel,

    options_modified: bool,
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

        let timeext_history_size = board.options.get_timeext_history_size();
        let final_score_drop_threshold = board.options.get_timeext_score_drop_threshold();

        Engine {
            rx,
            board,
            movegen: MoveGenerator::new(),
            hh: HistoryHeuristics::new(),
            tt: TranspositionTable::new(tt_size_mb),
            time_mgr: TimeManager::new(timeext_history_size, final_score_drop_threshold),
            node_limit: u64::MAX,
            depth_limit: MAX_DEPTH as i32,
            cancel_possible: false,
            node_count: 0,
            last_log_time: Instant::now(),
            next_check_node_count: 0,
            current_depth: 0,
            max_reached_depth: 0,
            check_stop_cmd: true,
            is_stopped: false,
            options_modified: false,
            log_level: LogLevel::Info
        }
    }

    pub fn new(rx: Receiver<Message>) -> Self {
        Engine::new_from_fen(rx, START_POS, DEFAULT_SIZE_MB)
    }

    pub fn set_log_level(&mut self, log_level: LogLevel) {
        self.log_level = log_level;
    }

    pub fn log(&self, log_level: LogLevel) -> bool {
        self.log_level <= log_level
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

            Message::SetTranspositionTableSize(size_mb) => self.set_tt_size(size_mb),

            Message::Perft(depth) => self.perft(depth),

            Message::IsReady => self.check_readiness(),

            Message::Go { depth, wtime, btime, winc, binc, movetime, movestogo, nodes } =>
                self.go(depth, wtime, btime, winc, binc, movetime, movestogo, nodes),

            Message::Fen => println!("{}", write_fen(&self.board)),

            Message::Profile => self.profile(),

            Message::SetOption(name, value) => {
                self.board.options.set_option(name, value);
                self.options_modified = true;
            },

            Message::SetArrayOption(name, index, value) => {
                self.board.options.set_array_option(name, index as usize, value);
                self.options_modified = true;
            },

            Message::Quit => {
                return false;
            },

            Message::Stop => (),
        }

        true
    }

    fn go(&mut self, depth: i32, wtime: i32, btime: i32, winc: i32, binc: i32, movetime: i32, movestogo: i32, nodes: u64) {
        self.depth_limit = depth;
        let timelimit_ms = if self.board.active_player() == WHITE {
            calc_timelimit(movetime, wtime, winc, movestogo)
        } else {
            calc_timelimit(movetime, btime, binc, movestogo)
        };

        let time_left = if self.board.active_player() == WHITE {
            wtime
        } else {
            btime
        };

        self.node_limit = nodes;

        let is_strict_timelimit = movetime > 0 || timelimit_ms == MAX_TIMELIMIT_MS
            || movestogo == 1 || (time_left - (TIMEEXT_MULTIPLIER * timelimit_ms) <= 20);

        let m = self.find_best_move(3, timelimit_ms, is_strict_timelimit, &[]);
        if m == NO_MOVE {
            println!("bestmove 0000")
        } else {
            println!(
                "bestmove {}",
                UCIMove::from_encoded_move(&self.board, m).to_uci()
            );
        }
    }

    fn check_readiness(&mut self) {
        if self.options_modified {
            self.options_modified = false;

            let history_size = self.board.options.get_timeext_history_size();
            let final_score_drop_threshold = self.board.options.get_timeext_score_drop_threshold();

            self.time_mgr.update_params(history_size, final_score_drop_threshold);

        }
        println!("readyok")
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

    fn set_tt_size(&mut self, size_mb: i32) {
        self.tt.resize(size_mb as u64, false);
    }

    pub fn reset(&mut self) {
        self.tt.clear();
        self.hh.clear();
    }

    fn perft(&mut self, depth: i32) {
        let start = SystemTime::now();

        let nodes = perft(&mut self.movegen, &mut self.board, depth);

        let duration = match SystemTime::now().duration_since(start) {
            Ok(v) => v,
            Err(e) => panic!("error calculating duration: {:?}", e),
        };

        println!("Nodes: {}", nodes);
        println!("Duration: {:?}", duration);

        let duration_micro = duration.as_micros();
        if duration_micro > 0 {
            let nodes_per_sec = nodes * 1_000_000 / duration_micro as u64;
            println!("Nodes per second: {}", nodes_per_sec);
        }
    }

    pub fn perform_move(&mut self, m: Move) {
        self.board.perform_move(m);
    }

    pub fn profile(&mut self) {
        println!("Profiling ...");
        self.go(MAX_DEPTH as i32, -1, -1, 0, 0, -1, 1, 500000); // search 500.000 nodes
        exit(0);
    }

    pub fn is_search_stopped(&self) -> bool {
        match self.rx.try_recv() {
            Ok(msg) => {
                match msg {
                    Message::IsReady => println!("readyok"),

                    Message::Stop => {
                        return true;
                    }

                    _ => ()
                }
            },

            Err(e) => {
                match e {
                    TryRecvError::Empty => (),
                    TryRecvError::Disconnected => {
                        eprintln!("Error communicating with UCI thread: {}", e);
                        return true;
                    }
                }
            }
        }

        false
    }

}

const TIME_SAFETY_MARGIN_MS: i32 = 20;

fn calc_timelimit(movetime: i32, time_left: i32, time_increment: i32, movestogo: i32) -> i32 {
    if movetime == -1 && time_left == -1 {
        return MAX_TIMELIMIT_MS;
    }

    if movetime > 0 {
        return max(0, movetime - TIME_SAFETY_MARGIN_MS);
    }

    let time_for_move = time_left / max(1, movestogo);

    if time_for_move > time_left - TIME_SAFETY_MARGIN_MS {
        return max(0, time_left - TIME_SAFETY_MARGIN_MS)
    }

    let time_bonus = if movestogo > 1 { time_increment } else { 0 };
    if time_for_move + time_bonus > time_left - TIME_SAFETY_MARGIN_MS {
        time_for_move
    } else {
        time_for_move + time_bonus
    }

}
