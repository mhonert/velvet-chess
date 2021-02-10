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
use crate::colors::{WHITE, BLACK};
use crate::fen::{create_from_fen, read_fen, write_fen, START_POS};
use crate::history_heuristics::HistoryHeuristics;
use crate::perft::perft;
use crate::pieces::{Q, get_piece_value};
use crate::search::Search;
use crate::transposition_table::{TranspositionTable, DEFAULT_SIZE_MB, MAX_DEPTH};
use crate::uci_move::UCIMove;
use std::cmp::max;
use std::process::exit;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::thread;
use std::time::{Instant, SystemTime};
use crate::eval::Eval;
use crate::score_util::{MIN_SCORE, MAX_SCORE};
use crate::random::Random;
use crate::moves::{NO_MOVE, Move};
use crate::move_gen::MoveGenerator;
use crate::tuning::Tuning;

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
    PrepareEval(Vec<(String, f64)>),
    PrepareQuiet(Vec<(String, f64)>),
    Eval(f64),
    Fen,
    PrintTestPositions,
    ResetTestPositions,
    Profile,
    SetOption(String, i32),
    SetArrayOption(String, i32, i32),
    Quit,
}

#[derive(Copy, Clone)]
pub struct EvalBoardPos {
    result: f64,
    pieces: [i8; 64],
    halfmove_count: u16,
    castling_state: u8,
    is_quiet: bool
}

impl EvalBoardPos {
    pub fn apply(&self, board: &mut Board) {
        board.eval_set_position(&self.pieces, self.halfmove_count, self.castling_state);
    }
}

pub struct Engine {
    pub rx: Receiver<Message>,
    pub board: Board,
    pub movegen: MoveGenerator,
    pub hh: HistoryHeuristics,
    pub tt: TranspositionTable,

    pub starttime: Instant,
    pub timelimit_ms: i32,
    pub node_limit: u64,
    pub depth_limit: i32,

    pub cancel_possible: bool,
    pub node_count: u64,
    pub last_log_time: Instant,
    pub next_check_node_count: u64,
    pub current_depth: i32,
    pub max_reached_depth: i32,

    pub is_stopped: bool,

    pub rnd: Random,

    options_modified: bool,

    test_positions: Vec<EvalBoardPos>
}

pub const TIMEEXT_MULTIPLIER: i32 = 5;

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
        Engine {
            rx,
            board: create_from_fen(&fen),
            movegen: MoveGenerator::new(),
            hh: HistoryHeuristics::new(),
            tt: TranspositionTable::new(tt_size_mb),
            starttime: Instant::now(),
            timelimit_ms: 0,
            node_limit: u64::max_value(),
            depth_limit: MAX_DEPTH as i32,
            cancel_possible: false,
            node_count: 0,
            last_log_time: Instant::now(),
            next_check_node_count: 0,
            current_depth: 0,
            max_reached_depth: 0,
            is_stopped: false,
            options_modified: false,
            test_positions: Vec::new(),
            rnd: Random::new_with_seed((Instant::now().elapsed().as_micros() & 0xFFFFFFFFFFFFFFFF) as u64)
        }
    }

    pub fn new(rx: Receiver<Message>) -> Self {
        Engine::new_from_fen(rx, START_POS, DEFAULT_SIZE_MB)
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

            Message::IsReady => self.is_ready(),

            Message::Go { depth, wtime, btime, winc, binc, movetime, movestogo, nodes } =>
                self.go(depth, wtime, btime, winc, binc, movetime, movestogo, nodes),

            Message::PrepareEval(fens) => self.prepare_eval(fens),

            Message::PrepareQuiet(fens) => self.prepare_quiet(fens),

            Message::Eval(k) => self.eval(k),

            Message::Fen => println!("{}", write_fen(&self.board)),

            Message::PrintTestPositions => self.print_test_positions(),

            Message::ResetTestPositions => self.reset_test_positions(),

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
            }

            Message::Stop => ()
        }

        true
    }

    fn go(&mut self, depth: i32, wtime: i32, btime: i32, winc: i32, binc: i32, movetime: i32, movestogo: i32, nodes: u64) {
        self.depth_limit = depth;
        self.timelimit_ms = if self.board.active_player() == WHITE {
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

        let is_strict_timelimit = movetime > 0 || (time_left - (TIMEEXT_MULTIPLIER * self.timelimit_ms) <= 20) || movestogo == 1;

        let m = self.find_best_move(3, is_strict_timelimit);
        if m == NO_MOVE {
            println!("bestmove 0000")
        } else {
            println!(
                "bestmove {}",
                UCIMove::from_encoded_move(&self.board, m).to_uci()
            );
        }
    }

    fn is_ready(&mut self) {
        if self.options_modified {
            self.options_modified = false;
            self.board.pst.recalculate(&self.board.options);
        }
        println!("readyok")
    }

    fn set_position(&mut self, fen: String, moves: Vec<UCIMove>) {
        match read_fen(&mut self.board, &fen) {
            Ok(_) => (),
            Err(err) => println!("position cmd: {}", err),
        }

        for m in moves {
            self.board.perform_move(m.to_move(&self.board));
        }
    }

    fn prepare_eval(&mut self, fens_with_result: Vec<(String, f64)>) {
        for (fen, result) in fens_with_result {
            match read_fen(&mut self.board, &fen) {
                Ok(_) => (),
                Err(err) => println!("prepare_eval cmd: {}", err),
            }

            if self.board.is_in_check(WHITE) || self.board.is_in_check(BLACK) {
                continue;
            }

            let mut pieces: [i8; 64] = [0; 64];
            for i in 0..64 {
                pieces[i] = self.board.get_item(i as i32);
            }


            self.test_positions.push(EvalBoardPos {
                result,
                pieces,
                halfmove_count: self.board.fullmove_count(),
                castling_state: self.board.get_castling_state(),
                is_quiet: true
            });
        }

        println!("prepared");
    }

    fn prepare_quiet(&mut self, fens_with_result: Vec<(String, f64)>) {
        for (fen, result) in fens_with_result {
            match read_fen(&mut self.board, &fen) {
                Ok(_) => (),
                Err(err) => println!("prepare_quiet cmd: {}", err),
            }

            if self.board.is_in_check(-self.board.active_player()) {
                continue;
            }

            let play_moves = (self.rnd.rand64() % 12) as i32;
            for _ in 0..play_moves {
                let m = self.find_best_move(9, true);
                if m == NO_MOVE {
                    break;
                }

                self.board.perform_move(m);
            }

            if !self.make_quiet() {
                continue;
            }

            let mut pieces: [i8; 64] = [0; 64];
            for i in 0..64 {
                pieces[i] = self.board.get_item(i as i32);
            }

            let pos = EvalBoardPos {
                result,
                pieces,
                halfmove_count: self.board.halfmove_count,
                castling_state: self.board.get_castling_state(),
                is_quiet: true
            };

            self.test_positions.push(pos);
        }

        println!("prepared");
    }

    fn make_quiet(&mut self) -> bool {
        for _ in 0..15 {
            if self.board.is_in_check(-self.board.active_player()) {
                return false;
            }

            if self.board.get_static_score().abs() > get_piece_value(Q as usize) as i32 {
                return false;
            }

            let mut is_quiet = self.is_quiet_position();
            if !is_quiet && self.make_quiet_position() && self.is_quiet_position() && self.board.get_static_score().abs() <= get_piece_value(Q as usize) as i32 {
                is_quiet = true;
            }

            let m = self.find_best_move(6, true);
            if m == NO_MOVE {
                return false;
            }

            if is_quiet && self.is_quiet_pv(m, 4) {
                return true;
            }

            self.board.perform_move(m);
        }

        false
    }

    fn eval(&mut self, k: f64) {

        let mut errors: f64 = 0.0;
        let k_div = k / 400.0;
        for pos in self.test_positions.to_vec().iter() {
            pos.apply(&mut self.board);
            let score = if pos.is_quiet {
                self.board.get_score()
            } else {
                self.quiescence_search(self.board.active_player(), MIN_SCORE, MAX_SCORE, 0) * self.board.active_player() as i32
            };

            let win_probability = 1.0 / (1.0 + 10.0f64.powf(-(score as f64) * k_div));
            let error = pos.result - win_probability;
            errors += error * error;
        }

        println!("result {}:{}", self.test_positions.len(), errors);
    }

    fn print_test_positions(&mut self) {

        print!("testpositions ");
        let mut is_first = true;
        for pos in self.test_positions.to_vec().iter() {
            pos.apply(&mut self.board);
            self.board.reset_half_move_clock();
            let fen = write_fen(&self.board);

            if !is_first {
                print!(";");
            } else {
                is_first = false;
            }

            print!("{}", fen);
        }

        println!();
    }

    fn reset_test_positions(&mut self) {
        self.test_positions.clear();
        println!("reset completed");
    }

    fn set_tt_size(&mut self, size_mb: i32) {
        self.tt.resize(size_mb as u64, false);
    }

    fn reset(&mut self) {
        self.tt.clear();
        self.hh.clear();
    }

    fn perft(&mut self, depth: i32) {
        let start = SystemTime::now();

        let nodes = perft(&mut self.movegen, &mut self.hh, &mut self.board, depth);

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
        return i32::max_value()
    }

    if movetime > 0 {
        return max(0, movetime - TIME_SAFETY_MARGIN_MS);
    }

    let time_for_move = time_left / max(1, movestogo);
    let time_bonus = if movestogo > 1 { time_increment / 2 } else { 0 };

    if time_for_move + time_bonus + TIME_SAFETY_MARGIN_MS >= time_left {
        max(0, time_for_move - TIME_SAFETY_MARGIN_MS)
    } else {
        max(0, time_for_move + time_increment)
    }
}
