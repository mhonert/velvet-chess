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

use std::sync::mpsc::{Receiver, Sender, RecvError};
use std::sync::mpsc;
use std::thread;
use crate::fen::{read_fen, START_POS, create_from_fen};
use crate::board::Board;
use std::time::{SystemTime, Instant};
use crate::perft::perft;
use crate::colors::WHITE;
use std::cmp::max;
use crate::search::Search;
use crate::history_heuristics::HistoryHeuristics;
use crate::move_gen::NO_MOVE;
use crate::uci_move::UCIMove;
use crate::pieces::EMPTY;

pub enum Message {
    SetPosition(String, Vec<UCIMove>),
    Go{depth: i32, wtime: i32, btime: i32, winc: i32, binc: i32, movetime: i32, movestogo: i32},
    Perft(i32),
    Quit()
}

pub struct Engine {
    pub rx: Receiver<Message>,
    pub board: Board,
    pub hh: HistoryHeuristics,

    pub starttime: Instant,
    pub timelimit_ms: i32,
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
    pub fn new(rx: Receiver<Message>) -> Self {
        Engine{rx,
            board: create_from_fen(START_POS),
            hh: HistoryHeuristics::new(),
            starttime: Instant::now(),
            timelimit_ms: 0}
    }

    fn start_loop(&mut self) {
        loop {
            match self.rx.recv() {
                Ok(msg) => {
                    if !self.handle_message(msg) {
                        return;
                    }
                },
                Err(err) => {
                    println!("Engine communication error: {:?}", err);
                    return;
                }
            }
        }
    }

    fn handle_message(&mut self, msg: Message) -> bool {
        match msg {
            Message::SetPosition(fen, moves) => self.set_position(fen, moves),

            Message::Perft(depth) => self.perft(depth),

            Message::Go{depth, wtime, btime, winc, binc, movetime, movestogo}
                => self.go(depth, wtime, btime, winc, binc, movetime, movestogo),

            Message::Quit() => {
                return false;
            }
        }

        true
    }

    fn go(&mut self, depth: i32, wtime: i32, btime: i32, winc: i32, binc: i32, movetime: i32, movestogo: i32) {
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

        let is_strict_timelimit = movetime != 0 || (time_left - (TIMEEXT_MULTIPLIER * self.timelimit_ms) <= 10);
        let m = self.find_best_move(depth, is_strict_timelimit);
        if m == NO_MOVE {
            println!("bestmove 0000")
        } else {
            println!("bestmove {}", UCIMove::from_encoded_move(&self.board, m).to_uci());
        }
    }

    fn set_position(&mut self, fen: String, moves: Vec<UCIMove>) {
        match read_fen(&mut self.board, &fen) {
            Ok(_) => (),
            Err(err) => println!("position cmd: {}", err)
        }

        for m in moves {
            let color = self.board.active_player();
            let piece = if m.promotion != EMPTY { m.promotion } else {self.board.get_item(m.start as i32)};
            self.board.perform_move(piece * color, m.start as i32, m.end as i32);
        }
    }

    fn perft(&mut self, depth: i32) {
        let start = SystemTime::now();

        let nodes = perft(&mut self.board, depth);

        let duration = match SystemTime::now().duration_since(start) {
            Ok(v) => v,
            Err(e) => panic!("error calculating duration: {:?}", e)
        };

        println!("Nodes: {}", nodes);
        println!("Duration: {:?}", duration);

        let duration_micro = duration.as_micros();
        if duration_micro > 0 {
            let nodes_per_sec = nodes * 1_000_000 / duration_micro as u64;
            println!("Nodes per second: {}", nodes_per_sec);
        }
    }
}

fn calc_timelimit(movetime: i32, time_left: i32, time_increment: i32, movestogo: i32) -> i32 {
    if movetime > 0 {
        return movetime;
    }

    let time_for_move = time_left / max(1, movestogo);
    let time_bonus = time_increment / 2;

    if time_for_move + time_bonus >= time_left {
        max(0, time_for_move)
    } else {
        max(0, time_for_move + time_increment)
    }
}


