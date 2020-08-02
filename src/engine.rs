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
use std::time::SystemTime;
use crate::perft::perft;

struct Engine {
    rx: Receiver<Message>,
    board: Board,
}

pub fn spawn_engine_thread() -> Sender<Message> {
    let (tx, rx) = mpsc::channel::<Message>();

    thread::spawn(move || {
        let mut engine = Engine{rx, board: create_from_fen(START_POS)};
        engine.start_loop();
    });

    tx
}

impl Engine {
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
            Message::SetPosition(fen) => self.set_position(fen),

            Message::Perft(depth) => self.perft(depth),

            Message::Quit() => {
                return false;
            }
        }

        true
    }

    fn set_position(&mut self, fen: String) {
        read_fen(&mut self.board, &fen);
        println!("Position set, active player: {}", self.board.active_player());
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


pub enum Message {
    SetPosition(String),
    Perft(i32),
    Quit()
}

