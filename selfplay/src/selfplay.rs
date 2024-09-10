/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
use std::sync::{mpsc, Arc};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Instant;
use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, START_POS};
use velvet::moves::{Move, NO_MOVE};
use velvet::search::Search;
use velvet::time_management::SearchLimits;
use velvet::transposition_table::TranspositionTable;

pub struct SearchControl {
    search: Search,
    limits: SearchLimits,
    time: i32,
    inc: i32,
    _tx: Sender<Message>,
    rx: Receiver<Message>,
    depths: usize,
    depth_count: usize,
}

impl SearchControl {
    pub fn new(time: i32, inc: i32) -> SearchControl {
        let mut board = create_from_fen(START_POS);
        board.reset_nn_eval();

        let (tx, rx) = mpsc::channel::<Message>();

        let search = Search::new(
            Arc::new(AtomicBool::new(true)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
            LogLevel::Error,
            SearchLimits::default(),
            TranspositionTable::new(16),
            board.clone(),
            false,
        );

        let limits = SearchLimits::new(None, None, Some(time), Some(inc), Some(inc), Some(inc), None, None, None).expect("Invalid search limits");

        SearchControl {
            search,
            limits,
            time,
            inc,
            _tx: tx,
            rx,
            depths: 0,
            depth_count: 0,
        }
    }

    pub fn reset(&mut self) {
        self.depths = 0;
        self.depth_count = 0;
    }

    pub fn new_game(&mut self, board: &Board, time: i32, inc: i32) {
        self.search.clear_tt();
        self.search.update(board, self.limits, false);
        self.time = time;
        self.inc = inc;
        self.depths = 0;
        self.depth_count = 0;
    }
    
    pub fn set_params(&mut self, params: &[(String, i16)]) {
        for (name, value) in params.iter() {
            if !self.search.set_param(name, *value) {
                panic!("Invalid feature option: {}", name);
            }
        }
    }

    // Returns the best move and a boolean indicating if the time was exceeded
    pub fn next_move(&mut self) -> (Move, bool) {
        let active_player = self.search.board.active_player();

        self.limits = SearchLimits::new(None, None, Some(self.time), Some(self.time), Some(self.inc), Some(self.inc), None, None, None).expect("Invalid search limits");
        self.limits.update(active_player, 20);

        self.search.update_limits(self.limits);
        let start = Instant::now();
        let (bm, _) = self.search.find_best_move_with_full_strength(Some(&self.rx), &[]);
        let duration = start.elapsed().as_millis() as i32;
        self.time -= duration;
        let time_loss = self.time < 0;
        self.time += self.inc;

        if self.search.max_reached_depth > 0 {
            self.depth_count += 1;
            self.depths += self.search.max_reached_depth;
        }

        (bm, time_loss)
    }
    
    pub fn perform_move(&mut self, m: Move) {
        self.search.board.perform_move(m);
    }

    pub fn avg_depth(&self) -> usize {
        if self.depth_count == 0 {
            return 0;
        }
        self.depths / self.depth_count
    }
}

pub enum Outcome {
    Win,
    Loss,
    Draw,
}

impl Outcome {
    pub fn invert(&self) -> Outcome {
        match self {
            Outcome::Win => Outcome::Loss,
            Outcome::Loss => Outcome::Win,
            Outcome::Draw => Outcome::Draw,
        }
    }
}

pub fn play_match(board: &mut Board, engines: &mut [&mut SearchControl; 2], time: i32, inc: i32) -> (Outcome, usize) {
    let mut time_losses = 0;
    loop {
        engines[0].new_game(board, time, inc);
        engines[1].new_game(board, time, inc);

        let mut i = 0;
        loop {
            let (bm, time_loss) = engines[i].next_move();
            if bm == NO_MOVE || time_loss {
                if time_loss {
                    time_losses += 1;
                    break;
                }
                return (if i == 0 { Outcome::Loss } else { Outcome::Win }, time_losses);
            }

            board.perform_move(bm);
            if board.is_insufficient_material_draw() || board.is_repetition_draw() || board.is_fifty_move_draw() {
                return (Outcome::Draw, time_losses);
            }

            engines[i].perform_move(bm);
            i = (i + 1) % 2;
            engines[i].perform_move(bm);
        }
    }
}

