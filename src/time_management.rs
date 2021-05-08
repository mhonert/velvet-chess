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

use crate::moves::{Move, NO_MOVE};
use std::time::{Instant, Duration};
use std::cmp::{max, min};
use crate::transposition_table::MAX_DEPTH;

pub const TIMEEXT_MULTIPLIER: i32 = 5;
pub const MAX_TIMELIMIT_MS: i32 = i32::max_value();

pub struct TimeManager {
    history_size: usize,
    score_drop_threshold: i32,

    starttime: Instant,
    timelimit_ms: i32,
    is_strict_timelimit: bool,

    already_extended_timelimit: bool,

    next_index: usize,
    current_depth: i32,
    history: Vec<Move>,
}

impl TimeManager {
    pub fn new(history_size: i32, score_drop_threshold: i32) -> Self {
        TimeManager{
            history_size: history_size as usize,
            score_drop_threshold,
            starttime: Instant::now(),
            timelimit_ms: 0,
            is_strict_timelimit: true,
            already_extended_timelimit: false,
            next_index: 0,
            current_depth: 0,
            history: vec!(NO_MOVE; MAX_DEPTH),
        }
    }

    pub fn update_params(&mut self, history_size: i32, score_drop_threshold: i32) {
        self.history_size = history_size as usize;
        self.score_drop_threshold = score_drop_threshold;
    }

    pub fn reset(&mut self, timelimit_ms: i32, is_strict_timelimit: bool) {
        self.starttime = Instant::now();
        self.timelimit_ms = timelimit_ms;
        self.is_strict_timelimit = is_strict_timelimit;

        self.already_extended_timelimit = false;
        self.history.fill(NO_MOVE);
        self.next_index = 0;
        self.current_depth = 0;
    }

    pub fn update_best_move(&mut self, new_best_move: Move, depth: i32) {
        if depth > self.current_depth {
            self.current_depth = depth;
            self.next_index += 1;
        }
        self.history[self.next_index - 1] = new_best_move;
    }

    pub fn is_time_for_another_iteration(&self, now: Instant, previous_iteration_time: Duration) -> bool {
        let duration_ms = previous_iteration_time.as_millis() as i32;
        self.remaining_time_ms(now) >= duration_ms * 2
    }

    pub fn search_duration_ms(&self, now: Instant) -> i32 {
        self.search_duration(now).as_millis() as i32
    }

    pub fn search_duration(&self, now: Instant) -> Duration {
        now.duration_since(self.starttime)
    }

    pub fn remaining_time_ms(&self, now: Instant) -> i32 {
        self.timelimit_ms - self.search_duration_ms(now)
    }

    pub fn is_timelimit_exceeded(&self, now: Instant) -> bool {
        self.remaining_time_ms(now) <= 0
    }

    pub fn should_extend_timelimit(&self) -> bool {
        if self.is_strict_timelimit || self.already_extended_timelimit {
            return false;
        }

        let highest_score_drop = self.history.iter()
            .take(self.next_index)
            .rev()
            .take(min(self.next_index, self.history_size))
            .map(Move::score)
            .rfold((0, 0, 0), |(highest_drop, count, prev_score), score|
                if count == 0 {
                    (0, 1, score)
                } else {
                    (max(highest_drop, max(0, prev_score - score)), count + 1, score)
                }
            ).0;

        highest_score_drop >= self.score_drop_threshold
    }

    pub fn extend_timelimit(&mut self) {
        self.already_extended_timelimit = true;
        self.timelimit_ms *= TIMEEXT_MULTIPLIER;
    }
}