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

use crate::colors::Color;
use crate::moves::{Move, NO_MOVE};
use crate::transposition_table::MAX_DEPTH;
use std::time::{Duration, Instant};

pub const TIMEEXT_MULTIPLIER: i32 = 3;
pub const MAX_TIMELIMIT_MS: i32 = i32::MAX;

pub const DEFAULT_MOVE_OVERHEAD_MS: i32 = 20;
pub const MAX_MOVE_OVERHEAD_MS: i32 = 1000;
pub const MIN_MOVE_OVERHEAD_MS: i32 = 0;

#[derive(Clone)]
pub struct TimeManager {
    starttime: Instant,
    timelimit_ms: i32,

    allow_time_extension: bool,
    time_extended: bool,

    next_index: usize,
    current_depth: i32,
    history: Vec<Move>,
    
    expected_best_move: Move,
}

impl TimeManager {
    pub fn new() -> Self {
        TimeManager {
            starttime: Instant::now(),
            timelimit_ms: 0,
            allow_time_extension: true,
            time_extended: false,
            next_index: 0,
            current_depth: 0,
            history: vec![NO_MOVE; MAX_DEPTH],
            expected_best_move: NO_MOVE,
        }
    }

    pub fn reset(&mut self, limit: SearchLimits) {
        self.starttime = Instant::now();
        self.timelimit_ms = limit.time_limit_ms;

        self.allow_time_extension = !limit.strict_time_limit;
        self.history.fill(NO_MOVE);
        self.next_index = 0;
        self.current_depth = 0;
        self.time_extended = false;
    }

    pub fn update_best_move(&mut self, new_best_move: Move, depth: i32) {
        if depth > self.current_depth {
            self.current_depth = depth;
            self.next_index += 1;
        }
        self.history[self.next_index - 1] = new_best_move;
    }

    pub fn is_time_for_another_iteration(&self, now: Instant, previous_iteration_time: Duration) -> bool {
        if self.time_extended && !(self.score_dropped() || self.best_move_changed()) {
            return false;
        }

        let duration_ms = previous_iteration_time.as_millis() as i32;
        let estimated_iteration_duration = duration_ms * 7 / 4;
        let remaining_time = self.remaining_time_ms(now);
        if remaining_time <= estimated_iteration_duration * 7 / 4 && self.best_move_consistent() {
            return false;
        }

        remaining_time >= estimated_iteration_duration
    }
    
    fn best_move_consistent(&self) -> bool {
        self.expected_best_move != NO_MOVE && !self.history.is_empty() && self.history.iter().take(self.next_index - 1).all(|&m| m == self.expected_best_move)
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

    pub fn try_extend_timelimit(&mut self) -> bool {
        if !self.allow_time_extension {
            return false;
        }
        
        if self.best_move_changed() || self.score_dropped() {
            self.allow_time_extension = false;
            self.timelimit_ms *= TIMEEXT_MULTIPLIER;
            self.time_extended = true;
            return true;
        }
        
        false
    }

    fn score_dropped(&self) -> bool {
        if self.next_index < 2 {
            false
        } else {
            (self.history[self.next_index - 2].score() - self.history[self.next_index - 1].score()) > 0
        }
    }

    fn best_move_changed(&self) -> bool {
        if self.next_index < 2 {
            false
        } else {
            self.history[self.next_index - 2] != self.history[self.next_index - 1]
        }
    }

    pub fn reduce_timelimit(&mut self) {
        self.allow_time_extension = false;
        self.timelimit_ms /= 32;
    }
    
    pub fn set_expected_best_move(&mut self, m: Move) {
        self.expected_best_move = m;
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SearchLimits {
    node_limit: u64,
    depth_limit: i32,
    mate_limit: i16,
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
            mate_limit: 0,
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

    pub fn infinite() -> SearchLimits {
        let mut limits = SearchLimits::default();
        limits.time_limit_ms = MAX_TIMELIMIT_MS;
        limits
    }

    pub fn nodes(node_limit: u64) -> SearchLimits {
        let mut limits = SearchLimits::default();
        limits.node_limit = node_limit;

        limits
    }

    pub fn new(
        depth_limit: Option<i32>, node_limit: Option<u64>, wtime: Option<i32>, btime: Option<i32>, winc: Option<i32>,
        binc: Option<i32>, move_time: Option<i32>, moves_to_go: Option<i32>, mate_limit: Option<i16>,
    ) -> Result<Self, &'static str> {
        let depth_limit = depth_limit.unwrap_or(MAX_DEPTH as i32);
        if depth_limit <= 0 {
            return Err("depth limit must be > 0");
        }

        let node_limit = node_limit.unwrap_or(u64::MAX);
        let mate_limit = mate_limit.unwrap_or(0);

        Ok(SearchLimits {
            depth_limit,
            node_limit,
            mate_limit,
            time_limit_ms: i32::MAX,
            strict_time_limit: true,

            wtime: wtime.unwrap_or(-1),
            btime: btime.unwrap_or(-1),
            winc: winc.unwrap_or(0),
            binc: binc.unwrap_or(0),
            move_time: move_time.unwrap_or(-1),
            moves_to_go: moves_to_go.unwrap_or(25),
        })
    }

    pub fn update(&mut self, active_player: Color, move_overhead_ms: i32) {
        let (time_left, inc) = if active_player.is_white() { (self.wtime, self.winc) } else { (self.btime, self.binc) };

        self.time_limit_ms = calc_time_limit(self.move_time, time_left, inc, self.moves_to_go, move_overhead_ms);

        self.strict_time_limit = self.move_time > 0
            || self.time_limit_ms == MAX_TIMELIMIT_MS
            || self.moves_to_go == 1
            || (time_left - (TIMEEXT_MULTIPLIER * self.time_limit_ms) <= move_overhead_ms);
    }

    pub fn node_limit(&self) -> u64 {
        self.node_limit
    }

    pub fn set_node_limit(&mut self, limit: u64) {
        self.node_limit = limit;
    }

    pub fn set_depth_limit(&mut self, limit: i32) {
        self.depth_limit = limit;
    }

    pub fn set_strict_time_limit(&mut self, strict: bool) {
        self.strict_time_limit = strict;
    }

    pub fn depth_limit(&self) -> i32 {
        self.depth_limit
    }

    pub fn mate_limit(&self) -> i16 {
        self.mate_limit
    }

    pub fn is_infinite(&self) -> bool {
        self.time_limit_ms == MAX_TIMELIMIT_MS && self.node_limit == u64::MAX && self.mate_limit == 0 && self.depth_limit == MAX_DEPTH as i32
    }

    pub fn has_time_limit(&self) -> bool {
        self.time_limit_ms != MAX_TIMELIMIT_MS
    }
}

fn calc_time_limit(movetime: i32, mut time_left: i32, time_increment: i32, moves_to_go: i32, move_overhead_ms: i32) -> i32 {
    if movetime == -1 && time_left == -1 {
        return MAX_TIMELIMIT_MS;
    }

    if movetime > 0 {
        return (movetime - move_overhead_ms).max(0);
    }

    time_left -= move_overhead_ms;
    if time_left <= 0 {
        return 0;
    }

    let time_for_move = time_left / moves_to_go.max(1);

    if time_for_move > time_left {
        return time_left;
    }

    if time_for_move + time_increment <= time_left {
        time_for_move + time_increment
    } else {
        time_for_move
    }
}
