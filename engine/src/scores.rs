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

use crate::transposition_table::MAX_DEPTH;

pub const MIN_SCORE: i32 = -8191;
pub const MAX_SCORE: i32 = 8191;

pub const MATED_SCORE: i32 = -8000;
pub const MATE_SCORE: i32 = 8000;

pub fn is_mate_score(score: i32) -> bool {
    score.abs() > MATE_SCORE - MAX_DEPTH as i32 * 2
}

pub fn sanitize_eval_score(score: i32) -> i32 {
    score.min(MATE_SCORE - 1000).max(MATED_SCORE + 1000)
}

pub fn sanitize_score(score: i32) -> i32 {
    score.min(MATE_SCORE).max(MATED_SCORE)
}
