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

pub const MAX_SCORE: i16 = 16383;
pub const MIN_SCORE: i16 = -MAX_SCORE;

pub const MATE_SCORE: i16 = 13000;
pub const MATED_SCORE: i16 = -MATE_SCORE;
const MATE_SCORE_RANGE: i16 = 499;

pub const MAX_EVAL: i16 = 9999;
pub const MIN_EVAL: i16 = -MAX_EVAL;

pub fn is_mate_or_mated_score(score: i16) -> bool {
    score.abs() >= (MATE_SCORE - MATE_SCORE_RANGE)
}

pub fn is_mate_score(score: i16) -> bool {
    score >= (MATE_SCORE - MATE_SCORE_RANGE)
}

pub fn is_mated_score(score: i16) -> bool {
    score <= (MATED_SCORE + MATE_SCORE_RANGE)
}

pub fn is_eval_score(score: i16) -> bool {
    score.abs() <= MAX_EVAL
}

pub fn mate_in(score: i16) -> Option<i16> {
    let mate_ply_distance = MATE_SCORE - score;
    if (0..=MATE_SCORE_RANGE).contains(&mate_ply_distance) {
        Some((mate_ply_distance + 1) / 2)
    } else {
        None
    }
}

pub fn sanitize_score(score: i16) -> i16 {
    let tmp = score.clamp(MATED_SCORE, MATE_SCORE);
    if is_mate_or_mated_score(tmp) {
        tmp
    } else {
        tmp.clamp(MIN_EVAL, MAX_EVAL)
    }
}

pub fn sanitize_eval_score(score: i32) -> i32 {
    score.clamp(MIN_EVAL as i32, MAX_EVAL as i32)
}

pub fn sanitize_mate_score(score: i16) -> i16 {
    score.clamp(MATE_SCORE - MATE_SCORE_RANGE, MATE_SCORE)
}

pub fn sanitize_mated_score(score: i16) -> i16 {
    score.clamp(MATED_SCORE, MATED_SCORE + MATE_SCORE_RANGE)
}

// clock_scaled_eval scales the evaluation by the remaining halfmove clock (50-move counter).
pub fn clock_scaled_eval(halfmove_clock: u8, is_tb_pos: bool, eval: i16) -> i16 {
    if is_tb_pos {
        return eval;
    }
    ((eval as i32) * (128 - halfmove_clock as i32) / 128) as i16
}

