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

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct Move(u32);

impl Move {
    #[inline]
    pub fn new(piece: i8, start: i32, end: i32) -> Self {
        Move(((piece.abs() as u32) << 3) | ((start as u32) << 6) | ((end as u32) << 12))
    }

    #[inline]
    pub fn with_score(&self, score: i32) -> Move {
        if score < 0 {
            Move(self.0 & 0x3FFFF | 0x80000000 | ((-score as u32) << 18))
        } else {
            Move(self.0 & 0x3FFFF | (score as u32) << 18)
        }
    }

    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_u32(packed_move: u32) -> Move {
        Move(packed_move)
    }

    #[inline]
    pub fn without_score(&self) -> Move {
        self.with_score(0)
    }

    /// Checks, whether the two moves are the same (except for the score)
    #[inline]
    pub fn is_same_move(&self, m: Move) -> bool {
        (self.0 & 0x3FFFF) == (m.0 & 0x3FFFF)
    }

    #[inline]
    pub fn piece_id(&self) -> i8 {
        ((self.0 >> 3) & 0x7) as i8
    }

    #[inline]
    pub fn start(&self) -> i32 {
        ((self.0 >> 6) & 0x3F) as i32
    }

    #[inline]
    pub fn end(&self) -> i32 {
        ((self.0 >> 12) & 0x3F) as i32
    }

    #[inline]
    pub fn score(&self) -> i32 {
        if self.0 & 0x80000000 != 0 {
            -(((self.0 & 0x7FFC0000) >> 18) as i32)
        } else {
            (self.0 >> 18) as i32
        }
    }
}

pub const NO_MOVE: Move = Move(0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score_util::{MAX_SCORE, MIN_SCORE};
    use crate::pieces::{Q, K};

    #[test]
    fn scored_move() {
        let score = -1037;
        let m = Move::new(K, 4, 12).with_score(score);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 4);
        assert_eq!(m.end(), 12);
        assert_eq!(score, m.score());
    }

    #[test]
    fn scored_move_for_max_score() {
        let m = Move::new(Q, 2, 63).with_score(MAX_SCORE);

        assert_eq!(m.piece_id(), Q);
        assert_eq!(m.start(), 2);
        assert_eq!(m.end(), 63);
        assert_eq!(MAX_SCORE, m.score());
    }

    #[test]
    fn scored_move_for_min_score() {
        let m = Move::new(K, 0, 1).with_score(MIN_SCORE);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 1);
        assert_eq!(MIN_SCORE, m.score());
    }

}
