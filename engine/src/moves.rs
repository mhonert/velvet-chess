/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use crate::moves::MoveType::PawnSpecial;
use crate::pieces::{P, Q};
use std::fmt;
use std::intrinsics::transmute;

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum MoveType {
    PawnQuiet = 0,
    Quiet = 1,
    PawnDoubleQuiet = 2,
    KingQuiet = 3,
    Capture = 4,
    KingCapture = 5,
    PawnSpecial = 6, // En Passant or Promotion
    Castling = 7,
}

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Move(u32);

const PIECE_MASK: u32 = 0b111;

const TYPE_SHIFT: u32 = 29;
const TYPE_MASK: u32 = 0b111 << TYPE_SHIFT;

const END_SHIFT: u32 = 3;
const END_MASK: u32 = 0b111111 << END_SHIFT;

const START_SHIFT: u32 = 9;
const START_MASK: u32 = 0b111111 << START_SHIFT;

const MOVE_ONLY_MASK: u32 = 0b11100000000000000111111111111111;

const SCORE_ONLY_MASK: u32 = !MOVE_ONLY_MASK;

const SCORE_SHIFT: u32 = 15;

impl Move {
    #[inline]
    pub const fn new(typ: MoveType, piece_id: i8, start: i32, end: i32) -> Self {
        Move(
            (piece_id as u32)
                | ((end as u32) << END_SHIFT)
                | ((start as u32) << START_SHIFT)
                | ((typ as u32) << TYPE_SHIFT),
        )
    }

    #[inline]
    pub fn with_score(&self, score: i32) -> Move {
        if score < 0 {
            Move((self.0 & MOVE_ONLY_MASK) | ((0b10000000000000 | (-score as u32)) << SCORE_SHIFT))
        } else {
            Move((self.0 & MOVE_ONLY_MASK) | ((score as u32) << SCORE_SHIFT))
        }
    }

    #[inline]
    pub fn with_typ(&self, typ: MoveType) -> Move {
        Move(((typ as u32) << TYPE_SHIFT) | (self.0 & !TYPE_MASK))
    }

    #[inline]
    // Calculates an index in the range 0..512 based upon the (target) piece and the end position of the move
    pub fn calc_piece_end_index(&self) -> usize {
        (self.0 & (PIECE_MASK | END_MASK)) as usize
    }

    #[inline]
    pub fn to_bit29(&self) -> u32 {
        self.0 & 0b00011111111111111111111111111111
    }

    #[inline]
    pub fn from_bit29(packed_move: u32) -> Move {
        Move(packed_move)
    }

    #[inline]
    pub fn without_score(&self) -> Move {
        self.with_score(0)
    }

    /// Checks, whether the two moves are the same (except for the score)
    #[inline]
    pub fn is_same_move(&self, m: Move) -> bool {
        (self.0 & MOVE_ONLY_MASK) == (m.0 & MOVE_ONLY_MASK)
    }

    /// Checks, whether the two moves are the same (except for the type and score)
    #[inline]
    pub fn is_same_untyped_move(&self, m: Move) -> bool {
        (self.0 & (MOVE_ONLY_MASK & !TYPE_MASK)) == (m.0 & (MOVE_ONLY_MASK & !TYPE_MASK))
    }

    #[inline]
    pub fn typ(&self) -> MoveType {
        unsafe { transmute((self.0 >> TYPE_SHIFT) as u8) }
    }

    #[inline]
    pub fn piece_id(&self) -> i8 {
        (self.0 & PIECE_MASK) as i8
    }

    #[inline]
    pub fn start(&self) -> i32 {
        ((self.0 & START_MASK) >> START_SHIFT) as i32
    }

    #[inline]
    pub fn end(&self) -> i32 {
        ((self.0 & END_MASK) >> END_SHIFT) as i32
    }

    #[inline]
    pub fn score(&self) -> i32 {
        if self.0 & (0b10000000000000 << SCORE_SHIFT) != 0 {
            -(((self.0 & (0b01111111111111 << SCORE_SHIFT)) >> SCORE_SHIFT) as i32)
        } else {
            ((self.0 & SCORE_ONLY_MASK) >> SCORE_SHIFT) as i32
        }
    }

    #[inline]
    pub fn is_promotion(&self) -> bool {
        self.typ() as u8 == PawnSpecial as u8 && self.piece_id() != P
    }

    #[inline]
    pub fn is_queen_promotion(&self) -> bool {
        self.typ() as u8 == PawnSpecial as u8 && self.piece_id() == Q
    }

    #[inline]
    pub fn is_en_passant(&self) -> bool {
        self.typ() as u8 == PawnSpecial as u8 && self.piece_id() == P
    }

    #[inline]
    pub fn is_quiet(&self) -> bool {
        matches!(
            self.typ(),
            MoveType::PawnQuiet
                | MoveType::Quiet
                | MoveType::PawnDoubleQuiet
                | MoveType::KingQuiet
                | MoveType::Castling
        )
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Move")
            .field("target-piece", &self.piece_id())
            .field("start", &self.start())
            .field("end", &self.end())
            .field("score", &self.score())
            .field("type", &(self.typ() as u8))
            .finish()
    }
}

pub const NO_MOVE: Move = Move(0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pieces::{K, Q};
    use crate::scores::{MAX_SCORE, MIN_SCORE};

    #[test]
    fn scored_move() {
        let score = -1037;
        let m = Move::new(MoveType::KingQuiet, K, 4, 12).with_score(score);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 4);
        assert_eq!(m.end(), 12);
        assert_eq!(score, m.score());
    }

    #[test]
    fn scored_move_for_max_score() {
        let m = Move::new(MoveType::Quiet, Q, 2, 63).with_score(MAX_SCORE);

        assert_eq!(m.piece_id(), Q);
        assert_eq!(m.start(), 2);
        assert_eq!(m.end(), 63);
        assert_eq!(MAX_SCORE, m.score());
    }

    #[test]
    fn scored_move_for_min_score() {
        let m = Move::new(MoveType::KingQuiet, K, 0, 1).with_score(MIN_SCORE);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 1);
        assert_eq!(MIN_SCORE, m.score());
    }

    #[test]
    fn move_type() {
        for &typ in [
            MoveType::Quiet,
            MoveType::PawnQuiet,
            MoveType::PawnDoubleQuiet,
            MoveType::KingQuiet,
            MoveType::Capture,
            MoveType::KingCapture,
            MoveType::PawnSpecial,
            MoveType::Castling,
        ]
        .iter()
        {
            let m = Move::new(typ, K, 63, 63).with_score(MIN_SCORE);

            assert_eq!(typ as u8, m.typ() as u8);
            assert_eq!(m.piece_id(), K);
            assert_eq!(m.start(), 63);
            assert_eq!(m.end(), 63);
            assert_eq!(m.score(), MIN_SCORE);
        }
    }
}
