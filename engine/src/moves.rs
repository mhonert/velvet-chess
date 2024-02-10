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

use std::fmt;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::intrinsics::transmute;

#[repr(u8)]
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub enum MoveType {
    #[default]
    Unused = 0b00_000,
    TableBaseMarker = 0b01_000,
    Unused2 = 0b11_000,

    PawnQuiet   = 0b00_001,
    PawnCapture = 0b10_001,
    PawnDoubleQuiet = 0b01_001,
    PawnEnPassant = 0b11_001,

    KnightQuiet = 0b00_010,
    KnightCapture = 0b10_010,
    KnightQuietPromotion = 0b01_010,
    KnightCapturePromotion = 0b11_010,

    BishopQuiet = 0b00_011,
    BishopCapture = 0b10_011,
    BishopQuietPromotion = 0b01_011,
    BishopCapturePromotion = 0b11_011,

    RookQuiet = 0b00_100,
    RookCapture = 0b10_100,
    RookQuietPromotion = 0b01_100,
    RookCapturePromotion = 0b11_100,

    QueenQuiet = 0b00_101,
    QueenCapture = 0b10_101,
    QueenQuietPromotion = 0b01_101,
    QueenCapturePromotion = 0b11_101,

    KingQuiet = 0b00_110,
    KingCapture = 0b10_110,
    KingQSCastling = 0b01_110,
    KingKSCastling = 0b11_110,

    // Only relevant for TTPackedMove:
    QueenQuiet8 = 0b00_111,
    QueenCapture8 = 0b10_111,

    Unused3 = 0b01_111,
    Unused4 = 0b11_111,
}

impl MoveType {
    pub fn new_quiet_promotion(piece_id: i8) -> MoveType {
        unsafe { transmute(0b01_000 | piece_id as u8) }
    }

    pub fn new_capture_promotion(piece_id: i8) -> MoveType {
        unsafe { transmute(0b11_000 | piece_id as u8) }
    }

    pub fn new_quiet(piece_id: i8) -> MoveType {
        unsafe { transmute(piece_id as u8) }
    }

    pub fn new_capture(piece_id: i8) -> MoveType {
        unsafe { transmute(0b10_000 | piece_id as u8) }
    }

    #[inline]
    pub fn piece_id(self) -> i8 {
        ((self as u8) & 0b111) as i8
    }

    #[inline]
    pub fn is_capture(self) -> bool {
        matches!(self,
            MoveType::PawnCapture | MoveType::PawnEnPassant | MoveType::KnightCapture | MoveType::BishopCapture | MoveType::RookCapture |
            MoveType::QueenCapture| MoveType::KingCapture |
            MoveType::KnightCapturePromotion | MoveType::BishopCapturePromotion | MoveType::RookCapturePromotion | MoveType::QueenCapturePromotion
        )
    }

    #[inline]
    pub fn is_quiet(self) -> bool {
        matches!(self,
            MoveType::PawnQuiet | MoveType::PawnDoubleQuiet | MoveType::KnightQuiet | MoveType::BishopQuiet | MoveType::RookQuiet |
            MoveType::QueenQuiet| MoveType::KingQuiet | MoveType::KingKSCastling | MoveType::KingQSCastling
        )
    }

    #[inline]
    pub fn is_promotion(self) -> bool {
        matches!(self,
            MoveType::KnightQuietPromotion | MoveType::KnightCapturePromotion |
            MoveType::BishopQuietPromotion | MoveType::BishopCapturePromotion |
            MoveType::RookQuietPromotion | MoveType::RookCapturePromotion |
            MoveType::QueenQuietPromotion | MoveType::QueenCapturePromotion
        )
    }

    #[inline]
    pub fn is_en_passant(self) -> bool {
        matches!(self, MoveType::PawnEnPassant)
    }

    #[inline]
    pub fn is_queen_promotion(self) -> bool {
        matches!(self, MoveType::QueenQuietPromotion | MoveType::QueenCapturePromotion)
    }

    #[inline]
    pub fn is_tb_move(self) -> bool {
        matches!(self, MoveType::TableBaseMarker)
    }
}


#[derive(Copy, Clone, Default)]
pub struct Move(u32);

// Move:
//    31       23       15       7      0
// 0b SSSSSStt tttEEEEE Esssssss ssssssss => 32 Bits
//

const TYPE_SHIFT: u32 = 21;
const TYPE_MASK: u32 = 0b11111;

const START_SHIFT: u32 = 26;
const START_MASK: u32 = 0b111111;

const END_SHIFT: u32 = 15;
const END_MASK: u32 = 0b111111;

const MOVE_ONLY_MASK: u32 =  0b11111111111111111000000000000000;

const SCORE_MASK: u32 = 0b111111111111111;

impl Hash for Move {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.without_score().0)
    }
}

impl PartialEq for Move {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.is_same_move(*other)
    }
}

impl Eq for Move {}

impl Move {
    #[inline]
    pub const fn new(typ: MoveType, start: i8, end: i8) -> Self {
        Move((typ as u32) << TYPE_SHIFT | (end as u32) << END_SHIFT | (start as u32) << START_SHIFT)
    }

    pub const fn from_u32(value: u32) -> Self {
        Move(value)
    }

    #[inline]
    pub fn with_score(&self, score: i16) -> Move {
        if score < 0 {
            Move((self.0 & MOVE_ONLY_MASK) | (0b100000000000000 | (-score as u32)))
        } else {
            Move((self.0 & MOVE_ONLY_MASK) | (score as u32))
        }
    }

    #[inline]
    // Same as with_score, but assumes that the score part of this move has not yet been set before
    // (saves clearing those bits)
    pub fn with_initial_score(&self, score: i16) -> Move {
        if score < 0 {
            Move(self.0 | (0b100000000000000 | (-score as u32)))
        } else {
            Move(self.0 | (score as u32))
        }
    }

    #[inline]
    // Calculates an index in the range 0..512 based upon the (target) piece and the end position of the move
    pub fn calc_piece_end_index(&self) -> usize {
        ((self.0 >> END_SHIFT) & 0b111111111) as usize
    }

    pub fn start(&self) -> i8 {
        ((self.0 >> START_SHIFT) & START_MASK) as i8
    }

    pub fn end(&self) -> i8 {
        ((self.0 >> END_SHIFT) & END_MASK) as i8
    }

    #[inline]
    pub fn without_score(&self) -> Move {
        Move(self.0 & MOVE_ONLY_MASK)
    }

    #[inline]
    pub fn is_capture(&self) -> bool {
        self.move_type().is_capture()
    }

    #[inline]
    pub fn is_quiet(&self) -> bool {
        self.move_type().is_quiet()
    }

    #[inline]
    pub fn is_promotion(&self) -> bool {
        self.move_type().is_promotion()
    }

    #[inline]
    pub fn move_type(&self) -> MoveType {
        unsafe { transmute(((self.0 >> TYPE_SHIFT) & TYPE_MASK) as u8) }
    }

    #[inline]
    pub fn is_queen_promotion(&self) -> bool {
        self.move_type().is_queen_promotion()
    }

    /// Checks, whether the two moves are the same (except for the score)
    #[inline]
    fn is_same_move(&self, m: Move) -> bool {
        (self.0 & MOVE_ONLY_MASK) == (m.0 & MOVE_ONLY_MASK)
    }

    #[inline]
    pub fn score(&self) -> i16 {
        if self.0 & 0b100000000000000 != 0 {
            -((self.0 & 0b011111111111111) as i16)
        } else {
            (self.0 & SCORE_MASK) as i16
        }
    }

    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.0
    }

    pub fn is_no_move(&self) -> bool {
        self.0 == 0 || self.move_type().is_tb_move()
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Move")
            .field("target-piece", &self.move_type().piece_id())
            .field("start", &self.start())
            .field("end", &self.end())
            .field("score", &self.score())
            .field("type", &self.move_type())
            .finish()
    }
}

pub const NO_MOVE: Move = Move(0);

#[cfg(test)]
mod tests {
    use crate::moves::MoveType::*;
    use super::*;
    use crate::scores::{MAX_SCORE, MIN_SCORE};

    #[test]
    fn move_score_encoding() {
        for score in [MIN_SCORE, -1, 0, 1, MAX_SCORE] {

            let m = Move::new(MoveType::KingQuiet, 4, 12).with_score(score);

            assert_eq!(m.move_type(), MoveType::KingQuiet);
            assert_eq!(m.start(), 4);
            assert_eq!(m.end(), 12);
            assert_eq!(m.score(), score);
        }
    }

    #[test]
    fn move_type() {
        for move_type in [
            PawnQuiet, PawnCapture, PawnEnPassant,
            KnightQuiet, KnightCapture, KnightQuietPromotion, KnightCapturePromotion,
            BishopQuiet, BishopCapture, BishopQuietPromotion, BishopCapturePromotion,
            RookQuiet, RookCapture, RookQuietPromotion, RookCapturePromotion,
            QueenQuiet, QueenCapture, QueenQuietPromotion, QueenCapturePromotion,
            KingQuiet, KingCapture, KingQSCastling, KingKSCastling,
        ] {
            let m = Move::new(move_type, 63, 63).with_score(MIN_SCORE);

            assert_eq!(m.move_type() as u8, move_type as u8);
            assert_eq!(m.start(), 63);
            assert_eq!(m.end(), 63);
            assert_eq!(m.score(), MIN_SCORE);
        }
    }

    #[test]
    fn position() {
        for start in 0..64 {
            for end in 0..64 {
                for piece in 1..=6 {
                    let move_type = MoveType::new_quiet(piece);
                    let m = Move::new(move_type, start, end).with_score(MIN_SCORE);

                    assert_eq!(m.move_type() as u8, move_type as u8);
                    assert_eq!(m.move_type().piece_id(), piece);
                    assert_eq!(m.start(), start);
                    assert_eq!(m.end(), end);
                    assert_eq!(m.score(), MIN_SCORE);
                }
            }
        }
    }
}
