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

use crate::bitboard::Direction::{AntiDiagonal, Diagonal, Horizontal, Vertical};
use crate::colors::Color;
use std::ops::Not;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct BitBoard(pub u64);

impl Iterator for BitBoard {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.0 == 0 {
            return None;
        }

        let pos = self.0.trailing_zeros();
        self.0 ^= 1 << pos as u64;
        Some(pos)
    }
}

impl BitBoard {
    pub fn piece_count(&self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn is_occupied(&self) -> bool {
        self.0 != 0
    }

    pub fn piece_pos(&self) -> u32 {
        self.0.trailing_zeros()
    }

    pub fn contains(&self, other: BitBoard) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn first(&self) -> BitBoard {
        BitBoard(self.0 & (self.0 as i64).wrapping_neg() as u64)
    }
}

macro_rules! impl_binary_op {
    ($OT:ident, $OP:ident) => {
        impl std::ops::$OT<BitBoard> for BitBoard {
            type Output = BitBoard;

            fn $OP(self, rhs: BitBoard) -> Self::Output {
                BitBoard(self.0.$OP(rhs.0))
            }
        }

        impl std::ops::$OT<u64> for BitBoard {
            type Output = BitBoard;

            fn $OP(self, rhs: u64) -> Self::Output {
                BitBoard(self.0.$OP(rhs))
            }
        }
    };
}

macro_rules! impl_binary_assign_op {
    ($OT:ident, $OP:ident) => {
        impl std::ops::$OT<BitBoard> for BitBoard {
            fn $OP(&mut self, rhs: BitBoard) {
                self.0.$OP(rhs.0);
            }
        }
    };
}

impl_binary_op!(BitOr, bitor);
impl_binary_op!(BitXor, bitxor);
impl_binary_op!(BitAnd, bitand);
impl_binary_op!(Shr, shr);
impl_binary_op!(Shl, shl);

impl_binary_assign_op!(BitOrAssign, bitor_assign);
impl_binary_assign_op!(BitXorAssign, bitxor_assign);
impl_binary_assign_op!(BitAndAssign, bitand_assign);
impl_binary_assign_op!(ShrAssign, shr_assign);
impl_binary_assign_op!(ShlAssign, shl_assign);

impl Not for BitBoard {
    type Output = BitBoard;

    fn not(self) -> Self::Output {
        BitBoard(!self.0)
    }
}

impl From<BitBoard> for u64 {
    fn from(bb: BitBoard) -> Self {
        bb.0
    }
}

#[derive(Copy, Clone, Default)]
pub struct BitBoards([u64; 15]);

const ALL: usize = 6;
const BY_COLOR: usize = 13;

impl BitBoards {
    #[inline(always)]
    pub fn by_piece(&self, piece: i8) -> BitBoard {
        BitBoard(unsafe { *self.0.get_unchecked((piece + 6) as usize) })
    }

    #[inline(always)]
    pub fn by_color(&self, color: Color) -> BitBoard {
        BitBoard(unsafe { *self.0.get_unchecked(BY_COLOR + color.idx()) })
    }

    #[inline(always)]
    pub fn occupancy(&self) -> BitBoard {
        BitBoard(unsafe { *self.0.get_unchecked(ALL) })
    }

    #[inline(always)]
    pub fn flip(&mut self, color: Color, piece: i8, pos: u32) {
        let mask = 1u64 << pos;
        unsafe {
            *self.0.get_unchecked_mut((piece + 6) as usize) ^= mask;
            *self.0.get_unchecked_mut(ALL) ^= mask;
            *self.0.get_unchecked_mut(BY_COLOR + color.idx()) ^= mask;
        }
    }
}

/// Mirrors the given bitboard position index vertically
pub fn v_mirror(pos: usize) -> usize {
    pos ^ 56
}

/// Mirrors the given bitboard position index vertically
pub fn v_mirror_i8(pos: i8) -> i8 {
    pos ^ 56
}

/// Mirrors the given bitboard position index vertically
pub fn v_mirror_u16(pos: u16) -> u16 {
    pos ^ 56
}

/// Mirrors the given bitboard position index horizontally
pub fn h_mirror_i8(pos: i8) -> i8 {
    pos ^ 7
}

/// Mirrors the given bitboard position index horizontally
pub fn h_mirror(pos: usize) -> usize {
    pos ^ 7
}

const KNIGHT_ATTACKS: [u64; 64] = calculate_single_move_patterns([21, 19, 12, 8, -12, -21, -19, -8]);
const KING_ATTACKS: [u64; 64] = calculate_single_move_patterns([1, 10, -1, -10, 9, 11, -9, -11]);

const LINE_MASKS: [LinePatterns; 64 * 4] = calc_line_patterns();

#[inline]
pub fn get_knight_attacks(pos: i32) -> BitBoard {
    BitBoard(unsafe { *KNIGHT_ATTACKS.get_unchecked(pos as usize) })
}

#[inline]
pub fn get_king_attacks(pos: i32) -> BitBoard {
    BitBoard(unsafe { *KING_ATTACKS.get_unchecked(pos as usize) })
}

#[inline]
pub fn gen_bishop_attacks(occupied: u64, pos: i32) -> u64 {
    get_line_attacks(occupied, unsafe { LINE_MASKS.get_unchecked(pos as usize + (Diagonal as usize * 64)) })
        | get_line_attacks(occupied, unsafe { LINE_MASKS.get_unchecked(pos as usize + (AntiDiagonal as usize * 64)) })
}

#[inline]
pub fn gen_rook_attacks(occupied: u64, pos: i32) -> u64 {
    get_line_attacks(occupied, unsafe { LINE_MASKS.get_unchecked(pos as usize + (Horizontal as usize * 64)) })
        | get_line_attacks(occupied, unsafe { LINE_MASKS.get_unchecked(pos as usize + (Vertical as usize * 64)) })
}

#[inline]
pub fn get_column_mask(pos: i32) -> u64 {
    0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001 << (pos & 7) as u64
}

#[inline]
/// Returns a mask where all bits from start to end positions (including start and end) are set
pub fn get_from_to_mask(start: i8, end: i8) -> u64 {
    let s = 1u64 << start;
    let e = 1u64 << end;

    (s - 1) ^ (e - 1) | s | e
}

// Calculate move patterns for pieces which can only move to one target field per direction (king and knight)
const fn calculate_single_move_patterns(directions: [i32; 8]) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];
    let mut index: usize = 0;

    let mut board_pos = 21;
    while board_pos <= 98 {
        if is_border(board_pos) {
            board_pos += 1;
            continue;
        }

        let mut pattern: u64 = 0;
        let mut dir_index = 0;
        while dir_index < directions.len() {
            let dir = directions[dir_index];
            let target_pos = board_pos + dir as i32;
            if !is_border(target_pos) {
                let row = (target_pos - 21) / 10;
                let col = (target_pos - 21) % 10;
                let bit_index = col + (row * 8);
                pattern |= 1 << bit_index as u64;
            }

            dir_index += 1;
        }
        patterns[index] = pattern;
        index += 1;

        board_pos += 1;
    }

    patterns
}

pub fn create_blocker_permutations(permutations: &mut Vec<u64>, prev_blockers: u64, blockers: u64) {
    permutations.push(blockers | prev_blockers);

    let mut rem_blockers = blockers;
    while rem_blockers != 0 {
        let pos = rem_blockers.trailing_zeros();
        rem_blockers ^= 1 << pos as u64;
        create_blocker_permutations(permutations, (prev_blockers | blockers) & ((1 << pos as u64) - 1), rem_blockers);
    }
}

pub fn mask_without_outline(mut mask: u64, pos: u32) -> u64 {
    if pos & 7 > 0 {
        mask &= !0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001;
    }

    if pos & 7 < 7 {
        mask &= !0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000;
    }

    if pos / 8 > 0 {
        mask &= !0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_11111111;
    }

    if pos / 8 < 7 {
        mask &= !0b11111111_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
    }

    mask
}

// Patterns for line movers (Bishop, Rook, Queen)
#[repr(usize)]
enum Direction {
    Horizontal = 0,
    Vertical = 1,
    Diagonal = 2,
    AntiDiagonal = 3,
}

const MAX_FIELD_DISTANCE: i32 = 7; // maximum distance between two fields on the board

const DIRECTIONS: [usize; 4] = [Horizontal as usize, Vertical as usize, Diagonal as usize, AntiDiagonal as usize];

#[derive(Copy, Clone)]
struct LinePatterns {
    lower: u64,
    upper: u64,
    combined: u64,
}

#[inline]
fn get_line_attacks(occupied: u64, patterns: &LinePatterns) -> u64 {
    // Uses the obstruction difference algorithm to determine line attacks
    // see the chess programming Wiki for a detailed explanation: https://www.chessprogramming.org/Obstruction_Difference
    let lower = patterns.lower & occupied;
    let upper = patterns.upper & occupied;
    let ms1b = 0x8000000000000000 >> ((lower | 1).leading_zeros() as u64);
    let ls1b = upper & upper.wrapping_neg();
    let odiff = ls1b.wrapping_shl(1).wrapping_sub(ms1b);
    patterns.combined & odiff
}

const DIRECTION_COL_OFFSET: [i32; 4] = [-1, 0, 1, -1];
const DIRECTION_ROW_OFFSET: [i32; 4] = [0, -1, -1, -1];

const fn calc_line_patterns() -> [LinePatterns; 64 * 4] {
    let mut patterns: [LinePatterns; 64 * 4] = [LinePatterns { lower: 0, upper: 0, combined: 0 }; 64 * 4];

    let mut index = 0;
    let mut dir_index = 0;
    while dir_index < DIRECTIONS.len() {
        let dir = DIRECTIONS[dir_index];
        let mut pos = 0;
        while pos < 64 {
            let lower = calc_pattern(pos, DIRECTION_COL_OFFSET[dir], DIRECTION_ROW_OFFSET[dir]);
            let upper = calc_pattern(pos, -DIRECTION_COL_OFFSET[dir], -DIRECTION_ROW_OFFSET[dir]);
            let combined = upper | lower;
            patterns[index] = LinePatterns { lower, upper, combined };
            index += 1;
            pos += 1;
        }
        dir_index += 1;
    }

    patterns
}

const fn calc_pattern(pos: i32, dir_col: i32, dir_row: i32) -> u64 {
    let mut col = pos % 8;
    let mut row = pos / 8;

    let mut pattern: u64 = 0;

    let mut i = 1;
    while i <= MAX_FIELD_DISTANCE {
        col += dir_col;
        row += dir_row;

        if col < 0 || col > 7 || row < 0 || row > 7 {
            break;
        }

        let pattern_index = row * 8 + col;
        pattern |= 1 << pattern_index as u64;

        i += 1;
    }

    pattern
}

const fn is_border(pos: i32) -> bool {
    if pos < 21 || pos > 98 {
        return true;
    }

    pos % 10 == 0 || pos % 10 == 9
}

// Pawn attack move patterns

pub fn get_pawn_attacks(pawns: BitBoard, color: Color) -> BitBoard {
    if color.is_white() {
        white_left_pawn_attacks(pawns) | white_right_pawn_attacks(pawns)
    } else {
        black_left_pawn_attacks(pawns) | black_right_pawn_attacks(pawns)
    }
}

pub fn white_left_pawn_attacks(pawns: BitBoard) -> BitBoard {
    (pawns & BitBoard(0xfefefefefefefefe)) >> 9 // mask right column
}

pub fn white_right_pawn_attacks(pawns: BitBoard) -> BitBoard {
    (pawns & BitBoard(0x7f7f7f7f7f7f7f7f)) >> 7 // mask right column
}

pub fn black_left_pawn_attacks(pawns: BitBoard) -> BitBoard {
    (pawns & BitBoard(0xfefefefefefefefe)) << 7 // mask right column
}

pub fn black_right_pawn_attacks(pawns: BitBoard) -> BitBoard {
    (pawns & BitBoard(0x7f7f7f7f7f7f7f7f)) << 9 // mask right column
}

// Positions where pawns may move two squares
pub const PAWN_DOUBLE_MOVE_LINES: [u64; 2] = [
    0b0000000000000000111111110000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000111111110000000000000000,
];

// Patterns to check, whether a piece is on a light or dark field
pub const LIGHT_COLORED_FIELD_PATTERN: u64 = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
pub const DARK_COLORED_FIELD_PATTERN: u64 = 0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn gets_column_mask() {
        for offset in (0..63).step_by(8) {
            assert_eq!(
                get_column_mask(0 + offset),
                0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001
            );
            assert_eq!(
                get_column_mask(1 + offset),
                0b00000010_00000010_00000010_00000010_00000010_00000010_00000010_00000010
            );
            assert_eq!(
                get_column_mask(2 + offset),
                0b00000100_00000100_00000100_00000100_00000100_00000100_00000100_00000100
            );
            assert_eq!(
                get_column_mask(3 + offset),
                0b00001000_00001000_00001000_00001000_00001000_00001000_00001000_00001000
            );
            assert_eq!(
                get_column_mask(4 + offset),
                0b00010000_00010000_00010000_00010000_00010000_00010000_00010000_00010000
            );
            assert_eq!(
                get_column_mask(5 + offset),
                0b00100000_00100000_00100000_00100000_00100000_00100000_00100000_00100000
            );
            assert_eq!(
                get_column_mask(6 + offset),
                0b01000000_01000000_01000000_01000000_01000000_01000000_01000000_01000000
            );
            assert_eq!(
                get_column_mask(7 + offset),
                0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000
            );
        }
    }

    #[test]
    fn valid_from_to_masks() {
        assert_eq!(get_from_to_mask(0, 7), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_11111111);
        assert_eq!(get_from_to_mask(7, 0), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_11111111);

        assert_eq!(get_from_to_mask(1, 6), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01111110);
        assert_eq!(get_from_to_mask(6, 1), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01111110);

        assert_eq!(get_from_to_mask(3, 4), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00011000);
        assert_eq!(get_from_to_mask(4, 3), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00011000);

        assert_eq!(get_from_to_mask(4, 6), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01110000);
        assert_eq!(get_from_to_mask(6, 4), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01110000);
    }
}
