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

use crate::bitboard::Direction::{Horizontal, AntiDiagonal, Vertical, Diagonal};
use crate::colors::{Color, WHITE};

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

const KNIGHT_ATTACKS: [u64; 64] = calculate_single_move_patterns([21, 19, 12, 8, -12, -21, -19, -8]);
const KING_ATTACKS: [u64; 64] = calculate_single_move_patterns([1, 10, -1, -10, 9, 11, -9, -11]);
const WHITE_KING_SHIELD: [u64; 64]= create_king_shield_patterns(-1);
const BLACK_KING_SHIELD: [u64; 64]= create_king_shield_patterns(1);
const KING_DANGER_ZONE: [u64; 64]= create_king_danger_zone_patterns();

const LINE_MASKS: [LinePatterns; 64 * 4] = calc_line_patterns();

#[inline]
pub fn get_knight_attacks(pos: i32) -> u64 {
    unsafe { *KNIGHT_ATTACKS.get_unchecked(pos as usize) }
}

#[inline]
pub fn get_king_attacks(pos: i32) -> u64 {
    unsafe { *KING_ATTACKS.get_unchecked(pos as usize) }
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
pub fn get_column_side_mask(pos: i32) -> u64 {
    let col_mask = get_column_mask(pos);
    let left_mask = (col_mask & 0b01111111_01111111_01111111_01111111_01111111_01111111_01111111_01111111) << 1;
    let right_mask = (col_mask & 0b11111110_11111110_11111110_11111110_11111110_11111110_11111110_11111110) >> 1;
    left_mask | right_mask
}

#[inline]
/// Returns a mask where all bits in rows before the given position are set
fn get_lower_block_mask(pos: i32) -> u64 {
    0b00000000_11111111_11111111_11111111_11111111_11111111_11111111_11111111 >> (56 - pos / 8 * 8)
}

#[inline]
/// Returns a mask where all bits in rows after the given position are set
fn get_upper_block_mask(pos: i32) -> u64 {
    0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_00000000u64.wrapping_shl((pos / 8 * 8) as u32)
}

#[inline]
pub fn get_white_pawn_freepath(pos: i32) -> u64 {
    get_column_mask(pos) & get_lower_block_mask(pos)
}

#[inline]
pub fn get_black_pawn_freepath(pos: i32) -> u64 {
    get_column_mask(pos) & get_upper_block_mask(pos)
}

#[inline]
pub fn get_white_pawn_freesides(pos: i32) -> u64 {
    get_column_side_mask(pos) & get_lower_block_mask(pos)
}

#[inline]
pub fn get_black_pawn_freesides(pos: i32) -> u64 {
    get_column_side_mask(pos) & get_upper_block_mask(pos)
}

#[inline]
pub fn get_white_king_shield(pos: i32) -> u64 {
    unsafe { *WHITE_KING_SHIELD.get_unchecked(pos as usize) }
}

#[inline]
pub fn get_black_king_shield(pos: i32) -> u64 {
    unsafe { *BLACK_KING_SHIELD.get_unchecked(pos as usize) }
}

#[inline]
pub fn get_king_danger_zone(pos: i32) -> u64 {
    unsafe { *KING_DANGER_ZONE.get_unchecked(pos as usize) }
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

const DIRECTIONS: [usize; 4] = [
    Horizontal as usize,
    Vertical as usize,
    Diagonal as usize,
    AntiDiagonal as usize,
];

#[derive(Copy, Clone)]
struct LinePatterns {
    lower: u64,
    upper: u64,
    combined: u64
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
    let mut patterns: [LinePatterns; 64 * 4] = [LinePatterns{lower: 0, upper: 0, combined: 0}; 64 * 4];

    let mut index = 0;
    let mut dir_index = 0;
    while dir_index < DIRECTIONS.len() {
        let dir = DIRECTIONS[dir_index];
        let mut pos = 0;
        while pos < 64 {
            let lower = calc_pattern(pos, DIRECTION_COL_OFFSET[dir], DIRECTION_ROW_OFFSET[dir]);
            let upper = calc_pattern(pos, -DIRECTION_COL_OFFSET[dir], -DIRECTION_ROW_OFFSET[dir]);
            let combined = upper | lower;
            patterns[index] = LinePatterns{lower, upper, combined};
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

pub fn get_pawn_attacks(pawns: u64, color: Color) -> u64 {
    if color == WHITE {
        white_left_pawn_attacks(pawns) | white_right_pawn_attacks(pawns)
    } else {
        black_left_pawn_attacks(pawns) | black_right_pawn_attacks(pawns)
    }
}

pub fn white_left_pawn_attacks(pawns: u64) -> u64 {
    (pawns & 0xfefefefefefefefe) >> 9 // mask right column
}

pub fn white_right_pawn_attacks(pawns: u64) -> u64 {
    (pawns & 0x7f7f7f7f7f7f7f7f) >> 7 // mask right column
}

pub fn black_left_pawn_attacks(pawns: u64) -> u64 {
    (pawns & 0xfefefefefefefefe) << 7 // mask right column
}

pub fn black_right_pawn_attacks(pawns: u64) -> u64 {
    (pawns & 0x7f7f7f7f7f7f7f7f) << 9 // mask right column
}

// Patterns to check, whether the fields between king and rook are empty
pub const WHITE_KING_SIDE_CASTLING_BIT_PATTERN: u64 =
    0b01100000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
pub const WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN: u64 =
    0b00001110_00000000_00000000_00000000_00000000_00000000_00000000_00000000;

pub const BLACK_KING_SIDE_CASTLING_BIT_PATTERN: u64 =
    0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01100000;
pub const BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN: u64 =
    0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001110;

// Positions where pawns may move two squares
pub const PAWN_DOUBLE_MOVE_LINES: [u64; 3] = [
    0b0000000000000000000000000000000000000000111111110000000000000000,
    0,
    0b0000000000000000111111110000000000000000000000000000000000000000,
];

// Patterns to check, whether a piece is on a light or dark field
pub const LIGHT_COLORED_FIELD_PATTERN: u64 =
    0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
pub const DARK_COLORED_FIELD_PATTERN: u64 =
    0b10101010_10101010_10101010_10101010_10101010_10101010_10101010_10101010;

// Pawn shield in front of king
const fn create_king_shield_patterns(direction: i32) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];

    let mut pos = 0;
    while pos < 64 {
        let row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        let mut distance = 1;
        while distance <= 2 {
            let shield_row = row + direction * distance;
            if shield_row < 0 || shield_row > 7 { // Outside board
                distance += 1;
                continue;
            }

            let front_pawn_pos = shield_row * 8 + col;
            pattern |= 1 << front_pawn_pos as u64;
            if col > 0 {
                let front_west_pawn_pos = shield_row * 8 + col - 1;
                pattern |= 1 << front_west_pawn_pos as u64;
            }

            if col < 7 {
                let front_east_pawn_pos = shield_row * 8 + col + 1;
                pattern |= 1 << front_east_pawn_pos as u64;
            }

            distance += 1;
        }
        patterns[pos as usize] = pattern;

        pos += 1;
    }

    patterns
}

const KING_DANGER_ZONE_SIZE: i32 = 2;

// King danger zone patterns
const fn create_king_danger_zone_patterns() -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];

    let mut pos = 0;
    while pos < 64 {
        let row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        let mut row_offset = -KING_DANGER_ZONE_SIZE;
        while row_offset <= KING_DANGER_ZONE_SIZE {
            let zone_row = row + row_offset;
            if zone_row < 0 || zone_row > 7 { // Outside board
                row_offset += 1;
                continue;
            }

            let mut col_offset = -KING_DANGER_ZONE_SIZE;
            while col_offset <= KING_DANGER_ZONE_SIZE {
                let zone_col = col + col_offset;
                if zone_col < 0 || zone_col > 7 { // Outside board
                    col_offset += 1;
                    continue;
                }

                let pattern_pos = zone_row * 8 + zone_col;
                pattern |= 1 << pattern_pos as u64;

                col_offset += 1;
            }

            row_offset += 1;
        }
        patterns[pos as usize] = pattern;

        pos += 1;
    }

    patterns
}

// Mirrors the given bitboard
#[inline]
pub fn mirror(bitboard: u64) -> u64 {
    bitboard.swap_bytes()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    pub fn gets_column_mask() {
        for offset in (0..63).step_by(8) {
            assert_eq!(get_column_mask(0 + offset), 0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001);
            assert_eq!(get_column_mask(1 + offset), 0b00000010_00000010_00000010_00000010_00000010_00000010_00000010_00000010);
            assert_eq!(get_column_mask(2 + offset), 0b00000100_00000100_00000100_00000100_00000100_00000100_00000100_00000100);
            assert_eq!(get_column_mask(3 + offset), 0b00001000_00001000_00001000_00001000_00001000_00001000_00001000_00001000);
            assert_eq!(get_column_mask(4 + offset), 0b00010000_00010000_00010000_00010000_00010000_00010000_00010000_00010000);
            assert_eq!(get_column_mask(5 + offset), 0b00100000_00100000_00100000_00100000_00100000_00100000_00100000_00100000);
            assert_eq!(get_column_mask(6 + offset), 0b01000000_01000000_01000000_01000000_01000000_01000000_01000000_01000000);
            assert_eq!(get_column_mask(7 + offset), 0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000);
        }
    }

    #[test]
    pub fn gets_column_side_mask() {
        for offset in (0..63).step_by(8) {
            assert_eq!(get_column_side_mask(0 + offset), 0b00000010_00000010_00000010_00000010_00000010_00000010_00000010_00000010);
            assert_eq!(get_column_side_mask(1 + offset), 0b00000101_00000101_00000101_00000101_00000101_00000101_00000101_00000101);
            assert_eq!(get_column_side_mask(2 + offset), 0b00001010_00001010_00001010_00001010_00001010_00001010_00001010_00001010);
            assert_eq!(get_column_side_mask(3 + offset), 0b00010100_00010100_00010100_00010100_00010100_00010100_00010100_00010100);
            assert_eq!(get_column_side_mask(4 + offset), 0b00101000_00101000_00101000_00101000_00101000_00101000_00101000_00101000);
            assert_eq!(get_column_side_mask(5 + offset), 0b01010000_01010000_01010000_01010000_01010000_01010000_01010000_01010000);
            assert_eq!(get_column_side_mask(6 + offset), 0b10100000_10100000_10100000_10100000_10100000_10100000_10100000_10100000);
            assert_eq!(get_column_side_mask(7 + offset), 0b01000000_01000000_01000000_01000000_01000000_01000000_01000000_01000000);
        }
    }

    #[test]
    pub fn gets_lower_block_mask() {
        assert_eq!(get_lower_block_mask(0),  0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000);
        assert_eq!(get_lower_block_mask(8),  0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_11111111);
        assert_eq!(get_lower_block_mask(16), 0b00000000_00000000_00000000_00000000_00000000_00000000_11111111_11111111);
        assert_eq!(get_lower_block_mask(24), 0b00000000_00000000_00000000_00000000_00000000_11111111_11111111_11111111);
        assert_eq!(get_lower_block_mask(32), 0b00000000_00000000_00000000_00000000_11111111_11111111_11111111_11111111);
        assert_eq!(get_lower_block_mask(40), 0b00000000_00000000_00000000_11111111_11111111_11111111_11111111_11111111);
        assert_eq!(get_lower_block_mask(48), 0b00000000_00000000_11111111_11111111_11111111_11111111_11111111_11111111);
        assert_eq!(get_lower_block_mask(56), 0b00000000_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[test]
    pub fn gets_upper_block_mask() {
        assert_eq!(get_upper_block_mask(0),  0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_00000000);
        assert_eq!(get_upper_block_mask(8),  0b11111111_11111111_11111111_11111111_11111111_11111111_00000000_00000000);
        assert_eq!(get_upper_block_mask(16), 0b11111111_11111111_11111111_11111111_11111111_00000000_00000000_00000000);
        assert_eq!(get_upper_block_mask(24), 0b11111111_11111111_11111111_11111111_00000000_00000000_00000000_00000000);
        assert_eq!(get_upper_block_mask(32), 0b11111111_11111111_11111111_00000000_00000000_00000000_00000000_00000000);
        assert_eq!(get_upper_block_mask(40), 0b11111111_11111111_00000000_00000000_00000000_00000000_00000000_00000000);
        assert_eq!(get_upper_block_mask(48), 0b11111111_00000000_00000000_00000000_00000000_00000000_00000000_00000000);
        assert_eq!(get_upper_block_mask(56), 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00000000);
    }
}