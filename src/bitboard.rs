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

pub struct Bitboard {
    knight_attacks: [u64; 64],
    king_attacks: [u64; 64],
    white_pawn_freepath: [u64; 64],
    black_pawn_freepath: [u64; 64],
    white_pawn_freesides: [u64; 64],
    black_pawn_freesides: [u64; 64],
    white_king_shield: [u64; 64],
    black_king_shield: [u64; 64],
    king_danger_zone: [u64; 64],
    line_masks: [LinePatterns; 64 * 4]
}

impl Bitboard {
    pub fn new() -> Self {
        let knight_attacks =
            calculate_single_move_patterns([21, 19, 12, 8, -12, -21, -19, -8].to_vec());
        let king_attacks =
            calculate_single_move_patterns([1, 10, -1, -10, 9, 11, -9, -11].to_vec());
        let white_pawn_freepath = create_pawn_free_path_patterns(-1);
        let black_pawn_freepath = create_pawn_free_path_patterns(1);
        let white_pawn_freesides = create_pawn_free_sides_patterns(-1);
        let black_pawn_freesides = create_pawn_free_sides_patterns(1);
        let white_king_shield = create_king_shield_patterns(-1);
        let black_king_shield = create_king_shield_patterns(1);
        let king_danger_zone = create_king_danger_zone_patterns();

        let line_masks = calc_line_patterns();

        Bitboard {
            knight_attacks,
            king_attacks,
            white_pawn_freepath,
            black_pawn_freepath,
            white_pawn_freesides,
            black_pawn_freesides,
            white_king_shield,
            black_king_shield,
            king_danger_zone,
            line_masks
        }
    }

    pub fn get_knight_attacks(&self, pos: i32) -> u64 {
        self.knight_attacks[pos as usize]
    }

    pub fn get_king_attacks(&self, pos: i32) -> u64 {
        self.king_attacks[pos as usize]
    }

    pub fn get_white_king_shield(&self, pos: i32) -> u64 {
        self.white_king_shield[pos as usize]
    }

    pub fn get_black_king_shield(&self, pos: i32) -> u64 {
        self.black_king_shield[pos as usize]
    }

    pub fn get_king_danger_zone(&self, pos: i32) -> u64 {
        self.king_danger_zone[pos as usize]
    }

    #[inline]
    pub fn get_diagonal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        get_line_attacks(occupied, &self.line_masks[pos as usize + (Diagonal as usize * 64)])
    }

    #[inline]
    pub fn get_anti_diagonal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        get_line_attacks(occupied, &self.line_masks[pos as usize + (AntiDiagonal as usize * 64)])
    }

    #[inline]
    pub fn get_horizontal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        get_line_attacks(occupied, &self.line_masks[pos as usize + (Horizontal as usize * 64)])
    }

    #[inline]
    pub fn get_vertical_attacks(&self, occupied: u64, pos: i32) -> u64 {
        get_line_attacks(occupied, &self.line_masks[pos as usize + (Vertical as usize * 64)])
    }

    pub fn get_white_pawn_freepath(&self, pos: i32) -> u64 {
        self.white_pawn_freepath[pos as usize]
    }

    pub fn get_black_pawn_freepath(&self, pos: i32) -> u64 {
        self.black_pawn_freepath[pos as usize]
    }

    pub fn get_white_pawn_freesides(&self, pos: i32) -> u64 {
        self.white_pawn_freesides[pos as usize]
    }

    pub fn get_black_pawn_freesides(&self, pos: i32) -> u64 {
        self.black_pawn_freesides[pos as usize]
    }
}

// Calculate move patterns for pieces which can only move to one target field per direction (king and knight)
fn calculate_single_move_patterns(directions: Vec<i32>) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];
    let mut index: usize = 0;
    for board_pos in 21..=98 {
        if is_border(board_pos) {
            continue;
        }

        let mut pattern: u64 = 0;
        for dir in directions.iter() {
            let target_pos = board_pos + *dir;
            if !is_border(target_pos) {
                let row = (target_pos - 21) / 10;
                let col = (target_pos - 21) % 10;
                let bit_index = col + (row * 8);
                pattern |= 1 << bit_index as u64;
            }
        }
        patterns[index] = pattern;
        index += 1;
    }

    patterns
}


// Patterns for line movers (Bishop, Rook, Queen)
#[repr(usize)]
enum Direction {
    Horizontal = 0,
    Vertical = 1,
    Diagonal = 2,
    AntiDiagonal = 3,
}

pub const MAX_FIELD_DISTANCE: i32 = 7; // maximum distance between two fields on the board

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

fn calc_line_patterns() -> [LinePatterns; 64 * 4] {
    let mut patterns: [LinePatterns; 64 * 4] = [LinePatterns{lower: 0, upper: 0, combined: 0}; 64 * 4];

    let mut index = 0;
    for dir in DIRECTIONS.iter() {
        for pos in 0..64 {
            let lower = calc_pattern(pos, DIRECTION_COL_OFFSET[*dir], DIRECTION_ROW_OFFSET[*dir]);
            let upper = calc_pattern(pos, -DIRECTION_COL_OFFSET[*dir], -DIRECTION_ROW_OFFSET[*dir]);
            let combined = upper | lower;
            patterns[index] = LinePatterns{lower, upper, combined};
            index += 1;
        }
    }

    patterns
}

fn calc_pattern(pos: i32, dir_col: i32, dir_row: i32) -> u64 {
    let mut col = pos % 8;
    let mut row = pos / 8;

    let mut pattern: u64 = 0;

    for _ in 1..=MAX_FIELD_DISTANCE {
        col += dir_col;
        row += dir_row;

        if col < 0 || col > 7 || row < 0 || row > 7 {
            break;
        }

        let pattern_index = row * 8 + col;
        pattern |= 1 << pattern_index as u64;
    }

    pattern
}

fn is_border(pos: i32) -> bool {
    if pos < 21 || pos > 98 {
        return true;
    }

    pos % 10 == 0 || pos % 10 == 9
}

// Pawn attack move patterns

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

// Patterns to check, whether the path in front of the pawn is free (i.e. not blocked by opponent pieces)
fn create_pawn_free_path_patterns(direction: i32) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];
    for pos in 0..64 {
        let mut row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        while row >= 1 && row <= 6 {
            row += direction;
            pattern |= 1 << ((row * 8 + col) as u64);
        }
        patterns[pos as usize] = pattern;
    }

    patterns
}

// Patterns to check, whether the sides of the path in front of the pawn is free (i.e. not controlled by opponent pawns)
fn create_pawn_free_sides_patterns(direction: i32) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];
    for pos in 0..64 {
        let mut row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        while row >= 1 && row <= 6 {
            row += direction;
            if col - 1 >= 0 {
                pattern |= 1 << ((row * 8 + (col - 1)) as u64);
            }

            if col + 1 < 8 {
                pattern |= 1 << ((row * 8 + (col + 1)) as u64);
            }
        }
        patterns[pos as usize] = pattern;
    }

    patterns
}

// Pawn shield in front of king
fn create_king_shield_patterns(direction: i32) -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];

    for pos in 0..64 {
        let row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        for distance in 1..=2 {
            let shield_row = row + direction * distance;
            if shield_row < 0 || shield_row > 7 { // Outside board
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
        }
        patterns[pos as usize] = pattern;
    }

    patterns
}

const KING_DANGER_ZONE_SIZE: i32 = 2;

// King danger zone patterns
fn create_king_danger_zone_patterns() -> [u64; 64] {
    let mut patterns: [u64; 64] = [0; 64];

    for pos in 0..64 {
        let row = pos / 8;
        let col = pos & 7;
        let mut pattern: u64 = 0;

        for row_offset in -KING_DANGER_ZONE_SIZE..=KING_DANGER_ZONE_SIZE {
            let zone_row = row + row_offset;
            if zone_row < 0 || zone_row > 7 { // Outside board
                continue;
            }

            for col_offset in -KING_DANGER_ZONE_SIZE..=KING_DANGER_ZONE_SIZE {
                let zone_col = col + col_offset;
                if zone_col < 0 || zone_col > 7 { // Outside board
                    continue;
                }

                let pattern_pos = zone_row * 8 + zone_col;
                pattern |= 1 << pattern_pos as u64;
            }
        }
        patterns[pos as usize] = pattern;
    }

    patterns
}