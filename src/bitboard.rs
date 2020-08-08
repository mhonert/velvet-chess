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

use crate::bitboard::Direction::{
    East, North, NorthEast, NorthWest, South, SouthEast, SouthWest, West,
};

pub struct Bitboard {
    knight_attacks: [u64; 64],
    king_attacks: [u64; 64],
    ray_attacks: [u64; 65 * 8],
    white_pawn_freepath: [u64; 64],
    black_pawn_freepath: [u64; 64],
}

impl Bitboard {
    pub fn new() -> Self {
        let knight_attacks =
            calculate_single_move_patterns([21, 19, 12, 8, -12, -21, -19, -8].to_vec());
        let king_attacks =
            calculate_single_move_patterns([1, 10, -1, -10, 9, 11, -9, -11].to_vec());
        let ray_attacks = calculate_ray_attacks();
        let white_pawn_freepath = create_pawn_free_path_patterns(-1);
        let black_pawn_freepath = create_pawn_free_path_patterns(1);

        Bitboard {
            knight_attacks,
            king_attacks,
            ray_attacks,
            white_pawn_freepath,
            black_pawn_freepath,
        }
    }

    pub fn get_knight_attacks(&self, pos: i32) -> u64 {
        self.knight_attacks[pos as usize]
    }

    pub fn get_king_attacks(&self, pos: i32) -> u64 {
        self.king_attacks[pos as usize]
    }

    fn get_positive_ray_attacks(&self, occupied: u64, dir: Direction, pos: i32) -> u64 {
        let dir_offset = dir as usize * 65;
        let mut attacks = self.ray_attacks[dir_offset + pos as usize];
        let blocker = attacks & occupied;

        if blocker == 0 {
            return attacks;
        }

        let first_blocker_pos = 63 - blocker.leading_zeros();
        attacks ^= self.ray_attacks[dir_offset + first_blocker_pos as usize];
        attacks
    }

    fn get_negative_ray_attacks(&self, occupied: u64, dir: Direction, pos: i32) -> u64 {
        let dir_offset = dir as usize * 65;
        let mut attacks = self.ray_attacks[dir_offset + pos as usize];
        let blocker = attacks & occupied;

        let first_blocker_pos = blocker.trailing_zeros();
        attacks ^= self.ray_attacks[dir_offset + first_blocker_pos as usize];
        attacks
    }

    pub fn get_diagonal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        self.get_positive_ray_attacks(occupied, Direction::NorthEast, pos)
            | self.get_negative_ray_attacks(occupied, Direction::SouthWest, pos)
    }

    pub fn get_anti_diagonal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        self.get_positive_ray_attacks(occupied, Direction::NorthWest, pos)
            | self.get_negative_ray_attacks(occupied, Direction::SouthEast, pos)
    }

    pub fn get_horizontal_attacks(&self, occupied: u64, pos: i32) -> u64 {
        self.get_positive_ray_attacks(occupied, Direction::West, pos)
            | self.get_negative_ray_attacks(occupied, Direction::East, pos)
    }

    pub fn get_vertical_attacks(&self, occupied: u64, pos: i32) -> u64 {
        self.get_positive_ray_attacks(occupied, Direction::North, pos)
            | self.get_negative_ray_attacks(occupied, Direction::South, pos)
    }

    pub fn get_white_pawn_freepath(&self, pos: i32) -> u64 {
        self.white_pawn_freepath[pos as usize]
    }

    pub fn get_black_pawn_freepath(&self, pos: i32) -> u64 {
        self.black_pawn_freepath[pos as usize]
    }
}

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

#[repr(usize)]
enum Direction {
    NorthWest = 0,
    North = 1,
    NorthEast = 2,
    East = 3,
    SouthEast = 4,
    South = 5,
    SouthWest = 6,
    West = 7,
}

pub const MAX_FIELD_DISTANCE: i32 = 7; // maximum distance between two fields on the board

const DIRECTIONS: [usize; 8] = [
    NorthWest as usize,
    North as usize,
    NorthEast as usize,
    East as usize,
    SouthEast as usize,
    South as usize,
    SouthWest as usize,
    West as usize,
];

const DIRECTION_COL_OFFSET: [i32; 8] = [-1, 0, 1, 1, 1, 0, -1, -1];
const DIRECTION_ROW_OFFSET: [i32; 8] = [-1, -1, -1, 0, 1, 1, 1, 0];

fn calculate_ray_attacks() -> [u64; 65 * 8] {
    let mut patterns: [u64; 65 * 8] = [0; 65 * 8];

    let mut index = 0;
    for dir in DIRECTIONS.iter() {
        for pos in 0..64 {
            let mut col = pos % 8;
            let mut row = pos / 8;

            let mut attack_bitboard: u64 = 0;

            for _ in 1..=MAX_FIELD_DISTANCE {
                col += DIRECTION_COL_OFFSET[*dir];
                row += DIRECTION_ROW_OFFSET[*dir];

                if col < 0 || col > 7 || row < 0 || row > 7 {
                    break;
                }

                let pattern_index = row * 8 + col;
                attack_bitboard |= 1 << pattern_index as u64;
            }
            patterns[index] = attack_bitboard;
            index += 1;
        }
        patterns[index] = 0;
        index += 1;
    }

    patterns
}

fn is_border(pos: i32) -> bool {
    if pos < 21 || pos > 98 {
        return true;
    }

    pos % 10 == 0 || pos % 10 == 9
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
