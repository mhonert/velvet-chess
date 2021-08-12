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

pub const EMPTY: i8 = 0;
pub const P: i8 = 1;
pub const N: i8 = 2;
pub const B: i8 = 3;
pub const R: i8 = 4;
pub const Q: i8 = 5;
pub const K: i8 = 6;

const KING_VALUE: i16 = 1500;
const QUEEN_VALUE: i16 = 1000;
const ROOK_VALUE: i16 = 525;
const BISHOP_VALUE: i16 = 350;
const KNIGHT_VALUE: i16 = 350;
const PAWN_VALUE: i16 = 100;

const PIECE_VALUES: [i16; 7] = [
    0,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    KING_VALUE,
];

#[inline]
pub const fn get_piece_value(piece_id: usize) -> i16 {
    PIECE_VALUES[piece_id]
}
