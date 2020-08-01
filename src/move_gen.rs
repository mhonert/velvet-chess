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

use crate::board::{Board};
use crate::colors::Color;

pub struct MoveGenerator {
}

impl MoveGenerator {
    pub fn generate_moves(&self, board: &Board, active_player: Color) -> Vec<Move> {
        let mut moves: Vec<Move> = Vec::new();

        moves
    }
}

pub type Move = u32;

pub fn encode_move(piece: i32, start: u32, end: u32) -> Move {
    (piece.abs() as u32) | (start << 3) | (end << 10)
}

pub fn decode_piece_id(m: Move) -> u32  {
    m & 0x7
}

pub fn decode_start_index(m: Move) -> i32  {
    ((m >> 3) & 0x7F) as i32
}

pub fn decode_end_index(m: Move) -> i32  {
    ((m >> 10) & 0x7F) as i32
}
