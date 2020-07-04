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

use crate::pieces;

pub struct Board {
    items: [i8; 64],
    active_player: Color,
    castling_state: u8,
    enpassant_state: (u8, u8),
    white_king: u8,
    black_king: u8,
    halfmove_clock: u16,
    halfmove_count: u16,
}

impl Board {
    pub fn new(items: &Vec<i8>, active_player: Color, castling_state: u8, enpassant_target: Option<u8>,
               halfmove_clock: u16, fullmove_num: u16, ) -> Self {

        if items.len() != 64 {
            panic!("Expected a vector with 64 elements, but got {}", items.len() );
        }

        let halfmove_count = (fullmove_num - 1) * 2 + if active_player == WHITE { 0 } else { 1 };

        let mut board = Board {
            items: [0; 64],
            active_player,
            castling_state,
            enpassant_state: (0, 0),
            white_king: 0,
            black_king: 0,
            halfmove_clock,
            halfmove_count,
        };

        if enpassant_target.is_some() {
            board.set_enpassant(enpassant_target.unwrap())
        }

        for i in 0..64 {
            let item = items[i];
            board.items[i] = item;

            if item == pieces::K {
                board.white_king = i as u8;
            } else if item == -pieces::K {
                board.black_king = i as u8;
            }
        }

        board
    }

    fn set_enpassant(&mut self, pos: u8) {
        if pos >= WhiteBoardPos::PawnLineStart as u8 {
            self.enpassant_state = (0, 1 << (pos - WhiteBoardPos::PawnLineStart as u8));
        } else {
            self.enpassant_state = (1 << (pos - BlackBoardPos::PawnLineStart as u8), 0);
        };
    }

    pub fn get_item(&self, pos: usize) -> i8 {
        self.items[pos]
    }

    pub fn active_player(&self) -> Color {
        self.active_player
    }

    pub fn can_castle(&self, castling: Castling) -> bool {
        self.castling_state & castling as u8 != 0
    }

    pub fn can_enpassant(&self, color: Color, location: u8) -> bool {
        if color == WHITE
            && location >= WhiteBoardPos::EnPassantLineStart as u8
            && location <= WhiteBoardPos::EnPassantLineEnd as u8
        {
            return self.enpassant_state.1 & (1 << (location - WhiteBoardPos::EnPassantLineStart as u8)) != 0;
        } else if color == BLACK
            && location >= BlackBoardPos::EnPassantLineStart as u8
            && location <= BlackBoardPos::EnPassantLineEnd as u8
        {
            return self.enpassant_state.0 & (1 << (location - BlackBoardPos::EnPassantLineStart as u8)) != 0;
        }

        false
    }

    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn fullmove_count(&self) -> u16 {
        self.halfmove_count / 2 + 1
    }
}

pub type Color = i8;

pub const WHITE: Color = 1;
pub const BLACK: Color = -1;

#[repr(u8)]
pub enum Castling {
    WhiteKingSide = 1 << 1,
    BlackKingSide = 1 << 2,
    WhiteQueenSide = 1 << 3,
    BlackQueenSide = 1 << 4,
}

#[repr(u8)]
pub enum WhiteBoardPos {
    KingSideRook = 63,
    QueenSideRook = 56,

    PawnLineStart = 48,
    PawnLineEnd = 55,

    EnPassantLineStart = 16,
    EnPassantLineEnd = 23,
}

#[repr(u8)]
pub enum BlackBoardPos {
    QueenSideRook = 0,
    KingSideRook = 7,

    PawnLineStart = 8,
    PawnLineEnd = 15,

    EnPassantLineStart = 40,
    EnPassantLineEnd = 47,
}
