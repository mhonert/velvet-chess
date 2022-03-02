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

use crate::bitboard::{BitBoard, get_from_to_mask};
use crate::board::Board;
use crate::board::castling::Castling::{BlackKingSide, BlackQueenSide, WhiteKingSide, WhiteQueenSide};
use crate::colors::{Color};
use crate::zobrist::castling_zobrist_key;

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Castling {
    WhiteKingSide = 1 << 0,
    BlackKingSide = 1 << 1,
    WhiteQueenSide = 1 << 2,
    BlackQueenSide = 1 << 3,
}

#[derive(Default, Copy, Clone, PartialEq, Debug)]
pub struct CastlingState(u8);

impl CastlingState {
    const CLEAR_BY_COLOR: [u8; 2] = [
        !(WhiteQueenSide as u8 | WhiteKingSide as u8),
        !(BlackQueenSide as u8 | BlackKingSide as u8),
    ];

    pub const KING_SIDE: [Castling; 2] = [
        Castling::WhiteKingSide,
        Castling::BlackKingSide,
    ];

    pub const QUEEN_SIDE: [Castling; 2] = [
        Castling::WhiteQueenSide,
        Castling::BlackQueenSide,
    ];

    const ALL_CASTLING: u8 = Castling::WhiteKingSide as u8
        | Castling::WhiteQueenSide as u8
        | Castling::BlackKingSide as u8
        | Castling::BlackQueenSide as u8;

    pub const ALL: CastlingState = CastlingState(Self::ALL_CASTLING);

    #[inline]
    pub fn can_castle(&self, side: Castling) -> bool {
        (self.0 & side as u8) != 0
    }

    #[inline]
    pub fn can_castle_king_side(&self, color: Color) -> bool {
        self.can_castle(Self::king_side(color))
    }

    #[inline]
    pub fn can_castle_queen_side(&self, color: Color) -> bool {
        self.can_castle(Self::queen_side(color))
    }

    #[inline]
    pub fn set_has_castled(&mut self, color: Color) {
        let idx = color.idx();
        self.0 &= Self::CLEAR_BY_COLOR[idx];
    }

    #[inline]
    pub fn set_can_castle(&mut self, castling: Castling) {
        self.0 |= castling as u8;
    }

    #[inline]
    pub fn clear(&mut self, color: Color) {
        self.0 &= Self::CLEAR_BY_COLOR[color.idx()];
    }

    #[inline]
    pub fn clear_side(&mut self, side: Castling) {
        self.0 ^= side as u8;
    }

    #[inline]
    pub fn zobrist_key(&self) -> u64 {
        castling_zobrist_key(self.0)
    }

    fn king_side(color: Color) -> Castling {
        Self::KING_SIDE[color.idx()]
    }

    fn queen_side(color: Color) -> Castling {
        Self::QUEEN_SIDE[color.idx()]
    }
}

#[derive(Clone, Copy)]
pub struct CastlingRules {
    chess960: bool,
    king_start: [i8; 2],
    king_side_rook: [i8; 2],
    queen_side_rook: [i8; 2],
}

impl Default for CastlingRules {
    /// Returns the Castling Rules for standard chess
    fn default() -> Self {
        CastlingRules::new(false, 4, 7, 0)
    }
}

impl CastlingRules {
    const KS_KING_END: [u8; 2] = [ 63 - 1, 7 - 1 ];
    const KS_ROOK_END: [u8; 2] = [ 63 - 2, 7 - 2 ];

    const QS_KING_END: [u8; 2] = [ 56 + 2, 2 ];
    const QS_ROOK_END: [u8; 2] = [ 56 + 3, 3 ];

    pub fn new(
        chess960: bool,
        king_start_col: i8,
        king_side_rook_col: i8,
        queen_side_rook_col: i8,
    ) -> Self {
        let w_king_start = 56 + king_start_col;
        let w_king_side_rook = 56 + king_side_rook_col;
        let w_queen_side_rook = 56 + queen_side_rook_col;

        let b_king_start = king_start_col;
        let b_king_side_rook = king_side_rook_col;
        let b_queen_side_rook = queen_side_rook_col;

        CastlingRules{
            chess960,
            king_start: [w_king_start, b_king_start],
            king_side_rook: [w_king_side_rook, b_king_side_rook],
            queen_side_rook: [w_queen_side_rook, b_queen_side_rook],
        }
    }

    pub fn is_chess960(&self) -> bool {
        self.chess960
    }

    pub fn is_ks_castling(&self, color: Color, move_to: i32) -> bool {
        self.ks_rook_start(color) == move_to
    }

    pub fn is_qs_castling(&self, color: Color, move_to: i32) -> bool {
        self.qs_rook_start(color) == move_to
    }

    pub fn is_king_start(&self, color: Color, pos: i32) -> bool {
        self.king_start(color) == pos
    }

    pub fn king_start(&self, color: Color) -> i32 {
        self.king_start[color.idx()] as i32
    }

    pub fn ks_rook_start(&self, color: Color) -> i32 {
        self.king_side_rook[color.idx()] as i32
    }

    pub fn qs_rook_start(&self, color: Color) -> i32 {
        self.queen_side_rook[color.idx()] as i32
    }

    pub fn ks_king_end(color: Color) -> i32 {
        Self::KS_KING_END[color.idx()] as i32
    }

    pub fn ks_rook_end(color: Color) -> i32 {
        Self::KS_ROOK_END[color.idx()] as i32
    }

    pub fn qs_king_end(color: Color) -> i32 {
        Self::QS_KING_END[color.idx()] as i32
    }

    pub fn qs_rook_end(color: Color) -> i32 {
        Self::QS_ROOK_END[color.idx()] as i32
    }

    pub fn is_ks_castling_valid(&self, color: Color, board: &Board, empty_bb: u64) -> bool {
        let idx = color.idx();
        let king_start = self.king_start[idx];
        let king_end = Self::KS_KING_END[idx] as i8;
        let rook_start = self.king_side_rook[idx];
        let rook_end = Self::KS_ROOK_END[idx] as i8;
        Self::is_castling_valid(board, color.flip(), empty_bb, king_start, king_end, rook_start, rook_end)
    }

    pub fn is_qs_castling_valid(&self, color: Color, board: &Board, empty_bb: u64) -> bool {
        let idx = color.idx();
        let king_start = self.king_start[idx];
        let king_end = Self::QS_KING_END[idx] as i8;
        let rook_start = self.queen_side_rook[idx];
        let rook_end = Self::QS_ROOK_END[idx] as i8;
        Self::is_castling_valid(board, color.flip(), empty_bb, king_start, king_end, rook_start, rook_end)
    }

    fn is_castling_valid(board: &Board, opp_color: Color, mut empty_bb: u64, king_start: i8, king_end: i8, rook_start: i8, rook_end: i8) -> bool {
        empty_bb |= (1u64 << king_start) | (1u64 << rook_start);

        let king_route_bb = get_from_to_mask(king_start, king_end);
        if empty_bb & king_route_bb != king_route_bb {
            return false;
        }

        let rook_route_bb = get_from_to_mask(rook_start, rook_end);
        if empty_bb & rook_route_bb != rook_route_bb {
            return false;
        }

        for pos in BitBoard(king_route_bb) {
            if board.is_attacked(opp_color, pos as i32) {
                return false;
            }
        }

        true
    }
}

