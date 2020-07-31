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
use crate::pieces::EMPTY;
use crate::zobrist::Zobrist;
use crate::colors::{Color, WHITE, BLACK};
use crate::piece_sq_tables::PieceSquareTables;
use crate::score_util::{unpack_score, unpack_eg_score};

const MAX_GAME_HALFMOVES: usize = 5898 * 2;

pub struct Board {
    zobrist: Zobrist,
    pst: PieceSquareTables,
    items: [i8; 64],
    hash: u64,
    active_player: Color,
    castling_state: u8,
    enpassant_state: u16,
    white_king: u8,
    black_king: u8,
    halfmove_clock: u16,
    halfmove_count: u16,
    score: i16,
    eg_score: i16,

    history_counter: usize,
    state_history: [u64; MAX_GAME_HALFMOVES],
    hash_history: [u64; MAX_GAME_HALFMOVES],
    score_history: [i16; MAX_GAME_HALFMOVES],
    eg_score_history: [i16; MAX_GAME_HALFMOVES]
}

impl Board {
    pub fn new(items: &[i8], active_player: Color, castling_state: u8, enpassant_target: Option<u8>,
               halfmove_clock: u16, fullmove_num: u16, ) -> Self {

        if items.len() != 64 {
            panic!("Expected a vector with 64 elements, but got {}", items.len() );
        }

        let halfmove_count = (fullmove_num - 1) * 2 + if active_player == WHITE { 0 } else { 1 };

        let mut board = Board {
            zobrist: Zobrist::new(),
            pst: PieceSquareTables::new(),
            items: [0; 64],
            hash: 0,
            active_player,
            castling_state,
            enpassant_state: 0,
            white_king: 0,
            black_king: 0,
            halfmove_clock,
            halfmove_count,
            score: 0,
            eg_score: 0,
            history_counter: 0,
            state_history: [0; MAX_GAME_HALFMOVES],
            hash_history: [0; MAX_GAME_HALFMOVES],
            score_history: [0; MAX_GAME_HALFMOVES],
            eg_score_history: [0; MAX_GAME_HALFMOVES]
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

        board.recalculate_hash();
        board
    }

    pub fn recalculate_hash(&mut self) {
        self.hash = 0;

        for pos in 0..64 {
            let piece = self.items[pos];
            if piece != EMPTY {
                self.hash ^= self.zobrist.piece_numbers(piece, pos);
            }
        }

        if self.active_player == BLACK {
            self.hash ^= self.zobrist.player
        }

        self.update_hash_for_castling(Castling::BlackKingSide as u8
            | Castling::BlackQueenSide as u8
            | Castling::WhiteKingSide as u8
            | Castling::WhiteQueenSide as u8);

        self.update_hash_for_enpassant(0);
    }

    fn update_hash_for_castling(&mut self, previous_castling_state: u8) {
        self.hash ^= self.zobrist.castling[previous_castling_state as usize];
        self.hash ^= self.zobrist.castling[self.castling_state as usize];
    }

    fn set_enpassant(&mut self, pos: u8) {
        let previous_state = self.enpassant_state;

        println!("Set en passant: {}", pos);
        if pos >= WhiteBoardPos::PawnLineStart as u8 {
            self.enpassant_state = 1 << ((pos - WhiteBoardPos::PawnLineStart as u8) as u16 + 8);
        } else {
            self.enpassant_state = 1 << ((pos - BlackBoardPos::PawnLineStart as u8) as u16);
        };

        self.update_hash_for_enpassant(previous_state);
    }

    fn update_hash_for_enpassant(&mut self, previous_state: u16) {
        let new_state = self.enpassant_state;
        if previous_state != new_state {
            if previous_state != 0 {
                self.hash ^= self.zobrist.en_passant[previous_state.trailing_zeros() as usize];
            }

            if new_state != 0 {
                println!("New state: {}, tz: {}, lz: {}", new_state, new_state.trailing_zeros(), new_state.leading_zeros());
                self.hash ^= self.zobrist.en_passant[new_state.trailing_zeros() as usize];
            }
        }
    }

    pub fn get_item(&self, pos: u32) -> i8 {
        self.items[pos as usize]
    }

    pub fn get_hash(&self) -> u64 {
        self.hash
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
            return self.enpassant_state & (1 << (location - WhiteBoardPos::EnPassantLineStart as u8)) != 0;
        } else if color == BLACK
            && location >= BlackBoardPos::EnPassantLineStart as u8
            && location <= BlackBoardPos::EnPassantLineEnd as u8
        {
            return self.enpassant_state & (1 << (location - BlackBoardPos::EnPassantLineStart as u8 + 8)) != 0;
        }

        false
    }

    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    pub fn fullmove_count(&self) -> u16 {
        self.halfmove_count / 2 + 1
    }

    fn increase_half_move_count(&mut self) {
        self.halfmove_count += 1;
        self.halfmove_clock += 1;

        // self.hash_code ^= PLAYER_RNG_NUMBER;
    }

    pub fn perform_move(&mut self, target_piece_id: u32, move_start: u32, move_end: u32) -> i32 {
        self.store_state();
        self.increase_half_move_count();

        // let own_piece = self.remove_piece(move_start);

        0

    }

    pub fn remove_piece(&mut self, pos: usize) -> i32 {
        let piece = self.items[pos];
        self.subtract_piece_score(pos, piece);
        self.hash ^= self.zobrist.piece_numbers(piece, pos);

        let color = piece.signum();
        self.remove(piece, color, pos)
    }

    fn subtract_piece_score(&mut self, pos: usize, piece: i8) {
        let packed_score = self.pst.get_packed_score(piece, pos);

        self.score -= unpack_score(packed_score);
        self.eg_score -= unpack_eg_score(packed_score);
    }

    fn remove(&mut self, piece: i8, color: Color, pos: usize) -> i32 {

        0
    }

    fn store_state(&mut self) {
        let state = (self.castling_state as u64) << 56 |
            (self.halfmove_clock as u64) << 32 |
            (self.enpassant_state as u64);

        self.state_history[self.history_counter] = state;
        self.hash_history[self.history_counter] = self.hash;
        self.score_history[self.history_counter] = self.score;
        self.eg_score_history[self.history_counter] = self.eg_score;
        self.history_counter += 1;
    }
}

#[repr(u8)]
pub enum Castling {
    WhiteKingSide = 1 << 0,
    BlackKingSide = 1 << 1,
    WhiteQueenSide = 1 << 2,
    BlackQueenSide = 1 << 3,
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
