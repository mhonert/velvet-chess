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

use crate::bitboard::{black_left_pawn_attacks, black_right_pawn_attacks, white_left_pawn_attacks, white_right_pawn_attacks, DARK_COLORED_FIELD_PATTERN, LIGHT_COLORED_FIELD_PATTERN, get_knight_attacks, get_king_attacks, get_white_pawn_freepath, get_white_pawn_freesides, get_black_pawn_freepath, get_black_pawn_freesides, get_bishop_attacks, get_rook_attacks};
use crate::boardpos::{BlackBoardPos, WhiteBoardPos};
use crate::castling::Castling;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, EMPTY, K, N, P, Q, R, get_piece_value};
use crate::pos_history::PositionHistory;
use crate::score_util::{unpack_eg_score, unpack_score};
use crate::options::{Options, PieceSquareTables};
use crate::zobrist::{piece_zobrist_key, player_zobrist_key, castling_zobrist_key, enpassant_zobrist_key};

const MAX_GAME_HALFMOVES: usize = 5898 * 2;

const BASE_PIECE_PHASE_VALUE: i32 = 2;
const PAWN_PHASE_VALUE: i32 = -1; // relative to the base piece value
const QUEEN_PHASE_VALUE: i32 = 4; // relative to the base piece value

pub const MAX_PHASE: i32 = 16 * PAWN_PHASE_VALUE + 30 * BASE_PIECE_PHASE_VALUE + 2 * QUEEN_PHASE_VALUE;


pub struct Board {
    pub options: Options,
    pub pst: PieceSquareTables,
    pos_history: PositionHistory,
    items: [i8; 64],
    bitboards: [u64; 13],
    bitboards_all_pieces: [u64; 3],
    hash: u64,
    castling_state: u8,
    enpassant_state: u16,
    pub white_king: i32,
    pub black_king: i32,
    halfmove_clock: u16,
    pub halfmove_count: u16,
    pub score: i16,
    pub eg_score: i16,

    history_counter: usize,
    state_history: [u64; MAX_GAME_HALFMOVES],
    hash_history: [u64; MAX_GAME_HALFMOVES],
    score_history: [i16; MAX_GAME_HALFMOVES],
    eg_score_history: [i16; MAX_GAME_HALFMOVES],
}

impl Board {
    pub fn new(
        items: &[i8],
        active_player: Color,
        castling_state: u8,
        enpassant_target: Option<i8>,
        halfmove_clock: u16,
        fullmove_num: u16,
    ) -> Self {
        if items.len() != 64 {
            panic!(
                "Expected a vector with 64 elements, but got {}",
                items.len()
            );
        }

        let options = Options::new();
        let pst = PieceSquareTables::new(&options);

        let mut board = Board {
            options,
            pst,
            pos_history: PositionHistory::new(),
            items: [0; 64],
            bitboards: [0; 13],
            bitboards_all_pieces: [0; 3],
            hash: 0,
            castling_state,
            enpassant_state: 0,
            white_king: 0,
            black_king: 0,
            halfmove_clock,
            halfmove_count: 0,
            score: 0,
            eg_score: 0,
            history_counter: 0,
            state_history: [0; MAX_GAME_HALFMOVES],
            hash_history: [0; MAX_GAME_HALFMOVES],
            score_history: [0; MAX_GAME_HALFMOVES],
            eg_score_history: [0; MAX_GAME_HALFMOVES],
        };

        board.set_position(
            items,
            active_player,
            castling_state,
            enpassant_target,
            halfmove_clock,
            fullmove_num,
        );
        board
    }

    pub fn set_position(
        &mut self,
        items: &[i8],
        active_player: Color,
        castling_state: u8,
        enpassant_target: Option<i8>,
        halfmove_clock: u16,
        fullmove_num: u16,
    ) {
        self.pos_history.clear();
        if items.len() != 64 {
            panic!(
                "Expected a vector with 64 elements, but got {}",
                items.len()
            );
        }

        self.halfmove_count = (fullmove_num - 1) * 2 + if active_player == WHITE { 0 } else { 1 };
        self.halfmove_clock = halfmove_clock;
        self.hash = 0;
        self.castling_state = castling_state;
        self.enpassant_state = 0;

        if enpassant_target.is_some() {
            self.set_enpassant(enpassant_target.unwrap())
        }

        self.bitboards = [0; 13];
        self.bitboards_all_pieces = [0; 3];
        self.items = [EMPTY; 64];
        self.history_counter = 0;
        self.score = 0;
        self.eg_score = 0;

        for i in 0..64 {
            let item = items[i];

            if item != EMPTY {
                self.add_piece(item.signum(), item.abs(), i);
            } else {
                self.items[i] = item;
            }

            if item == K {
                self.white_king = i as i32;
            } else if item == -K {
                self.black_king = i as i32;
            }
        }

        self.recalculate_hash();
    }

    pub fn recalculate_hash(&mut self) {
        self.hash = 0;

        for pos in 0..64 {
            let piece = self.items[pos];
            if piece != EMPTY {
                self.hash ^= piece_zobrist_key(piece, pos);
            }
        }

        if self.active_player() == BLACK {
            self.hash ^= player_zobrist_key()
        }

        self.update_hash_for_castling(ALL_CASTLING);

        self.update_hash_for_enpassant(0);
    }

    pub fn eval_set_position(&mut self,
                             items: &[i8],
                             halfmove_count: u16,
                             castling_state: u8) {
        self.score = 0;
        self.eg_score = 0;
        self.halfmove_count = halfmove_count;
        self.castling_state = castling_state;
        self.bitboards_all_pieces = [0; 3];
        self.bitboards = [0; 13];

        for pos in 0..64 {
            let piece = items[pos];
            self.items[pos] = EMPTY;
            if piece != EMPTY {
                self.add_piece_without_inc_update(piece.signum(), piece, pos as i32);
                self.add_piece_score(piece, pos);

                if piece == K {
                    self.white_king = pos as i32;
                } else if piece == -K {
                    self.black_king = pos as i32;
                }
            }
        }
    }

    pub fn get_castling_state(&self) -> u8 {
        self.castling_state
    }

    fn update_hash_for_castling(&mut self, previous_castling_state: u8) {
        self.hash ^= castling_zobrist_key(previous_castling_state);
        self.hash ^= castling_zobrist_key(self.castling_state);
    }

    fn set_enpassant(&mut self, pos: i8) {
        let previous_state = self.enpassant_state;

        if pos >= WhiteBoardPos::PawnLineStart as i8 {
            self.enpassant_state = 1 << ((pos - WhiteBoardPos::PawnLineStart as i8) as u16 + 8);
        } else {
            self.enpassant_state = 1 << ((pos - BlackBoardPos::PawnLineStart as i8) as u16);
        };

        self.update_hash_for_enpassant(previous_state);
    }

    fn update_hash_for_enpassant(&mut self, previous_state: u16) {
        let new_state = self.enpassant_state;
        if previous_state != new_state {
            if previous_state != 0 {
                self.hash ^= enpassant_zobrist_key(previous_state);
            }

            if new_state != 0 {
                self.hash ^= enpassant_zobrist_key(new_state);
            }
        }
    }

    pub fn get_item(&self, pos: i32) -> i8 {
        unsafe { *self.items.get_unchecked(pos as usize) }
    }

    pub fn get_hash(&self) -> u64 {
        self.hash
    }

    pub fn active_player(&self) -> Color {
        if (self.halfmove_count & 1) == 0 {
            WHITE
        } else {
            BLACK
        }
    }

    pub fn can_castle(&self, castling: Castling) -> bool {
        (self.castling_state & castling as u8) != 0
    }

    pub fn has_white_castled(&self) -> bool {
        (self.castling_state & Castling::WhiteHasCastled as u8) != 0
    }

    pub fn has_black_castled(&self) -> bool {
        (self.castling_state & Castling::BlackHasCastled as u8) != 0
    }

    pub fn get_enpassant_state(&self) -> u16 {
        self.enpassant_state
    }

    pub fn can_enpassant(&self, color: Color, location: u8) -> bool {
        if color == WHITE
            && location >= WhiteBoardPos::EnPassantLineStart as u8
            && location <= WhiteBoardPos::EnPassantLineEnd as u8
        {
            return self.enpassant_state
                & (1 << (location - WhiteBoardPos::EnPassantLineStart as u8))
                != 0;
        } else if color == BLACK
            && location >= BlackBoardPos::EnPassantLineStart as u8
            && location <= BlackBoardPos::EnPassantLineEnd as u8
        {
            return self.enpassant_state
                & (1 << (location - BlackBoardPos::EnPassantLineStart as u8 + 8))
                != 0;
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

        self.hash ^= player_zobrist_key();
    }

    pub fn perform_move(&mut self, target_piece_id: i8, move_start: i32, move_end: i32) -> i8 {
        self.store_state();
        self.increase_half_move_count();

        let own_piece = self.remove_piece(move_start);
        let color = own_piece.signum();

        self.clear_en_passant();

        if self.get_item(move_end) != EMPTY {
            // Capture move (except en passant)
            let removed_piece = self.remove_piece(move_end);
            self.add_piece(color, target_piece_id, move_end as usize);

            self.reset_half_move_clock();

            if target_piece_id == K {
                if color == WHITE {
                    self.white_king = move_end;
                    self.set_white_king_moved();
                } else {
                    self.black_king = move_end;
                    self.set_black_king_moved();
                }
            }

            self.pos_history.push(self.hash);
            return removed_piece.abs();
        }

        self.add_piece(color, target_piece_id, move_end as usize);
        if own_piece == P {
            self.reset_half_move_clock();

            // Special en passant handling
            if move_start - move_end == 16 {
                self.set_enpassant(move_start as i8);
            } else if move_start - move_end == 7 {
                self.remove_piece(move_start + WHITE as i32);
                self.pos_history.push(self.hash);
                return EN_PASSANT;
            } else if move_start - move_end == 9 {
                self.remove_piece(move_start - WHITE as i32);
                self.pos_history.push(self.hash);
                return EN_PASSANT;
            }
        } else if own_piece == -P {
            self.reset_half_move_clock();

            // Special en passant handling
            if move_start - move_end == -16 {
                self.set_enpassant(move_start as i8);
            } else if move_start - move_end == -7 {
                self.remove_piece(move_start + BLACK as i32);
                self.pos_history.push(self.hash);
                return EN_PASSANT;
            } else if move_start - move_end == -9 {
                self.remove_piece(move_start - BLACK as i32);
                self.pos_history.push(self.hash);
                return EN_PASSANT;
            }
        } else if own_piece == K {
            self.white_king = move_end;

            // Special castling handling
            if move_start - move_end == -2 {
                self.remove_piece(WhiteBoardPos::KingSideRook as i32);
                self.add_piece(WHITE, R, WhiteBoardPos::KingStart as usize + 1);
                self.set_white_has_castled();
            } else if move_start - move_end == 2 {
                self.remove_piece(WhiteBoardPos::QueenSideRook as i32);
                self.add_piece(WHITE, R, WhiteBoardPos::KingStart as usize - 1);
                self.set_white_has_castled();
            } else {
                self.set_white_king_moved();
            }
        } else if own_piece == -K {
            self.black_king = move_end;

            // Special castling handling
            if move_start - move_end == -2 {
                self.remove_piece(BlackBoardPos::KingSideRook as i32);
                self.add_piece(BLACK, R, BlackBoardPos::KingStart as usize + 1);
                self.set_black_has_castled();
            } else if move_start - move_end == 2 {
                self.remove_piece(BlackBoardPos::QueenSideRook as i32);
                self.add_piece(BLACK, R, BlackBoardPos::KingStart as usize - 1);
                self.set_black_has_castled();
            } else {
                self.set_black_king_moved();
            }
        }

        // Position history

        self.pos_history.push(self.hash);
        EMPTY
    }

    pub fn perform_null_move(&mut self) {
        self.store_state();
        self.increase_half_move_count();
        self.clear_en_passant();
    }

    pub fn reset_half_move_clock(&mut self) {
        self.halfmove_clock = 0;
    }

    pub fn set_white_has_castled(&mut self) {
        let previous_state = self.castling_state;
        self.castling_state |= Castling::WhiteHasCastled as u8;
        self.castling_state &= !(Castling::WhiteKingSide as u8);
        self.castling_state &= !(Castling::WhiteQueenSide as u8);
        self.update_hash_for_castling(previous_state);
    }

    pub fn set_black_has_castled(&mut self) {
        let previous_state = self.castling_state;
        self.castling_state |= Castling::BlackHasCastled as u8;
        self.castling_state &= !(Castling::BlackKingSide as u8);
        self.castling_state &= !(Castling::BlackQueenSide as u8);
        self.update_hash_for_castling(previous_state);
    }

    pub fn set_white_king_moved(&mut self) {
        if self.can_castle(Castling::WhiteKingSide) || self.can_castle(Castling::WhiteQueenSide) {
            let previous_state = self.castling_state;
            self.castling_state &= !(Castling::WhiteKingSide as u8);
            self.castling_state &= !(Castling::WhiteQueenSide as u8);
            self.update_hash_for_castling(previous_state);
        }
    }

    pub fn set_black_king_moved(&mut self) {
        if self.can_castle(Castling::BlackKingSide) || self.can_castle(Castling::BlackQueenSide) {
            let previous_state = self.castling_state;
            self.castling_state &= !(Castling::BlackKingSide as u8);
            self.castling_state &= !(Castling::BlackQueenSide as u8);
            self.update_hash_for_castling(previous_state);
        }
    }

    pub fn undo_move(&mut self, piece: i8, move_start: i32, move_end: i32, removed_piece_id: i8) {
        self.pos_history.pop();

        let color = piece.signum();
        self.remove_piece_without_inc_update(move_end);
        self.add_piece_without_inc_update(color, piece, move_start);

        if removed_piece_id == EN_PASSANT {
            if (move_start - move_end).abs() == 7 {
                self.add_piece_without_inc_update(-color, P * -color, move_start + color as i32);
            } else if (move_start - move_end).abs() == 9 {
                self.add_piece_without_inc_update(-color, P * -color, move_start - color as i32);
            }
        } else if removed_piece_id != EMPTY {
            self.add_piece_without_inc_update(-color, removed_piece_id * -color, move_end);
        }

        if piece == K {
            // White King
            self.white_king = move_start;

            // Undo Castle?
            if move_start - move_end == -2 {
                self.remove_piece_without_inc_update(WhiteBoardPos::KingStart as i32 + 1);
                self.add_piece_without_inc_update(WHITE, R, WhiteBoardPos::KingSideRook as i32);
            } else if move_start - move_end == 2 {
                self.remove_piece_without_inc_update(WhiteBoardPos::KingStart as i32 - 1);
                self.add_piece_without_inc_update(WHITE, R, WhiteBoardPos::QueenSideRook as i32);
            }
        } else if piece == -K {
            // Black King
            self.black_king = move_start;

            // Undo Castle?
            if move_start - move_end == -2 {
                self.remove_piece_without_inc_update(BlackBoardPos::KingStart as i32 + 1);
                self.add_piece_without_inc_update(BLACK, -R, BlackBoardPos::KingSideRook as i32);
            } else if move_start - move_end == 2 {
                self.remove_piece_without_inc_update(BlackBoardPos::KingStart as i32 - 1);
                self.add_piece_without_inc_update(BLACK, -R, BlackBoardPos::QueenSideRook as i32);
            }
        }

        self.restore_state();
    }

    pub fn undo_null_move(&mut self) {
        self.restore_state();
    }

    fn add_piece_without_inc_update(&mut self, color: Color, piece: i8, pos: i32) {
        unsafe {
            *self.items.get_unchecked_mut(pos as usize) = piece;
            *self.bitboards_all_pieces.get_unchecked_mut((color + 1) as usize) |= 1u64 << pos as u64;
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) |= 1u64 << pos as u64;
        }
    }

    pub fn add_piece(&mut self, color: Color, piece_id: i8, pos: usize) {
        let piece = piece_id * color;
        unsafe {
            *self.items.get_unchecked_mut(pos as usize) = piece;
        }

        self.add_piece_score(piece, pos);
        self.hash ^= piece_zobrist_key(piece, pos);

        unsafe {
            *self.bitboards_all_pieces.get_unchecked_mut((color + 1) as usize) |= 1u64 << pos as u64;
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) |= 1u64 << pos as u64;
        }
    }

    fn add_piece_score(&mut self, piece: i8, pos: usize) {
        let packed_score = self.pst.get_packed_score(piece, pos);

        self.score += unpack_score(packed_score);
        self.eg_score += unpack_eg_score(packed_score);
    }

    fn clear_en_passant(&mut self) {
        let previous_state = self.enpassant_state;

        if previous_state != 0 {
            self.enpassant_state = 0;
            self.update_hash_for_enpassant(previous_state);
        }
    }

    pub fn remove_piece(&mut self, pos: i32) -> i8 {
        let piece = self.get_item(pos);
        self.subtract_piece_score(piece, pos as usize);
        self.hash ^= piece_zobrist_key(piece, pos as usize);

        let color = piece.signum();
        self.remove(piece, color, pos)
    }

    fn subtract_piece_score(&mut self, piece: i8, pos: usize) {
        let packed_score = self.pst.get_packed_score(piece, pos);

        self.score -= unpack_score(packed_score);
        self.eg_score -= unpack_eg_score(packed_score);
    }

    fn remove_piece_without_inc_update(&mut self, pos: i32) {
        let piece = self.get_item(pos);
        let color = piece.signum();
        self.remove(piece, color, pos);
    }

    fn remove(&mut self, piece: i8, color: Color, pos: i32) -> i8 {
        unsafe {
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) &= !(1u64 << pos as u64);
            *self.bitboards_all_pieces.get_unchecked_mut((color + 1) as usize) &= !(1u64 << pos as u64);
            *self.items.get_unchecked_mut(pos as usize) = EMPTY;
        }

        if piece == R {
            if pos == WhiteBoardPos::QueenSideRook as i32 {
                self.set_rook_moved(Castling::WhiteQueenSide);
            } else if pos == WhiteBoardPos::KingSideRook as i32 {
                self.set_rook_moved(Castling::WhiteKingSide);
            }
        } else if piece == -R {
            if pos == BlackBoardPos::QueenSideRook as i32 {
                self.set_rook_moved(Castling::BlackQueenSide);
            } else if pos == BlackBoardPos::KingSideRook as i32 {
                self.set_rook_moved(Castling::BlackKingSide);
            }
        }
        piece
    }

    fn set_rook_moved(&mut self, castling: Castling) {
        if self.can_castle(castling) {
            let previous_state = self.castling_state;
            self.castling_state ^= castling as u8;
            self.update_hash_for_castling(previous_state);
        }
    }

    fn store_state(&mut self) {
        let state = (self.castling_state as u64) << 56
            | (self.halfmove_clock as u64) << 32
            | (self.enpassant_state as u64);

        unsafe {
            *self.state_history.get_unchecked_mut(self.history_counter) = state;
            *self.hash_history.get_unchecked_mut(self.history_counter) = self.hash;
            *self.score_history.get_unchecked_mut(self.history_counter) = self.score;
            *self.eg_score_history.get_unchecked_mut(self.history_counter) = self.eg_score;
        }
        self.history_counter += 1;
    }

    fn restore_state(&mut self) {
        self.halfmove_count -= 1;
        self.history_counter -= 1;

        unsafe {
            self.hash = *self.hash_history.get_unchecked(self.history_counter);
            self.score = *self.score_history.get_unchecked(self.history_counter);
            self.eg_score = *self.eg_score_history.get_unchecked(self.history_counter);
            let state = *self.state_history.get_unchecked(self.history_counter);

            self.castling_state = (state >> 56) as u8;
            self.halfmove_clock = ((state >> 32) & 0xFFFF) as u16;
            self.enpassant_state = (state & 0xFFFF) as u16;
        }

    }

    pub fn king_pos(&self, color: Color) -> i32 {
        if color == WHITE {
            return self.white_king as i32;
        }

        self.black_king as i32
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        self.is_attacked(-color, self.king_pos(color))
    }

    pub fn get_all_piece_bitboard(&self, color: Color) -> u64 {
        unsafe { *self.bitboards_all_pieces.get_unchecked((color + 1) as usize) }
    }

    pub fn get_occupancy_bitboard(&self) -> u64 {
        self.get_all_piece_bitboard(WHITE) | self.get_all_piece_bitboard(BLACK)
    }

    pub fn is_attacked(&self, opponent_color: Color, pos: i32) -> bool {
        self.find_smallest_attacker(self.get_occupancy_bitboard(), opponent_color, pos) >= 0
    }

    pub fn get_bitboard(&self, piece: i8) -> u64 {
        unsafe { *self.bitboards.get_unchecked((piece + 6) as usize) }
    }

    // Returns the position of the smallest attacker or -1 if there is no attacker
    pub fn find_smallest_attacker(&self, occupied_bb: u64, opp_color: Color, pos: i32) -> i32 {
        let target_bb = 1 << pos as u64;
        if opp_color == WHITE {
            // Check pawns
            let white_pawns = self.get_bitboard(P) & occupied_bb;
            if white_left_pawn_attacks(white_pawns) & target_bb != 0 {
                return pos + 9;
            } else if white_right_pawn_attacks(white_pawns) & target_bb != 0 {
                return pos + 7;
            }

            // Check knights
            let knights = self.get_bitboard(N) & occupied_bb;
            let attacking_knights = knights & get_knight_attacks(pos);
            if attacking_knights != 0 {
                return attacking_knights.trailing_zeros() as i32;
            }

            // Check bishops
            let bishops = self.get_bitboard(B) & occupied_bb;
            let bishop_attacks = get_bishop_attacks(occupied_bb, pos);
            let attacking_bishops = bishops & bishop_attacks;
            if attacking_bishops != 0 {
                return attacking_bishops.trailing_zeros() as i32;
            }

            // Check rooks
            let rooks = self.get_bitboard(R) & occupied_bb;
            let rook_attacks = get_rook_attacks(occupied_bb, pos);
            let attacking_rooks = rooks & rook_attacks;
            if attacking_rooks != 0 {
                return attacking_rooks.trailing_zeros() as i32;
            }

            // Check queens
            let queens = self.get_bitboard(Q) & occupied_bb;
            let attacking_queens = queens & (rook_attacks | bishop_attacks);
            if attacking_queens != 0 {
                return attacking_queens.trailing_zeros() as i32;
            }

            // Check king
            let king_pos = self.king_pos(WHITE);
            let attacking_king = get_king_attacks(king_pos) & target_bb;
            if attacking_king != 0 {
                let king_bb = 1u64 << (king_pos as u64);
                if (king_bb & occupied_bb) != 0 {
                    return king_pos;
                }
            }
        } else {
            // Check pawns
            let black_pawns = self.get_bitboard(-P) & occupied_bb;
            if black_left_pawn_attacks(black_pawns) & target_bb != 0 {
                return pos - 7;
            } else if black_right_pawn_attacks(black_pawns) & target_bb != 0 {
                return pos - 9;
            }

            // Check knights
            let knights = self.get_bitboard(-N) & occupied_bb;
            let attacking_knights = knights & get_knight_attacks(pos);
            if attacking_knights != 0 {
                return attacking_knights.trailing_zeros() as i32;
            }

            // Check bishops
            let bishops = self.get_bitboard(-B) & occupied_bb;
            let bishop_attacks = get_bishop_attacks(occupied_bb, pos);
            let attacking_bishops = bishops & bishop_attacks;
            if attacking_bishops != 0 {
                return attacking_bishops.trailing_zeros() as i32;
            }

            // Check rooks
            let rooks = self.get_bitboard(-R) & occupied_bb;
            let rook_attacks = get_rook_attacks(occupied_bb, pos);
            let attacking_rooks = rooks & rook_attacks;
            if attacking_rooks != 0 {
                return attacking_rooks.trailing_zeros() as i32;
            }

            // Check queens
            let queens = self.get_bitboard(-Q) & occupied_bb;
            let attacking_queens = queens & (rook_attacks | bishop_attacks);
            if attacking_queens != 0 {
                return attacking_queens.trailing_zeros() as i32;
            }

            // Check king
            let king_pos = self.king_pos(BLACK);
            let attacking_king = get_king_attacks(king_pos) & target_bb;
            if attacking_king != 0 {
                let king_bb = 1u64 << (king_pos as u64);
                if (king_bb & occupied_bb) != 0 {
                    return king_pos;
                }
            }
        }
        -1
    }

    pub fn is_legal_move(&mut self, color: Color, piece_id: i8, start: i32, end: i32) -> bool {
        let previous_piece = self.get_item(start);
        let move_state = self.perform_move(piece_id, start, end);
        let is_legal = !self.is_in_check(color);
        self.undo_move(previous_piece, start, end, move_state);
        is_legal
    }

    // Return true, if the engine considers the current position as a draw.
    // Note: it already considers the first repetition of a position as a draw to stop searching a branch that leads to a draw earlier.
    pub fn is_engine_draw(&self) -> bool {
        self.pos_history.is_single_repetition()
            || self.is_fifty_move_draw()
            || self.is_insufficient_material_draw()
    }

    fn is_fifty_move_draw(&self) -> bool {
        self.halfmove_clock >= 100
    }

    fn is_insufficient_material_draw(&self) -> bool {
        match (self.get_all_piece_bitboard(WHITE) | self.get_all_piece_bitboard(BLACK)).count_ones()
        {
            2 => true, // K vs K

            3 => {
                // K vs K+N or K vs K+B
                let knights_or_bishops = self.get_bitboard(N)
                    | self.get_bitboard(-N)
                    | self.get_bitboard(B)
                    | self.get_bitboard(-B);
                knights_or_bishops != 0
            }

            4 => {
                // Check for K+B vs K+B where bishops are on fields with the same color
                let white_bishops = self.get_bitboard(B);
                let black_bishops = self.get_bitboard(-B);

                ((white_bishops & LIGHT_COLORED_FIELD_PATTERN) != 0
                    && (black_bishops & LIGHT_COLORED_FIELD_PATTERN) != 0)
                    || ((white_bishops & DARK_COLORED_FIELD_PATTERN) != 0
                        && (black_bishops & DARK_COLORED_FIELD_PATTERN) != 0)
            }

            _ => false,
        }
    }

    pub fn is_pawn_move_close_to_promotion(
        &self,
        piece: i8,
        pos: i32,
        moves_left: i32,
        blockers: u64,
        opp_pawns: u64
    ) -> bool {
        if piece == P {
            let distance_to_promotion = pos / 8;
            return distance_to_promotion <= moves_left
                && (get_white_pawn_freepath(pos as i32) & blockers) == 0
                && (get_white_pawn_freesides(pos as i32) & opp_pawns) == 0;

        } else if piece == -P {
            let distance_to_promotion = 7 - pos / 8;
            return distance_to_promotion <= moves_left
                && (get_black_pawn_freepath(pos as i32) & blockers) == 0
                && (get_black_pawn_freesides(pos as i32) & opp_pawns) == 0;
        }

        false
    }

    /* Perform a Static Exchange Evaluation (SEE) to check, whether the net gain of the capture is still positive,
       after applying all immediate and discovered re-capture attacks.

       Returns:
       - a positive integer for winning captures
       - a negative integer for losing captures
       - a 0 otherwise
    */
    pub fn see_score(
        &mut self,
        opp_color: Color,
        from: i32,
        target: i32,
        own_piece_id: u32,
        captured_piece_id: u32,
    ) -> i32 {
        let mut score = get_piece_value(captured_piece_id as usize);
        let mut occupied = self.get_occupancy_bitboard() & !(1 << from as u64);
        let mut trophy_piece_score = get_piece_value(own_piece_id as usize);

        loop {
            // Opponent attack
            let attacker_pos = self.find_smallest_attacker(occupied, opp_color, target);
            if attacker_pos < 0 {
                return score as i32;
            }
            score -= trophy_piece_score;
            trophy_piece_score = get_piece_value(self.get_item(attacker_pos).abs() as usize);
            if score + trophy_piece_score < 0 {
                return score as i32;
            }

            occupied &= !(1 << attacker_pos);

            // Own attack
            let own_attacker_pos = self.find_smallest_attacker(occupied, -opp_color, target);
            if own_attacker_pos < 0 {
                return score as i32;
            }

            score += trophy_piece_score;
            trophy_piece_score = get_piece_value(self.get_item(own_attacker_pos).abs() as usize);
            if score - trophy_piece_score > 0 {
                return score as i32;
            }

            occupied &= !(1 << own_attacker_pos);
        }
    }

    pub fn get_static_score(&self) -> i32 {
        let mut score: i32 = 0;

        for i in 0..64 {
            let item = self.items[i];

            if item == EMPTY {
                continue;
            }
            score += get_piece_value(item.abs() as usize) as i32 * item.signum() as i32;
        }

        score
    }

    pub fn calc_phase_value(&self) -> i32 {
        calc_phase_value(self.get_all_piece_bitboard(WHITE) | self.get_all_piece_bitboard(BLACK),
                         self.get_bitboard(-P) | self.get_bitboard(P),
                         self.get_bitboard(Q), self.get_bitboard(Q))
    }
}

pub fn calc_phase_value(all_pieces: u64, all_pawns: u64, white_queens: u64, black_queens: u64) -> i32 {
    let pieces_except_king_count: i32 = all_pieces.count_ones() as i32 - 2; // -2 for two kings

    let white_queen_phase_score = if white_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
    let black_queen_phase_score = if black_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
    let queen_phase_score: i32 = white_queen_phase_score + black_queen_phase_score;
    let pawn_count: i32 = all_pawns.count_ones() as i32;

    pawn_count * PAWN_PHASE_VALUE + pieces_except_king_count * BASE_PIECE_PHASE_VALUE + queen_phase_score
}

#[inline]
pub fn interpolate_score(phase: i32, score: i32, eg_score: i32) -> i32 {
    let eg_phase: i32 = MAX_PHASE - phase;

    ((score * phase) + (eg_score * eg_phase)) / MAX_PHASE
}


pub const EN_PASSANT: i8 = 1 << 7;

const ALL_CASTLING: u8 = Castling::WhiteKingSide as u8
    | Castling::WhiteQueenSide as u8
    | Castling::BlackKingSide as u8
    | Castling::BlackQueenSide as u8;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::castling::Castling;

    #[test]
    fn update_hash_when_piece_moves() {
        let items: [i8; 64] = [
            0, 0, 0, -K, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, K, 0, 0, 0, 0,
        ];

        let mut board = Board::new(&items, WHITE, 0, None, 0, 1);

        board.recalculate_hash();
        let initial_hash = board.hash;

        let removed = board.perform_move(K, 59, 60);
        let hash_perform_move = board.hash;
        assert_ne!(initial_hash, hash_perform_move);

        board.undo_move(K, 59, 60, removed);
        let hash_undo_move = board.hash;
        assert_eq!(initial_hash, hash_undo_move);
    }

    #[test]
    fn incrementally_updates_hash() {
        let items: [i8; 64] = [
            0, 0, 0, -K, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, K, 0, 0, 0, 0,
        ];

        let mut board = Board::new(&items, WHITE, 0, None, 0, 1);
        board.recalculate_hash();

        board.add_piece(WHITE, P, 48);
        board.add_piece(BLACK, R, 1);
        board.set_rook_moved(Castling::BlackQueenSide);
        board.increase_half_move_count();

        let hash_incremental = board.hash;
        board.recalculate_hash();
        let hash_recalculated = board.hash;

        assert_eq!(hash_incremental, hash_recalculated);
    }

    #[test]
    fn performs_and_undos_white_castling_moves() {
        let items: [i8; 64] = [
            -R, 0, 0, 0, -K, 0, 0, -R, -P, -P, -P, -P, -P, -P, -P, -P, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, P, P, P, P, P, P,
            P, P, R, 0, 0, 0, K, 0, 0, R,
        ];

        let mut board = Board::new(&items, WHITE, ALL_CASTLING, None, 0, 1);

        let initial_items = board.items.clone();
        let initial_hash = board.hash;
        let initial_castling_state = board.castling_state;

        let previous_piece = board.get_item(WhiteBoardPos::KingStart as i32);
        let removed_piece_id = board.perform_move(
            K,
            WhiteBoardPos::KingStart as i32,
            WhiteBoardPos::KingStart as i32 - 2,
        );

        assert_ne!(&initial_items[..], &board.items[..]);
        assert_ne!(initial_hash, board.hash);
        assert_ne!(initial_castling_state, board.castling_state);

        board.undo_move(
            previous_piece,
            WhiteBoardPos::KingStart as i32,
            WhiteBoardPos::KingStart as i32 - 2,
            removed_piece_id,
        );

        assert_eq!(&initial_items[..], &board.items[..]);
        assert_eq!(initial_hash, board.hash);
        assert_eq!(initial_castling_state, board.castling_state);
    }

    #[test]
    fn performs_and_undos_black_castling_moves() {
        let items: [i8; 64] = [
            -R, 0, 0, 0, -K, 0, 0, -R, -P, -P, -P, -P, -P, -P, -P, -P, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, P, P, P, P, P, P,
            P, P, R, N, B, Q, K, B, N, R,
        ];

        let mut board = Board::new(&items, BLACK, ALL_CASTLING, None, 0, 1);

        let initial_items = board.items.clone();
        let initial_hash = board.hash;
        let initial_castling_state = board.castling_state;

        let previous_piece = board.get_item(BlackBoardPos::KingStart as i32);
        let removed_piece_id = board.perform_move(
            K,
            BlackBoardPos::KingStart as i32,
            BlackBoardPos::KingStart as i32 - 2,
        );

        assert_ne!(&initial_items[..], &board.items[..]);
        assert_ne!(initial_hash, board.hash);
        assert_ne!(initial_castling_state, board.castling_state);

        board.undo_move(
            previous_piece,
            BlackBoardPos::KingStart as i32,
            BlackBoardPos::KingStart as i32 - 2,
            removed_piece_id,
        );

        assert_eq!(&initial_items[..], &board.items[..]);
        assert_eq!(initial_hash, board.hash);
        assert_eq!(initial_castling_state, board.castling_state);
    }

    #[test]
    fn find_white_pawn_left_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  P,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            34,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_white_pawn_right_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  0,  P,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            36,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_black_pawn_left_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0, -P,  0,  0,  0,
            0,  0,  0,  B,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, BLACK, 0, None, 0, 1);
        assert_eq!(
            20,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), BLACK, 27)
        );
    }

    #[test]
    fn find_black_pawn_right_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0, -P,  0,  0,  0,  0,  0,
            0,  0,  0,  B,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, BLACK, 0, None, 0, 1);
        assert_eq!(
            18,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), BLACK, 27)
        );
    }

    #[test]
    fn find_knight_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  N,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            37,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_bishop_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  B,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            45,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_rook_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            R,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            24,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_queen_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  Q,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            29,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn find_king_attack() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert_eq!(
            35,
            board.find_smallest_attacker(board.get_occupancy_bitboard(), WHITE, 27)
        );
    }

    #[test]
    fn recognizes_white_in_check() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  K,  0, -Q,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert!(board.is_in_check(WHITE));
        assert!(!board.is_in_check(BLACK));
    }

    #[test]
    fn recognizes_black_in_check() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  Q,  0,  0,
            0,  0,  0, -B,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert!(board.is_in_check(BLACK));
        assert!(!board.is_in_check(WHITE));
    }

    #[test]
    fn see_white_discovered_attacks() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  0,  0, -K,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  K,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0, -Q,  0,  0,
            0,  0,  0,  0, -P,  0,  0,  0,
            0,  0,  0,  0,  R,  0,  0,  0,
            0,  0,  0,  0,  R,  0,  0,  0,
        ];

        let mut board = Board::new(&items, WHITE, 0, None, 0, 1);
        assert!(board.see_score(BLACK, 52, 44, R as u32, P as u32) > 0);
    }

    #[test]
    fn see_black_discovered_attacks() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  0,  0, -K,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  K,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  Q,  0,  0,
            0,  0,  0,  0,  P,  0,  0,  0,
            0,  0,  0,  0, -R,  0,  0,  0,
            0,  0,  0,  0, -R,  0,  0,  0,
        ];

        let mut board = Board::new(&items, BLACK, 0, None, 0, 1);
        assert!(board.see_score(WHITE, 52, 44, R as u32, P as u32) > 0);
    }

    #[test]
    fn updates_hash_for_piece_movements() {
        #[rustfmt::skip]
            let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let mut board = Board::new(&items, BLACK, 0, None, 0, 1);
        let initial_hash = board.get_hash();

        board.perform_move(K, 59, 60);
        let hash_after_move = board.get_hash();
        assert_ne!(initial_hash, hash_after_move);

        board.perform_move(K, 60, 59);
        let hash_reverted_move = board.get_hash();
        assert_eq!(initial_hash, hash_reverted_move);
    }

    #[test]
    fn updates_hash_for_en_passant_changes() {
        #[rustfmt::skip]
            let items: [i8; 64] = [
            0,  0,  0, -K,  0,  0,  0,  0,
            0,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  K,  0,  0,  0,  0,
        ];

        let mut board = Board::new(&items, BLACK, 0, None, 0, 1);
        let initial_hash = board.get_hash();

        board.set_enpassant(51);

        assert_ne!(
            initial_hash,
            board.get_hash(),
            "hash must be different if en passant flag is set"
        );

        board.clear_en_passant();
        assert_eq!(
            initial_hash,
            board.get_hash(),
            "hash must be eq to initial hash if en passant flag is cleared"
        );
    }
}
