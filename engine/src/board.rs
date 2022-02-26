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

pub mod castling;

use std::cmp::max;

use crate::bitboard::{BitBoard, black_left_pawn_attacks, black_right_pawn_attacks, DARK_COLORED_FIELD_PATTERN, get_black_pawn_freepath,  get_king_attacks, get_knight_attacks, get_pawn_attacks, get_white_pawn_freepath, LIGHT_COLORED_FIELD_PATTERN, white_left_pawn_attacks, white_right_pawn_attacks};
use crate::board::castling::{Castling, CastlingRules, CastlingState};
use crate::colors::{ToIndex, BLACK, Color, WHITE};
use crate::moves::{Move, MoveType};
use crate::nn::eval::NeuralNetEval;
use crate::pieces::{B, EMPTY, get_piece_value, K, N, P, Q, R};
use crate::pos_history::PositionHistory;
use crate::transposition_table::MAX_DEPTH;
use crate::zobrist::{enpassant_zobrist_key, piece_zobrist_key, player_zobrist_key};
use crate::magics::Magics;

#[repr(u8)]
pub enum WhiteBoardPos {
    PawnLineStart = 48,
    EnPassantLineStart = 16,
    EnPassantLineEnd = 23,
}

#[repr(u8)]
pub enum BlackBoardPos {
    PawnLineStart = 8,
    EnPassantLineStart = 40,
    EnPassantLineEnd = 47,
}

#[derive(Clone)]
pub struct Board {
    pub pos_history: PositionHistory,
    pub bitboards: [u64; 13],
    pub state: StateEntry,
    pub halfmove_count: u16,
    pub castling_rules: CastlingRules,

    magics: Magics,
    nn_eval: Box<NeuralNetEval>,
    items: [i8; 64],
    bitboards_all_pieces: [u64; 2],
    king_pos: [i32; 2],

    history: Vec<StateEntry>,
}

#[derive(Copy, Clone)]
pub struct StateEntry {
    hash: u64,
    en_passant: u16,
    castling: CastlingState,
    halfmove_clock: u8,
    history_start: u8,
}

impl Board {
    pub fn new(items: &[i8], active_player: Color, castling_state: CastlingState, enpassant_target: Option<i8>,
               halfmove_clock: u8, fullmove_num: u16, castling_rules: CastlingRules
    ) -> Self {
        assert_eq!(items.len(), 64, "Expected a vector with 64 elements, but got {}", items.len());

        let mut board = Board {
            magics: Magics::default(),
            pos_history: PositionHistory::default(),
            nn_eval: NeuralNetEval::new(),
            castling_rules,
            items: [0; 64],
            bitboards: [0; 13],
            bitboards_all_pieces: [0; 2],
            state: StateEntry{en_passant: 0, castling: CastlingState::default(), halfmove_clock: 0, hash: 0, history_start: 0},
            king_pos: [0; 2],
            halfmove_count: 0,
            history: Vec::with_capacity(MAX_DEPTH),
        };

        board.set_position(items, active_player, castling_state, enpassant_target, halfmove_clock, fullmove_num, castling_rules);
        board
    }

    pub fn reset(&mut self, pos_history: PositionHistory, bitboards: [u64; 13], halfmove_count: u16, state: StateEntry) {
        let white_bb = bitboards[7..=12].iter().fold(0, |acc, i| acc | i);
        let black_bb = bitboards[0..=5].iter().fold(0, |acc, i| acc | i);

        let white_king = bitboards[12].trailing_zeros() as i32;
        let black_king = bitboards[0].trailing_zeros() as i32;

        self.pos_history = pos_history;
        self.bitboards = bitboards;
        self.bitboards_all_pieces = [black_bb, white_bb];
        self.state = state;
        self.halfmove_count = halfmove_count;
        self.king_pos = [black_king, white_king];
        self.history.clear();

        self.items.fill(0);

        for color in (BLACK..=WHITE).step_by(2) {
            for piece_id in 1i8..=6i8 {
                let piece = piece_id * color;
                for pos in BitBoard(bitboards[(piece + 6) as usize]) {
                    self.items[pos as usize] = piece;
                }
            }
        }

        self.recalculate_hash();
        self.nn_eval.init_pos(&self.bitboards);
    }

    pub fn set_position(&mut self, items: &[i8], active_player: Color, castling_state: CastlingState, enpassant_target: Option<i8>,
                        halfmove_clock: u8, fullmove_num: u16, castling_rules: CastlingRules
    ) {
        self.pos_history.clear();
        assert_eq!(items.len(), 64, "Expected a vector with 64 elements, but got {}", items.len());

        self.halfmove_count = (max(1, fullmove_num) - 1) * 2 + if active_player == WHITE { 0 } else { 1 };
        self.state.halfmove_clock = halfmove_clock;
        self.state.history_start = halfmove_clock;
        self.state.hash = 0;
        self.state.castling = castling_state;
        self.state.en_passant = 0;
        self.castling_rules = castling_rules;

        if let Some(target) = enpassant_target {
            self.set_enpassant(target)
        }

        self.bitboards = [0; 13];
        self.bitboards_all_pieces = [0; 2];
        self.items = [EMPTY; 64];

        for i in 0..64 {
            let item = items[i];

            if item != EMPTY {
                self.add_piece(item.signum(), item.abs(), i);
            } else {
                self.items[i] = item;
            }

            if item == K {
                self.set_king_pos(WHITE, i as i32);
            } else if item == -K {
                self.set_king_pos(BLACK, i as i32);
            }
        }

        self.recalculate_hash();
    }

    pub fn recalculate_hash(&mut self) {
        self.state.hash = 0;

        for pos in 0..64 {
            let piece = self.items[pos];
            if piece != EMPTY {
                self.state.hash ^= piece_zobrist_key(piece, pos);
            }
        }

        if self.active_player() == BLACK {
            self.state.hash ^= player_zobrist_key()
        }

        self.update_hash_for_castling(CastlingState::ALL);

        self.update_hash_for_enpassant(0);
    }

    pub fn halfmove_count(&self) -> u16 {
        self.halfmove_count
    }

    fn update_hash_for_castling(&mut self, previous_castling_state: CastlingState) {
        self.state.hash ^= previous_castling_state.zobrist_key();
        self.state.hash ^= self.state.castling.zobrist_key();
    }

    fn set_enpassant(&mut self, pos: i8) {
        let previous_state = self.state.en_passant;

        if pos >= WhiteBoardPos::PawnLineStart as i8 {
            self.state.en_passant = 1 << ((pos - WhiteBoardPos::PawnLineStart as i8) as u16 + 8);
        } else {
            self.state.en_passant = 1 << ((pos - BlackBoardPos::PawnLineStart as i8) as u16);
        };

        self.update_hash_for_enpassant(previous_state);
    }

    fn update_hash_for_enpassant(&mut self, previous_state: u16) {
        if previous_state != 0 {
            self.state.hash ^= enpassant_zobrist_key(previous_state);
        }

        if self.state.en_passant != 0 {
            self.state.hash ^= enpassant_zobrist_key(self.state.en_passant);
        }
    }

    pub fn get_item(&self, pos: i32) -> i8 {
        unsafe { *self.items.get_unchecked(pos as usize) }
    }

    pub fn get_hash(&self) -> u64 {
        self.state.hash
    }

    pub fn active_player(&self) -> Color {
        if (self.halfmove_count & 1) == 0 {
            WHITE
        } else {
            BLACK
        }
    }

    pub fn can_castle(&self, castling: Castling) -> bool {
        self.state.castling.can_castle(castling)
    }

    pub fn can_castle_king_side(&self, color: Color) -> bool {
        self.state.castling.can_castle_king_side(color)
    }

    pub fn can_castle_queen_side(&self, color: Color) -> bool {
        self.state.castling.can_castle_queen_side(color)
    }

    pub fn get_enpassant_state(&self) -> u16 {
        self.state.en_passant
    }

    pub fn can_enpassant(&self, color: Color, location: u8) -> bool {
        if color == WHITE
            && location >= WhiteBoardPos::EnPassantLineStart as u8
            && location <= WhiteBoardPos::EnPassantLineEnd as u8
        {
            return self.state.en_passant
                & (1 << (location - WhiteBoardPos::EnPassantLineStart as u8))
                != 0;
        } else if color == BLACK
            && location >= BlackBoardPos::EnPassantLineStart as u8
            && location <= BlackBoardPos::EnPassantLineEnd as u8
        {
            return self.state.en_passant
                & (1 << (location - BlackBoardPos::EnPassantLineStart as u8 + 8))
                != 0;
        }

        false
    }

    pub fn halfmove_clock(&self) -> u8 {
        self.state.halfmove_clock
    }

    pub fn fullmove_count(&self) -> u16 {
        self.halfmove_count / 2 + 1
    }

    fn increase_half_move_count(&mut self) {
        self.halfmove_count += 1;
        self.state.halfmove_clock += 1;
        self.state.history_start += 1;

        self.state.hash ^= player_zobrist_key();
    }

    pub fn perform_move(&mut self, m: Move) -> (i8, i8) {
        self.nn_eval.start_move();
        self.pos_history.push(self.state.hash);
        self.store_state();
        self.increase_half_move_count();

        let move_start = m.start();
        let move_end = m.end();
        let target_piece_id = m.piece_id();

        let own_piece = self.remove_piece(move_start);
        let color = own_piece.signum();

        self.clear_en_passant();

        match m.typ() {
            MoveType::PawnQuiet => {
                self.add_piece(color, target_piece_id, move_end as usize);
                self.reset_half_move_clock();
            }

            MoveType::Quiet => {
                self.add_piece(color, target_piece_id, move_end as usize);
            }

            MoveType::PawnDoubleQuiet => {
                self.add_piece(color, target_piece_id, move_end as usize);
                self.reset_half_move_clock();
                self.set_enpassant(move_start as i8);
            }

            MoveType::Capture => {
                // Capture move (except en passant)
                let removed_piece = self.remove_piece(move_end);
                self.add_piece(color, target_piece_id, move_end as usize);

                self.reset_half_move_clock();

                if removed_piece.abs() >= R {
                    self.nn_eval.check_refresh();
                }

                return (own_piece, removed_piece.abs());
            }

            MoveType::KingCapture => {
                let removed_piece = self.remove_piece(move_end);
                self.add_piece(color, target_piece_id, move_end as usize);

                self.reset_half_move_clock();

                self.set_king_pos(color, move_end);
                self.set_king_moved(color);

                if removed_piece.abs() >= R {
                    self.nn_eval.check_refresh();
                }

                return (own_piece, removed_piece.abs());
            }

            MoveType::PawnSpecial => {
                self.reset_half_move_clock();

                if self.get_item(move_end) != EMPTY {
                    // Capture move with promotion
                    let removed_piece = self.remove_piece(move_end);
                    self.add_piece(color, target_piece_id, move_end as usize);

                    self.nn_eval.check_refresh();

                    return (own_piece, removed_piece.abs());
                }

                self.add_piece(color, target_piece_id, move_end as usize);
                if own_piece == P {
                    // Special en passant handling
                    if move_start - move_end == 7 {
                        self.remove_piece(move_start + WHITE as i32);
                        return (own_piece, P);
                    } else if move_start - move_end == 9 {
                        self.remove_piece(move_start - WHITE as i32);
                        return (own_piece, P);
                    }
                } else if own_piece == -P {
                    // Special en passant handling
                    if move_start - move_end == -7 {
                        self.remove_piece(move_start + BLACK as i32);
                        return (own_piece, P);
                    } else if move_start - move_end == -9 {
                        self.remove_piece(move_start - BLACK as i32);
                        return (own_piece, P);
                    }
                }

                if target_piece_id >= R { // Rook or Queen Promotion
                    self.nn_eval.check_refresh();
                }
            }

            MoveType::KingQuiet => {
                self.add_piece(color, target_piece_id, move_end as usize);
                self.set_king_pos(color, move_end);
                self.set_king_moved(color);
            }

            MoveType::Castling => {
                self.remove_piece(move_end);
                self.set_has_castled(color);

                if self.castling_rules.is_ks_castling(color, move_end) {
                    self.set_king_pos(color, CastlingRules::ks_king_end(color));
                    self.add_piece(color, K, CastlingRules::ks_king_end(color) as usize);
                    self.add_piece(color, R, CastlingRules::ks_rook_end(color) as usize);
                } else {
                    self.set_king_pos(color, CastlingRules::qs_king_end(color));
                    self.add_piece(color, K, CastlingRules::qs_king_end(color) as usize);
                    self.add_piece(color, R, CastlingRules::qs_rook_end(color) as usize);
                }
            }

        }

        (own_piece, EMPTY)
    }

    pub fn perform_null_move(&mut self) {
        self.store_state();
        self.increase_half_move_count();
        self.state.history_start = 0;
        self.clear_en_passant();
    }

    fn reset_half_move_clock(&mut self) {
        self.state.halfmove_clock = 0;
        self.state.history_start = 0;
    }

    fn set_has_castled(&mut self, color: Color) {
        let previous_state = self.state.castling;
        self.state.castling.set_has_castled(color);
        self.update_hash_for_castling(previous_state);
    }

    fn set_king_moved(&mut self, color: Color) {
        let previous_state = self.state.castling;
        self.state.castling.clear(color);
        self.update_hash_for_castling(previous_state);
    }

    pub fn undo_move(&mut self, m: Move, piece: i8, removed_piece_id: i8) {
        self.nn_eval.start_undo();
        self.pos_history.pop();

        let move_start = m.start();
        let move_end = m.end();

        let color = piece.signum();

        self.halfmove_count -= 1;
        self.restore_state();

        match m.typ() {
            MoveType::Quiet | MoveType::PawnQuiet | MoveType::PawnDoubleQuiet => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
            }

            MoveType::Capture => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(-color, removed_piece_id * -color, move_end);

                if removed_piece_id >= R {
                    self.nn_eval.check_refresh();
                }
            }

            MoveType::KingCapture => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(-color, removed_piece_id * -color, move_end);

                if piece == K {
                    // White King
                    self.set_king_pos(WHITE, move_start);
                } else {
                    // Black King
                    self.set_king_pos(BLACK, move_start);
                }

                if removed_piece_id >= R {
                    self.nn_eval.check_refresh();
                }
            }

            MoveType::PawnSpecial => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);

                if m.is_en_passant() {
                    if (move_start - move_end).abs() == 7 {
                        self.add_piece_without_inc_update(-color, P * -color, move_start + color as i32);
                    } else if (move_start - move_end).abs() == 9 {
                        self.add_piece_without_inc_update(-color, P * -color, move_start - color as i32);
                    }
                } else if removed_piece_id != EMPTY {
                    self.add_piece_without_inc_update(-color, removed_piece_id * -color, move_end);
                    self.nn_eval.check_refresh();

                } else if m.is_promotion() {
                    self.nn_eval.check_refresh();
                }
            }

            MoveType::KingQuiet => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);

                if piece == K {
                    self.set_king_pos(WHITE, move_start);
                } else if piece == -K {
                    self.set_king_pos(BLACK, move_start);
                }
            }

            MoveType::Castling => {
                if self.castling_rules.is_ks_castling(color, move_end) {
                    self.remove_piece_without_inc_update(CastlingRules::ks_king_end(color) as i32);
                    self.remove_piece_without_inc_update(CastlingRules::ks_rook_end(color) as i32);
                } else {
                    self.remove_piece_without_inc_update(CastlingRules::qs_king_end(color) as i32);
                    self.remove_piece_without_inc_update(CastlingRules::qs_rook_end(color) as i32);
                }

                self.add_piece_without_inc_update(color, R * color, move_end);
                self.set_king_pos(color, move_start);
                self.add_piece_without_inc_update(color, piece, move_start);
            }
        }
    }

    pub fn undo_null_move(&mut self) {
        self.halfmove_count -= 1;
        self.restore_state();
    }

    fn add_piece_without_inc_update(&mut self, color: Color, piece: i8, pos: i32) {
         unsafe {
            *self.items.get_unchecked_mut(pos as usize) = piece;
            *self.bitboards_all_pieces.get_unchecked_mut(color.idx()) |= 1u64 << pos as u64;
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) |= 1u64 << pos as u64;
        }

        self.nn_eval.add_piece(pos as usize, piece);
    }

    pub fn add_piece(&mut self, color: Color, piece_id: i8, pos: usize) {
        let piece = piece_id * color;

        unsafe {
            *self.items.get_unchecked_mut(pos as usize) = piece;
        }

        self.state.hash ^= piece_zobrist_key(piece, pos);

        unsafe {
            *self.bitboards_all_pieces.get_unchecked_mut(color.idx()) |= 1u64 << pos as u64;
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) |= 1u64 << pos as u64;
        }

        self.nn_eval.add_piece(pos, piece);
    }

    fn clear_en_passant(&mut self) {
        let previous_state = self.state.en_passant;

        if previous_state != 0 {
            self.state.en_passant = 0;
            self.update_hash_for_enpassant(previous_state);
        }
    }

    pub fn remove_piece(&mut self, pos: i32) -> i8 {
        let piece = self.get_item(pos);
        self.state.hash ^= piece_zobrist_key(piece, pos as usize);

        if piece == R {
            if self.castling_rules.is_ks_castling(WHITE, pos) {
                self.set_rook_moved(Castling::WhiteKingSide);
            } else if self.castling_rules.is_qs_castling(WHITE, pos) {
                self.set_rook_moved(Castling::WhiteQueenSide);
            }
        } else if piece == -R {
            if self.castling_rules.is_ks_castling(BLACK, pos) {
                self.set_rook_moved(Castling::BlackKingSide);
            } else if self.castling_rules.is_qs_castling(BLACK, pos) {
                self.set_rook_moved(Castling::BlackQueenSide);
            }
        }

        let color = piece.signum();
        self.remove(piece, color, pos)
    }

    fn remove_piece_without_inc_update(&mut self, pos: i32) {
        let piece = self.get_item(pos);
        let color = piece.signum();
        self.remove(piece, color, pos);
    }

    #[inline]
    fn remove(&mut self, piece: i8, color: Color, pos: i32) -> i8 {
        self.nn_eval.remove_piece(pos as usize, piece);

        unsafe {
            *self.bitboards.get_unchecked_mut((piece + 6) as usize) &= !(1u64 << pos as u64);
            *self.bitboards_all_pieces.get_unchecked_mut(color.idx()) &= !(1u64 << pos as u64);
            *self.items.get_unchecked_mut(pos as usize) = EMPTY;
        }

        piece
    }

    fn set_rook_moved(&mut self, castling: Castling) {
        if self.can_castle(castling) {
            let previous_state = self.state.castling;
            self.state.castling.clear_side(castling);
            self.update_hash_for_castling(previous_state);
        }
    }

    fn store_state(&mut self) {
        self.history.push(self.state);
    }

    fn restore_state(&mut self) {
        self.state = self.history.pop().unwrap();
    }

    #[inline]
    pub fn is_in_check(&self, color: Color) -> bool {
        if color == WHITE {
            self.is_attacked(BLACK, self.king_pos(WHITE))
        } else {
            self.is_attacked(WHITE, self.king_pos(BLACK))
        }
    }

    pub fn get_all_piece_bitboard(&self, color: Color) -> u64 {
        unsafe { *self.bitboards_all_pieces.get_unchecked(color.idx()) }
    }

    pub fn get_occupancy_bitboard(&self) -> u64 {
        self.get_all_piece_bitboard(WHITE) | self.get_all_piece_bitboard(BLACK)
    }

    pub fn is_attacked(&self, opponent_color: Color, pos: i32) -> bool {
        let empty_bb = !self.get_occupancy_bitboard();
        let target_bb = 1 << pos as u64;

        // Check knights
        if self.get_bitboard(N * opponent_color) & get_knight_attacks(pos) != 0 {
            return true;
        }

        // Check diagonal
        let queens = self.get_bitboard(Q * opponent_color);
        if (self.get_bitboard(B * opponent_color) | queens) & self.magics.get_bishop_attacks(empty_bb, pos) != 0 {
            return true;
        }

        // Check orthogonal
        if (self.get_bitboard(R * opponent_color) | queens) & self.magics.get_rook_attacks(empty_bb, pos) != 0 {
            return true;
        }

        // Check pawns
        let pawns = self.get_bitboard(P * opponent_color);
        if get_pawn_attacks(pawns, opponent_color) & target_bb != 0 {
            return true;
        }

        // Check king
        if get_king_attacks(self.king_pos(opponent_color)) & target_bb != 0 {
            return true;
        }

        false
    }

    pub fn get_bitboard(&self, piece: i8) -> u64 {
        unsafe { *self.bitboards.get_unchecked((piece + 6) as usize) }
    }

    #[inline]
    pub fn get_bishop_attacks(&self, empty_bb: u64, pos: i32) -> u64 {
        self.magics.get_bishop_attacks(empty_bb, pos)
    }

    #[inline]
    pub fn get_rook_attacks(&self, empty_bb: u64, pos: i32) -> u64 {
        self.magics.get_rook_attacks(empty_bb, pos)
    }

    #[inline]
    pub fn get_queen_attacks(&self, empty_bb: u64, pos: i32) -> u64 {
        self.magics.get_queen_attacks(empty_bb, pos)
    }

    // Returns the position of the smallest attacker or -1 if there is no attacker
    fn find_smallest_attacker(&self, empty_bb: u64, occupied_bb: u64, opp_color: Color, pos: i32) -> i32 {
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
            let bishop_attacks = self.magics.get_bishop_attacks(empty_bb, pos);
            let attacking_bishops = bishops & bishop_attacks;
            if attacking_bishops != 0 {
                return attacking_bishops.trailing_zeros() as i32;
            }

            // Check rooks
            let rooks = self.get_bitboard(R) & occupied_bb;
            let rook_attacks = self.magics.get_rook_attacks(empty_bb, pos);
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
            let bishop_attacks = self.magics.get_bishop_attacks(empty_bb, pos);
            let attacking_bishops = bishops & bishop_attacks;
            if attacking_bishops != 0 {
                return attacking_bishops.trailing_zeros() as i32;
            }

            // Check rooks
            let rooks = self.get_bitboard(-R) & occupied_bb;
            let rook_attacks = self.magics.get_rook_attacks(empty_bb, pos);
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

    pub fn is_legal_move(&mut self, color: Color, m: Move) -> bool {
        let (previous_piece, move_state) = self.perform_move(m);
        let is_legal = !self.is_in_check(color);
        self.undo_move(m, previous_piece, move_state);
        is_legal
    }

    // Return true, if the engine considers the current position as a draw.
    pub fn is_draw(&self) -> bool {
        self.pos_history.is_repetition_draw(self.state.hash, self.state.history_start)
            || self.is_fifty_move_draw()
            || self.is_insufficient_material_draw()
    }


    pub fn is_repetition_draw(&self) -> bool {
        self.pos_history.is_repetition_draw(self.state.hash, self.state.history_start)
    }

    pub fn is_fifty_move_draw(&self) -> bool {
        self.state.halfmove_clock >= 100
    }

    pub fn is_insufficient_material_draw(&self) -> bool {
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

    #[inline]
    pub fn is_pawn_move_close_to_promotion(&self, piece: i8, pos: i32, blockers: u64) -> bool {
        if piece == P {
            return (get_white_pawn_freepath(pos as i32) & blockers) == 0;

        } else if piece == -P {
            return (get_black_pawn_freepath(pos as i32) & blockers) == 0;
        }

        false
    }

    #[inline]
    pub fn is_pawn_endgame(&self) -> bool {
        (self.get_occupancy_bitboard() & !(self.get_bitboard(P) | self.get_bitboard(-P))).count_ones() == 2
    }

    /* Perform a Static Exchange Evaluation (SEE) to check, whether the net gain of the capture is still positive,
       after applying all immediate and discovered re-capture attacks.
    */
    pub fn has_negative_see(&mut self, opp_color: Color, from: i32, target: i32, own_piece_id: i8, captured_piece_id: i8, threshold: i16, mut occupied_bb: u64) -> bool {
        let mut score = get_piece_value(captured_piece_id as usize);
        occupied_bb &= !(1 << from as u64);
        let mut trophy_piece_score = get_piece_value(own_piece_id as usize);

        loop {
            let empty_bb = !occupied_bb;
            // Opponent attack
            let attacker_pos = self.find_smallest_attacker(empty_bb, occupied_bb, opp_color, target);
            if attacker_pos < 0 {
                return score < threshold;
            }
            score -= trophy_piece_score;
            trophy_piece_score = get_piece_value(self.get_item(attacker_pos).abs() as usize);
            if score + trophy_piece_score < 0 {
                return score < threshold;
            }

            occupied_bb &= !(1 << attacker_pos);

            // Own attack
            let own_attacker_pos = self.find_smallest_attacker(empty_bb, occupied_bb, -opp_color, target);
            if own_attacker_pos < 0 {
                return score < threshold;
            }

            score += trophy_piece_score;
            trophy_piece_score = get_piece_value(self.get_item(own_attacker_pos).abs() as usize);
            if score - trophy_piece_score > 0 {
                return score < threshold;
            }

            occupied_bb &= !(1 << own_attacker_pos);
        }
    }

    #[inline]
    fn set_king_pos(&mut self, color: Color, pos: i32) {
        unsafe { *self.king_pos.get_unchecked_mut(color.idx()) = pos };
    }

    #[inline]
    pub fn king_pos(&self, color: Color) -> i32 {
        unsafe { *self.king_pos.get_unchecked(color.idx()) }
    }

    pub fn reset_nn_eval(&mut self) {
        self.nn_eval.check_refresh();
    }

    pub fn eval(&mut self) -> i32 {
        self.nn_eval.eval(self.active_player(), self.halfmove_clock(), &self.bitboards)
    }

}

#[cfg(test)]
mod tests {
    use crate::moves::MoveType;

    use super::*;

    #[test]
    fn update_hash_when_piece_moves() {
        let items: [i8; 64] = [
            0, 0, 0, -K, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, K, 0, 0, 0, 0,
        ];

        let mut board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());

        board.recalculate_hash();
        let initial_hash = board.get_hash();

        let m = Move::new(MoveType::KingQuiet, K, 59, 60);
        let (previous, state) = board.perform_move(m);
        let hash_perform_move = board.get_hash();
        assert_ne!(initial_hash, hash_perform_move);

        board.undo_move(m, previous, state);
        let hash_undo_move = board.get_hash();
        assert_eq!(initial_hash, hash_undo_move);
    }

    #[test]
    fn incrementally_updates_hash() {
        let items: [i8; 64] = [
            0, 0, 0, -K, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, K, 0, 0, 0, 0,
        ];

        let mut board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        board.recalculate_hash();

        board.add_piece(WHITE, P, 48);
        board.add_piece(BLACK, R, 1);
        board.set_rook_moved(Castling::BlackQueenSide);
        board.increase_half_move_count();

        let hash_incremental = board.get_hash();
        board.recalculate_hash();
        let hash_recalculated = board.get_hash();

        assert_eq!(hash_incremental, hash_recalculated);
    }

    #[test]
    fn performs_and_undos_white_castling_moves() {
        let items: [i8; 64] = [
            -R, 0, 0, 0, -K, 0, 0, -R, -P, -P, -P, -P, -P, -P, -P, -P, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, P, P, P, P, P, P,
            P, P, R, 0, 0, 0, K, 0, 0, R,
        ];

        let mut board = Board::new(&items, WHITE, CastlingState::ALL, None, 0, 1, CastlingRules::default());

        let initial_items = board.items;
        let initial_hash = board.get_hash();
        let initial_castling_state = board.state.castling;

        let m = Move::new(MoveType::Castling, K, board.castling_rules.king_start(WHITE) as i32, board.castling_rules.ks_rook_start(WHITE));
        let (previous, state) = board.perform_move(m);

        assert_ne!(&initial_items[..], &board.items[..]);
        assert_ne!(initial_hash, board.get_hash());
        assert_ne!(initial_castling_state, board.state.castling);

        board.undo_move(m, previous, state);

        assert_eq!(&initial_items[..], &board.items[..]);
        assert_eq!(initial_hash, board.get_hash());
        assert_eq!(initial_castling_state, board.state.castling);
    }

    #[test]
    fn performs_and_undos_black_castling_moves() {
        let items: [i8; 64] = [
            -R, 0, 0, 0, -K, 0, 0, -R, -P, -P, -P, -P, -P, -P, -P, -P, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, P, P, P, P, P, P,
            P, P, R, N, B, Q, K, B, N, R,
        ];

        let mut board = Board::new(&items, BLACK, CastlingState::ALL, None, 0, 1, CastlingRules::default());

        let initial_items = board.items;
        let initial_hash = board.get_hash();
        let initial_castling_state = board.state.castling;

        let m = Move::new(MoveType::Castling, K, board.castling_rules.king_start(BLACK), board.castling_rules.ks_rook_start(BLACK));

        let (previous, state) = board.perform_move(m);

        assert_ne!(&initial_items[..], &board.items[..]);
        assert_ne!(initial_hash, board.get_hash());
        assert_ne!(initial_castling_state, board.state.castling);

        board.undo_move(m, previous, state);

        assert_eq!(&initial_items[..], &board.items[..]);
        assert_eq!(initial_hash, board.get_hash());
        assert_eq!(initial_castling_state, board.state.castling);
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            34,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            36,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            20,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), BLACK, 27)
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

        let board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            18,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), BLACK, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            37,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            45,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            24,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            29,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert_eq!(
            35,
            board.find_smallest_attacker(!board.get_occupancy_bitboard(), board.get_occupancy_bitboard(), WHITE, 27)
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
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

        let mut board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert!(!board.has_negative_see(BLACK, 52, 44, R, P, 0, board.get_occupancy_bitboard()));
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

        let mut board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert!(!board.has_negative_see(WHITE, 52, 44, R, P, 0, board.get_occupancy_bitboard()));
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

        let mut board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        let initial_hash = board.get_hash();

        board.perform_move(Move::new(MoveType::KingQuiet, K, 59, 60));
        let hash_after_move = board.get_hash();
        assert_ne!(initial_hash, hash_after_move);

        board.perform_move(Move::new(MoveType::KingQuiet, K, 60, 59));
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

        let mut board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
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
