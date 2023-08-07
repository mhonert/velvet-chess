/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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

use crate::bitboard::{
    black_left_pawn_attacks, black_right_pawn_attacks, get_king_attacks, get_knight_attacks, get_pawn_attacks,
    white_left_pawn_attacks, white_right_pawn_attacks, BitBoard, BitBoards, DARK_COLORED_FIELD_PATTERN,
    LIGHT_COLORED_FIELD_PATTERN,
};
use crate::board::castling::{Castling, CastlingRules, CastlingState, KING_SIDE_CASTLING, QUEEN_SIDE_CASTLING};
use crate::colors::{Color, BLACK, WHITE};
use crate::magics::{get_bishop_attacks, get_queen_attacks, get_rook_attacks};
use crate::moves::{Move, MoveType};
use crate::nn::eval::NeuralNetEval;
use crate::params;
use crate::pieces::{B, EMPTY, K, N, P, Q, R};
use crate::pos_history::PositionHistory;
use crate::transposition_table::MAX_DEPTH;
use crate::zobrist::{enpassant_zobrist_key, piece_zobrist_key, player_zobrist_key};

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
    pub bitboards: BitBoards,
    pub state: StateEntry,
    pub halfmove_count: u16,
    pub castling_rules: CastlingRules,

    nn_eval: Box<NeuralNetEval>,
    items: [i8; 64],
    king_pos: [i8; 2],

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
    pub fn new(
        items: &[i8], active_player: Color, castling_state: CastlingState, enpassant_target: Option<i8>,
        halfmove_clock: u8, fullmove_num: u16, castling_rules: CastlingRules,
    ) -> Self {
        assert_eq!(items.len(), 64, "Expected a vector with 64 elements, but got {}", items.len());

        let mut board = Board {
            pos_history: PositionHistory::default(),
            nn_eval: NeuralNetEval::new(),
            castling_rules,
            items: [0; 64],
            bitboards: BitBoards::default(),
            state: StateEntry {
                en_passant: 0,
                castling: CastlingState::default(),
                halfmove_clock: 0,
                hash: 0,
                history_start: 0,
            },
            king_pos: [0; 2],
            halfmove_count: 0,
            history: Vec::with_capacity(MAX_DEPTH),
        };

        board.set_position(
            items,
            active_player,
            castling_state,
            enpassant_target,
            halfmove_clock,
            fullmove_num,
            castling_rules,
        );
        board
    }

    pub fn reset(
        &mut self, pos_history: PositionHistory, bitboards: BitBoards, halfmove_count: u16, state: StateEntry,
        castling_rules: CastlingRules,
    ) {
        let white_king = bitboards.by_piece(K).piece_pos() as i8;
        let black_king = bitboards.by_piece(-K).piece_pos() as i8;

        self.castling_rules = castling_rules;
        self.pos_history = pos_history;
        self.bitboards = bitboards;
        self.state = state;
        self.halfmove_count = halfmove_count;
        self.king_pos = [white_king, black_king];
        self.history.clear();

        self.items.fill(0);

        for piece_id in 1i8..=6i8 {
            let piece = piece_id;
            for pos in bitboards.by_piece(piece) {
                self.items[pos as usize] = piece;
            }

            let piece = -piece_id;
            for pos in bitboards.by_piece(piece) {
                self.items[pos as usize] = piece;
            }
        }

        self.recalculate_hash();
        self.nn_eval.init_pos(&self.bitboards, self.king_pos(WHITE), self.king_pos(BLACK));
    }

    pub fn set_position(
        &mut self, items: &[i8], active_player: Color, castling_state: CastlingState, enpassant_target: Option<i8>,
        halfmove_clock: u8, fullmove_num: u16, castling_rules: CastlingRules,
    ) {
        self.pos_history.clear();
        assert_eq!(items.len(), 64, "Expected a vector with 64 elements, but got {}", items.len());

        self.halfmove_count = (max(1, fullmove_num) - 1) * 2 + if active_player.is_white() { 0 } else { 1 };
        self.state.halfmove_clock = halfmove_clock;
        self.state.history_start = halfmove_clock;
        self.state.hash = 0;
        self.state.castling = castling_state;
        self.state.en_passant = 0;
        self.castling_rules = castling_rules;

        if let Some(target) = enpassant_target {
            self.set_enpassant(target)
        }

        self.bitboards = BitBoards::default();
        self.items = [EMPTY; 64];

        for i in 0..64 {
            let item = items[i];

            if item != EMPTY {
                self.add_piece(Color::from_piece(item), item.abs(), i);
            } else {
                self.items[i] = item;
            }

            if item == K {
                self.set_king_pos(WHITE, i as i8);
            } else if item == -K {
                self.set_king_pos(BLACK, i as i8);
            }
        }


        self.nn_eval.init_pos(&self.bitboards, self.king_pos(WHITE), self.king_pos(BLACK));

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

        if self.active_player().is_black() {
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
        Color::from_halfmove_count(self.halfmove_count)
    }

    pub fn can_castle(&self, castling: Castling) -> bool {
        self.state.castling.can_castle(castling)
    }

    pub fn any_castling(&self) -> bool {
        self.state.castling.any_castling()
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

    pub fn piece_count(&self) -> u32 {
        self.occupancy_bb().piece_count()
    }

    pub fn can_enpassant(&self, color: Color, location: u8) -> bool {
        if color.is_white()
            && location >= WhiteBoardPos::EnPassantLineStart as u8
            && location <= WhiteBoardPos::EnPassantLineEnd as u8
        {
            return self.state.en_passant & (1 << (location - WhiteBoardPos::EnPassantLineStart as u8)) != 0;
        } else if color.is_black()
            && location >= BlackBoardPos::EnPassantLineStart as u8
            && location <= BlackBoardPos::EnPassantLineEnd as u8
        {
            return self.state.en_passant & (1 << (location - BlackBoardPos::EnPassantLineStart as u8 + 8)) != 0;
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
        let color = Color::from_piece(own_piece);

        self.clear_en_passant();

        match m.typ() {
            MoveType::PawnQuiet => {
                self.nn_eval.remove_add_piece(false, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);
                self.reset_half_move_clock();
            }

            MoveType::Quiet => {
                self.nn_eval.remove_add_piece(false, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);
            }

            MoveType::PawnDoubleQuiet => {
                self.nn_eval.remove_add_piece(false, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);
                self.reset_half_move_clock();
                self.set_enpassant(move_start as i8);
            }

            MoveType::Capture => {
                // Capture move (except en passant)
                let removed_piece = self.remove_piece(move_end);
                self.nn_eval.remove_remove_add_piece(removed_piece.abs() >= R, move_end as usize, removed_piece, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);

                self.reset_half_move_clock();


                return (own_piece, removed_piece.abs());
            }

            MoveType::KingCapture => {
                let removed_piece = self.remove_piece(move_end);
                self.nn_eval.remove_remove_add_piece(true, move_end as usize, removed_piece, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);

                self.reset_half_move_clock();

                self.set_king_pos(color, move_end as i8);
                self.set_king_moved(color);


                return (own_piece, removed_piece.abs());
            }

            MoveType::PawnSpecial => {
                self.reset_half_move_clock();

                if self.get_item(move_end) != EMPTY {
                    // Capture move with promotion
                    let removed_piece = self.remove_piece(move_end);
                    self.nn_eval.remove_remove_add_piece(true, move_end as usize, removed_piece, move_start as usize, own_piece, move_end as usize, color.piece(target_piece_id));
                    self.add_piece(color, target_piece_id, move_end as usize);

                    return (own_piece, removed_piece.abs());
                }

                self.add_piece(color, target_piece_id, move_end as usize);
                if own_piece == P {
                    // Special en passant handling
                    if move_start - move_end == 7 {
                        self.nn_eval.remove_remove_add_piece(false, move_start as usize + 1, -P, move_start as usize, own_piece, move_end as usize, own_piece);
                        self.remove_piece(move_start + 1);
                        return (own_piece, P);
                    } else if move_start - move_end == 9 {
                        self.nn_eval.remove_remove_add_piece(false, move_start as usize - 1, -P, move_start as usize, own_piece, move_end as usize, own_piece);
                        self.remove_piece(move_start - 1);
                        return (own_piece, P);
                    }
                } else if own_piece == -P {
                    // Special en passant handling
                    if move_start - move_end == -7 {
                        self.nn_eval.remove_remove_add_piece(false, move_start as usize - 1, P, move_start as usize, own_piece, move_end as usize, own_piece);
                        self.remove_piece(move_start - 1);
                        return (own_piece, P);
                    } else if move_start - move_end == -9 {
                        self.nn_eval.remove_remove_add_piece(false, move_start as usize + 1, P, move_start as usize, own_piece, move_end as usize, own_piece);
                        self.remove_piece(move_start + 1);
                        return (own_piece, P);
                    }
                }

                // Promotion
                self.nn_eval.remove_add_piece(true, move_start as usize, own_piece, move_end as usize, color.piece(target_piece_id));
            }

            MoveType::KingQuiet => {
                self.nn_eval.remove_add_piece(true, move_start as usize, own_piece, move_end as usize, own_piece);
                self.add_piece(color, target_piece_id, move_end as usize);
                self.set_king_pos(color, move_end as i8);
                self.set_king_moved(color);
            }

            MoveType::Castling => {
                self.remove_piece(move_end);
                self.set_has_castled(color);

                if self.castling_rules.is_ks_castling(color, move_end) {
                    self.nn_eval.remove_add_piece(false, move_start as usize, own_piece, CastlingRules::ks_king_end(color) as usize, own_piece);
                    self.nn_eval.remove_add_piece(true, move_end as usize, color.piece(R), CastlingRules::ks_rook_end(color) as usize, color.piece(R));

                    self.set_king_pos(color, CastlingRules::ks_king_end(color) as i8);
                    self.add_piece(color, K, CastlingRules::ks_king_end(color) as usize);
                    self.add_piece(color, R, CastlingRules::ks_rook_end(color) as usize);
                } else {
                    self.nn_eval.remove_add_piece(false, move_start as usize, own_piece, CastlingRules::qs_king_end(color) as usize, own_piece);
                    self.nn_eval.remove_add_piece(true, move_end as usize, color.piece(R), CastlingRules::qs_rook_end(color) as usize, color.piece(R));

                    self.set_king_pos(color, CastlingRules::qs_king_end(color) as i8);
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

        let color = Color::from_piece(piece);

        self.halfmove_count -= 1;
        self.restore_state();

        match m.typ() {
            MoveType::Quiet | MoveType::PawnQuiet | MoveType::PawnDoubleQuiet => {
                self.nn_eval.remove_add_piece(false, move_end as usize, piece, move_start as usize, piece);
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
            }

            MoveType::Capture => {
                self.nn_eval.remove_add_add_piece(removed_piece_id >= R, move_end as usize, piece, move_start as usize, piece, move_end as usize, color.flip().piece(removed_piece_id));
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
            }

            MoveType::KingCapture => {
                self.nn_eval.remove_add_add_piece(true, move_end as usize, piece, move_start as usize, piece, move_end as usize, color.flip().piece(removed_piece_id));
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
                self.set_king_pos(color, move_start as i8);
            }

            MoveType::PawnSpecial => {
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);

                if m.is_en_passant() {
                    let offset = if color.is_white() { 1 } else { -1 };
                    if (move_start - move_end).abs() == 7 {
                        self.nn_eval.remove_add_add_piece(false, move_end as usize, piece, move_start as usize, piece, (move_start + offset) as usize, color.flip().piece(P));

                        self.add_piece_without_inc_update(color.flip(), color.flip().piece(P), move_start + offset);
                    } else if (move_start - move_end).abs() == 9 {
                        self.nn_eval.remove_add_add_piece(false, move_end as usize, piece, move_start as usize, piece, (move_start - offset) as usize, color.flip().piece(P));
                        self.add_piece_without_inc_update(color.flip(), color.flip().piece(P), move_start - offset);
                    }
                } else if removed_piece_id != EMPTY {
                    self.nn_eval.remove_add_add_piece(true, move_end as usize, color.piece(m.piece_id()), move_start as usize, piece, move_end as usize, color.flip().piece(removed_piece_id));
                    self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
                } else if m.is_promotion() {
                    self.nn_eval.remove_add_piece(true, move_end as usize, color.piece(m.piece_id()), move_start as usize, piece);
                }
            }

            MoveType::KingQuiet => {
                self.nn_eval.remove_add_piece(true, move_end as usize, piece, move_start as usize, piece);
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.set_king_pos(color, move_start as i8);
            }

            MoveType::Castling => {
                if self.castling_rules.is_ks_castling(color, move_end) {
                    self.nn_eval.remove_add_piece(false, CastlingRules::ks_rook_end(color) as usize, color.piece(R), move_end as usize, color.piece(R));
                    self.nn_eval.remove_add_piece(true, CastlingRules::ks_king_end(color) as usize, piece, move_start as usize, piece);
                    self.remove_piece_without_inc_update(CastlingRules::ks_king_end(color));
                    self.remove_piece_without_inc_update(CastlingRules::ks_rook_end(color));
                } else {
                    self.nn_eval.remove_add_piece(false, CastlingRules::qs_rook_end(color) as usize, color.piece(R), move_end as usize, color.piece(R));
                    self.nn_eval.remove_add_piece(true, CastlingRules::qs_king_end(color) as usize, piece, move_start as usize, piece);
                    self.remove_piece_without_inc_update(CastlingRules::qs_king_end(color));
                    self.remove_piece_without_inc_update(CastlingRules::qs_rook_end(color));
                }

                self.add_piece_without_inc_update(color, color.piece(R), move_end);
                self.set_king_pos(color, move_start as i8);
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
        }

        self.bitboards.flip(color, piece, pos as u32);
    }

    pub fn add_piece(&mut self, color: Color, piece_id: i8, pos: usize) {
        let piece = color.piece(piece_id);

        unsafe {
            *self.items.get_unchecked_mut(pos) = piece;
        }

        self.state.hash ^= piece_zobrist_key(piece, pos);

        self.bitboards.flip(color, piece, pos as u32);
    }

    fn clear_en_passant(&mut self) {
        let previous_state = self.state.en_passant;

        if previous_state != 0 {
            self.state.en_passant = 0;
            self.update_hash_for_enpassant(previous_state);
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, pos: i32) -> i8 {
        let piece = self.get_item(pos);
        self.state.hash ^= piece_zobrist_key(piece, pos as usize);

        let color = Color::from_piece(piece);

        if piece.abs() == R {
            if self.castling_rules.is_ks_castling(color, pos) {
                self.set_rook_moved(KING_SIDE_CASTLING[color.idx()]);
            } else if self.castling_rules.is_qs_castling(color, pos) {
                self.set_rook_moved(QUEEN_SIDE_CASTLING[color.idx()]);
            }
        }

        self.remove(piece, color, pos)
    }

    fn remove_piece_without_inc_update(&mut self, pos: i32) {
        let piece = self.get_item(pos);
        let color = Color::from_piece(piece);
        self.remove(piece, color, pos);
    }

    #[inline]
    fn remove(&mut self, piece: i8, color: Color, pos: i32) -> i8 {
        unsafe {
            self.bitboards.flip(color, piece, pos as u32);
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

    #[inline(always)]
    pub fn is_in_check(&self, color: Color) -> bool {
        if color.is_white() {
            self.is_attacked(BLACK, self.king_pos(WHITE) as i32)
        } else {
            self.is_attacked(WHITE, self.king_pos(BLACK) as i32)
        }
    }

    pub fn get_all_piece_bitboard(&self, color: Color) -> BitBoard {
        self.bitboards.by_color(color)
    }

    pub fn occupancy_bb(&self) -> BitBoard {
        self.bitboards.occupancy()
    }

    #[inline(always)]
    pub fn is_attacked(&self, opp: Color, pos: i32) -> bool {
        let empty_bb = !self.occupancy_bb();
        let target_bb = BitBoard(1 << pos as u64);

        // Check knights
        if (self.get_bitboard(opp.piece(N)) & get_knight_attacks(pos)).is_occupied() {
            return true;
        }

        // Check diagonal
        let queens = self.get_bitboard(opp.piece(Q));
        if ((self.get_bitboard(opp.piece(B)) | queens) & self.get_bishop_attacks(empty_bb, pos)).is_occupied() {
            return true;
        }

        // Check orthogonal
        if ((self.get_bitboard(opp.piece(R)) | queens) & self.get_rook_attacks(empty_bb, pos)).is_occupied() {
            return true;
        }

        // Check pawns
        let pawns = self.get_bitboard(opp.piece(P));
        if (target_bb & get_pawn_attacks(pawns, opp)).is_occupied() {
            return true;
        }

        // Check king
        if (target_bb & get_king_attacks(self.king_pos(opp) as i32)).is_occupied() {
            return true;
        }

        false
    }

    pub fn get_bitboard(&self, piece: i8) -> BitBoard {
        self.bitboards.by_piece(piece)
    }

    #[inline]
    pub fn get_bishop_attacks(&self, empty_bb: BitBoard, pos: i32) -> BitBoard {
        BitBoard(get_bishop_attacks(empty_bb.into(), pos))
    }

    #[inline]
    pub fn get_rook_attacks(&self, empty_bb: BitBoard, pos: i32) -> BitBoard {
        BitBoard(get_rook_attacks(empty_bb.into(), pos))
    }

    #[inline]
    pub fn get_queen_attacks(&self, empty_bb: BitBoard, pos: i32) -> BitBoard {
        BitBoard(get_queen_attacks(empty_bb.into(), pos))
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
        match self.occupancy_bb().piece_count() {
            2 => true, // K vs K

            3 => {
                // K vs K+N or K vs K+B
                let knights_or_bishops =
                    self.get_bitboard(N) | self.get_bitboard(-N) | self.get_bitboard(B) | self.get_bitboard(-B);
                knights_or_bishops.is_occupied()
            }

            4 => {
                // Check for K+B vs K+B where bishops are on fields with the same color
                let white_bishops = self.get_bitboard(B);
                let black_bishops = self.get_bitboard(-B);

                ((white_bishops & LIGHT_COLORED_FIELD_PATTERN).is_occupied()
                    && (black_bishops & LIGHT_COLORED_FIELD_PATTERN).is_occupied())
                    || ((white_bishops & DARK_COLORED_FIELD_PATTERN).is_occupied()
                        && (black_bishops & DARK_COLORED_FIELD_PATTERN).is_occupied())
            }

            _ => false,
        }
    }

    #[inline]
    pub fn has_non_pawns(&self, player: Color) -> bool {
        (self.get_all_piece_bitboard(player) & !self.get_bitboard(player.piece(P))).piece_count() > 1
    }

    /* Perform a Static Exchange Evaluation (SEE) to check, whether the net gain of the capture is still positive,
       after applying all immediate and discovered re-capture attacks.
    */
    pub fn has_negative_see(
        &mut self, mut opp_color: Color, from: i32, target: i32, own_piece_id: i8, captured_piece_id: i8,
        threshold: i32, mut occupied: BitBoard,
    ) -> bool {
        let mut score = params::see_piece_values(captured_piece_id as usize);
        occupied = occupied & !(1 << from as u64);
        let mut potential_gain = params::see_piece_values(own_piece_id as usize);

        let mut attackers = self.find_attackers(!occupied, occupied, target);

        // Pieces blocking line of sight for ray-attacking pieces (B, R, Q)
        let los_blockers = (self.bitboards.occupancy()
            ^ (self.bitboards.by_piece(N)
                | self.bitboards.by_piece(-N)
                | self.bitboards.by_piece(K)
                | self.bitboards.by_piece(-K)))
            & (self.get_queen_attacks(BitBoard(!0), target));

        let mut own_turn = false;

        while let Some((attacker, attacker_value)) = self.find_smallest_attacker(attackers, opp_color) {
            score -= potential_gain;
            potential_gain = attacker_value;
            if score + potential_gain < 0 {
                break;
            }
            occupied ^= attacker;
            attackers ^= attacker;
            if (los_blockers & attacker).is_occupied() {
                attackers |= self.find_ray_attackers(!occupied, occupied, target);
            }

            own_turn = !own_turn;
            score = -score;
            opp_color = opp_color.flip();
        }

        if own_turn {
            -score < threshold
        } else {
            score < threshold
        }
    }

    #[inline(always)]
    fn find_smallest_attacker(&self, attackers: BitBoard, color: Color) -> Option<(BitBoard, i32)> {
        let pieces = self.get_bitboard(color.piece(P)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(P as usize)));
        }

        let pieces = self.get_bitboard(color.piece(N)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(N as usize)));
        }

        let pieces = self.get_bitboard(color.piece(B)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(B as usize)));
        }

        let pieces = self.get_bitboard(color.piece(R)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(R as usize)));
        }

        let pieces = self.get_bitboard(color.piece(Q)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(Q as usize)));
        }

        let pieces = self.get_bitboard(color.piece(K)) & attackers;
        if pieces.is_occupied() {
            return Some((pieces.first(), params::see_piece_values(K as usize)));
        }

        None
    }

    #[inline(always)]
    fn find_attackers(&self, empty_bb: BitBoard, occupied_bb: BitBoard, pos: i32) -> BitBoard {
        let target_bb = BitBoard(1 << pos as u64);
        let mut attackers = self.find_ray_attackers(empty_bb, occupied_bb, pos);

        let white_pawns = self.get_bitboard(P);
        attackers |= white_pawns & (black_left_pawn_attacks(target_bb));
        attackers |= white_pawns & (black_right_pawn_attacks(target_bb));

        let black_pawns = self.get_bitboard(-P);
        attackers |= black_pawns & (white_left_pawn_attacks(target_bb));
        attackers |= black_pawns & (white_right_pawn_attacks(target_bb));

        let knight_attacks = get_knight_attacks(pos);
        attackers |= (self.get_bitboard(N) | self.get_bitboard(-N)) & knight_attacks;

        let king_attacks = get_king_attacks(pos);
        attackers |= (self.get_bitboard(K) | self.get_bitboard(-K)) & king_attacks;

        attackers &= occupied_bb;

        attackers
    }

    #[inline(always)]
    fn find_ray_attackers(&self, empty_bb: BitBoard, occupied_bb: BitBoard, pos: i32) -> BitBoard {
        let mut attackers = BitBoard(0);

        let bishop_attacks = self.get_bishop_attacks(empty_bb, pos);
        let rook_attacks = self.get_rook_attacks(empty_bb, pos);

        attackers |= (self.get_bitboard(B) | self.get_bitboard(-B)) & bishop_attacks;
        attackers |= (self.get_bitboard(R) | self.get_bitboard(-R)) & rook_attacks;
        attackers |= (self.get_bitboard(Q) | self.get_bitboard(-Q)) & (bishop_attacks | rook_attacks);

        attackers &= occupied_bb;

        attackers
    }

    #[inline]
    fn set_king_pos(&mut self, color: Color, pos: i8) {
        unsafe { *self.king_pos.get_unchecked_mut(color.idx()) = pos };
    }

    #[inline]
    pub fn king_pos(&self, color: Color) -> i8 {
        unsafe { *self.king_pos.get_unchecked(color.idx()) }
    }

    pub fn reset_nn_eval(&mut self) {
        self.nn_eval.init_pos(&self.bitboards, self.king_pos(WHITE), self.king_pos(BLACK));
    }

    pub fn eval(&mut self) -> i32 {
        self.nn_eval.eval(
            self.active_player(),
            &self.bitboards,
            self.king_pos(WHITE),
            self.king_pos(BLACK),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Once;
    use crate::magics::initialize_attack_tables;
    use crate::moves::MoveType;

    use super::*;

    static INIT: Once = Once::new();

    pub fn initialize() {
        INIT.call_once(|| {
            initialize_attack_tables();
        })
    }

    #[test]
    fn update_hash_when_piece_moves() {
        initialize();
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
        initialize();
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
        initialize();
        #[rustfmt::skip]
        let items: [i8; 64] = [
            -R,  0,  0,  0, -K,  0,  0, -R,
            -P, -P, -P, -P, -P, -P, -P, -P,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            P,  P,  P,  P,  P,  P,  P,  P,
            R,  0,  0,  0,  K,  0,  0,  R,
        ];

        let mut board = Board::new(&items, WHITE, CastlingState::ALL, None, 0, 1, CastlingRules::default());

        let initial_items = board.items;
        let initial_hash = board.get_hash();
        let initial_castling_state = board.state.castling;

        let m = Move::new(
            MoveType::Castling,
            K,
            board.castling_rules.king_start(WHITE) as i32,
            board.castling_rules.ks_rook_start(WHITE),
        );
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
        initialize();
        #[rustfmt::skip]
        let items: [i8; 64] = [
            -R,  0,  0,  0, -K,  0,  0, -R,
            -P, -P, -P, -P, -P, -P, -P, -P,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             0,  0,  0,  0,  0,  0,  0,  0,
             P,  P,  P,  P,  P,  P,  P,  P,
             R,  N,  B,  Q,  K,  B,  N,  R,
        ];

        let mut board = Board::new(&items, BLACK, CastlingState::ALL, None, 0, 1, CastlingRules::default());

        let initial_items = board.items;
        let initial_hash = board.get_hash();
        let initial_castling_state = board.state.castling;

        let m = Move::new(
            MoveType::Castling,
            K,
            board.castling_rules.king_start(BLACK),
            board.castling_rules.ks_rook_start(BLACK),
        );

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
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 34), params::see_piece_values(P as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_white_pawn_right_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 36), params::see_piece_values(P as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_black_pawn_left_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 20), params::see_piece_values(P as usize))), board.find_smallest_attacker(attackers, BLACK));
    }

    #[test]
    fn find_black_pawn_right_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 18), params::see_piece_values(P as usize))), board.find_smallest_attacker(attackers, BLACK));
    }

    #[test]
    fn find_knight_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 37), params::see_piece_values(N as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_bishop_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 45), params::see_piece_values(B as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_rook_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 24), params::see_piece_values(R as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_queen_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 29), params::see_piece_values(Q as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn find_king_attack() {
        initialize();
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
        let attackers = board.find_attackers(!board.occupancy_bb(), board.occupancy_bb(), 27);
        assert_eq!(Some((BitBoard(1 << 35), params::see_piece_values(K as usize))), board.find_smallest_attacker(attackers, WHITE));
    }

    #[test]
    fn recognizes_white_in_check() {
        initialize();
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
        initialize();
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
        initialize();
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
        assert!(!board.has_negative_see(BLACK, 52, 44, R, P, 0, board.occupancy_bb()));
    }

    #[test]
    fn see_black_discovered_attacks() {
        initialize();
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
        assert!(!board.has_negative_see(WHITE, 52, 44, R, P, 0, board.occupancy_bb()));
    }

    #[test]
    fn updates_hash_for_piece_movements() {
        initialize();
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
        initialize();
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

        assert_ne!(initial_hash, board.get_hash(), "hash must be different if en passant flag is set");

        board.clear_en_passant();
        assert_eq!(initial_hash, board.get_hash(), "hash must be eq to initial hash if en passant flag is cleared");
    }
}
