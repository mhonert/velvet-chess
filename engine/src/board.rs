/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
pub mod cycledetection;

use std::cmp::max;

use crate::bitboard::{
    black_left_pawn_attacks, black_right_pawn_attacks, get_king_attacks, get_knight_attacks, get_pawn_attacks,
    white_left_pawn_attacks, white_right_pawn_attacks, BitBoard, BitBoards, DARK_COLORED_FIELD_PATTERN,
    LIGHT_COLORED_FIELD_PATTERN,
};
use crate::board::castling::{Castling, CastlingRules, CastlingState, KING_SIDE_CASTLING, QUEEN_SIDE_CASTLING};
use crate::colors::{Color, BLACK, WHITE};
use crate::magics::{get_bishop_attacks, get_rook_attacks};
use crate::moves::{Move, MoveType};
use crate::nn::eval::NeuralNetEval;
use crate::pieces::{B, EMPTY, K, N, P, Q, R};
use crate::pos_history::PositionHistory;
use crate::scores::clock_scaled_eval;
use crate::transposition_table::MAX_DEPTH;
use crate::zobrist::{enpassant_zobrist_key, piece_zobrist_key, player_zobrist_key};

#[repr(u8)]
pub enum WhiteBoardPos {
    PawnLineStart = 48,
    PawnLineEnd = 55,
    EnPassantLineStart = 16,
    EnPassantLineEnd = 23,
}

#[repr(u8)]
pub enum BlackBoardPos {
    PawnLineStart = 8,
    PawnLineEnd = 15,
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

    history: Vec<StateEntry>,
}

#[derive(Copy, Clone, Default)]
pub struct StateEntry {
    hash: u64,
    piece_hashes: PieceHashes,
    en_passant: u8,
    castling: CastlingState,
    halfmove_clock: u8,
    history_start: u8,
}

#[derive(Copy, Clone, Default)]
pub struct PieceHashes {
    pub pawn: u16,
    pub white_non_pawn: u16,
    pub black_non_pawn: u16,
}

impl PieceHashes {
    pub fn reset(&mut self) {
        self.pawn = 0;
        self.white_non_pawn = 0;
        self.black_non_pawn = 0;
    }
}

static SEE_PIECE_VALUES: [i16; 7] = [0, 98, 349, 350, 523, 1016, 8000];

#[inline(always)]
fn see_piece_value(piece: i8) -> i16 {
    unsafe { *SEE_PIECE_VALUES.get_unchecked(piece.abs() as usize) }
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
                piece_hashes: PieceHashes::default(),
                history_start: 0,
            },
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
        self.castling_rules = castling_rules;
        self.pos_history = pos_history;
        self.bitboards = bitboards;
        self.state = state;
        self.halfmove_count = halfmove_count;
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
            }
        }

        assert!((0..=63).contains(&self.king_pos(WHITE)), "Cannot set position with missing white king");
        assert!((0..=63).contains(&self.king_pos(BLACK)), "Cannot set position with missing black king");

        self.nn_eval.init_pos(&self.bitboards, self.king_pos(WHITE), self.king_pos(BLACK));

        self.recalculate_hash();
    }

    pub fn recalculate_hash(&mut self) {
        self.state.hash = 0;
        self.state.piece_hashes.reset();

        for pos in 0..64 {
            let piece = self.items[pos];
            if piece == EMPTY {
                continue;
            }
            self.update_piece_hashes(piece, piece_zobrist_key(piece, pos));
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

        self.state.en_passant = pos as u8;
        self.update_hash_for_enpassant(previous_state);
    }

    fn update_hash_for_enpassant(&mut self, previous_state: u8) {
        if previous_state != 0 {
            self.state.hash ^= enpassant_zobrist_key(previous_state);
        }

        if self.state.en_passant != 0 {
            self.state.hash ^= enpassant_zobrist_key(self.state.en_passant);
        }
    }

    pub fn get_item(&self, pos: usize) -> i8 {
        unsafe { *self.items.get_unchecked(pos) }
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

    pub fn piece_count(&self) -> u32 {
        self.occupancy_bb().piece_count()
    }

    pub fn enpassant_target(&self) -> u8 {
        self.state.en_passant
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
        let color = self.active_player();
        self.store_state();
        self.increase_half_move_count();

        let move_start = m.start() as usize;
        let move_end = m.end() as usize;
        let target_piece_id = m.move_type().piece_id();

        self.clear_en_passant();

        match m.move_type() {
            MoveType::PawnQuiet => {
                let own_piece = color.piece(P);
                self.move_piece(color, own_piece, move_start, move_end);
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, own_piece);
                self.reset_half_move_clock();
                (own_piece, EMPTY)
            }
            MoveType::PawnDoubleQuiet => {
                let own_piece = color.piece(P);
                self.move_piece(color, own_piece, move_start, move_end);
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, own_piece);
                self.reset_half_move_clock();

                self.set_enpassant(move_start as i8 + if color.is_white() { -8 } else { 8 } );
                (own_piece, EMPTY)
            }
            MoveType::PawnEnPassant => {
                let own_piece = self.remove_piece(move_start);
                self.reset_half_move_clock();
                self.add_piece(color, target_piece_id, move_end);
                if own_piece == P {
                    // Special en passant handling
                    if move_start - move_end == 7 {
                        self.nn_eval.remove_remove_add_piece(move_start + 1, -P, move_start, own_piece, move_end, own_piece);
                        self.remove_piece(move_start + 1);
                        (own_piece, P)
                    } else {
                        self.nn_eval.remove_remove_add_piece(move_start - 1, -P, move_start, own_piece, move_end, own_piece);
                        self.remove_piece(move_start - 1);
                        (own_piece, P)
                    }
                } else {
                    // Special en passant handling
                    if (move_start as i64) - (move_end as i64) == -7 {
                        self.nn_eval.remove_remove_add_piece(move_start - 1, P, move_start , own_piece, move_end, own_piece);
                        self.remove_piece(move_start - 1);
                        (own_piece, P)
                    } else {
                        self.nn_eval.remove_remove_add_piece(move_start + 1, P, move_start, own_piece, move_end, own_piece);
                        self.remove_piece(move_start + 1);
                        (own_piece, P)
                    }
                }
            }
            MoveType::KnightQuiet | MoveType::BishopQuiet | MoveType::QueenQuiet => {
                let own_piece = color.piece(target_piece_id);
                self.move_piece(color, own_piece, move_start, move_end);
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, own_piece);
                (own_piece, EMPTY)
            },
            MoveType::RookQuiet => {
                let own_piece = color.piece(target_piece_id);
                self.move_piece(color, own_piece, move_start, move_end);
                if self.castling_rules.is_ks_castling(color, move_start as i8) {
                    self.set_rook_moved(KING_SIDE_CASTLING[color.idx()]);
                } else if self.castling_rules.is_qs_castling(color, move_start as i8) {
                    self.set_rook_moved(QUEEN_SIDE_CASTLING[color.idx()]);
                }
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, own_piece);
                (own_piece, EMPTY)
            }
            MoveType::PawnCapture => {
                let own_piece = self.remove_piece(move_start);
                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, own_piece);
                self.add_piece(color, target_piece_id, move_end);

                self.reset_half_move_clock();
                (own_piece, removed_piece.abs())
            }
            MoveType::KnightCapture | MoveType::BishopCapture => {
                let own_piece = self.remove_piece(move_start);
                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, own_piece);
                self.add_piece(color, target_piece_id, move_end);

                self.reset_half_move_clock();
                (own_piece, removed_piece.abs())
            },
            MoveType::RookCapture => {
                let own_piece = self.remove_piece(move_start);
                if self.castling_rules.is_ks_castling(color, move_start as i8) {
                    self.set_rook_moved(KING_SIDE_CASTLING[color.idx()]);
                } else if self.castling_rules.is_qs_castling(color, move_start as i8) {
                    self.set_rook_moved(QUEEN_SIDE_CASTLING[color.idx()]);
                }

                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, own_piece);
                self.add_piece(color, target_piece_id, move_end);

                self.reset_half_move_clock();
                (own_piece, removed_piece.abs())
            },
            MoveType::QueenCapture => {
                let own_piece = self.remove_piece(move_start);
                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, own_piece);
                self.add_piece(color, target_piece_id, move_end);

                self.reset_half_move_clock();
                (own_piece, removed_piece.abs())
            }
            MoveType::KnightQuietPromotion | MoveType::BishopQuietPromotion | MoveType::RookQuietPromotion | MoveType::QueenQuietPromotion => {
                let own_piece = self.remove_piece(move_start);
                self.reset_half_move_clock();
                self.add_piece(color, target_piece_id, move_end);
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, color.piece(target_piece_id));
                (own_piece, EMPTY)
            }
            MoveType::KnightCapturePromotion | MoveType::BishopCapturePromotion | MoveType::RookCapturePromotion | MoveType::QueenCapturePromotion => {
                let own_piece = self.remove_piece(move_start);
                self.reset_half_move_clock();

                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, color.piece(target_piece_id));
                self.add_piece(color, target_piece_id, move_end);

                (own_piece, removed_piece.abs())
            }
            MoveType::KingQuiet => {
                let own_piece = color.piece(target_piece_id);
                self.move_piece(color, own_piece, move_start, move_end);
                self.nn_eval.remove_add_piece(move_start, own_piece, move_end, own_piece);
                self.set_king_moved(color);
                (own_piece, EMPTY)
            }
            MoveType::KingCapture => {
                let own_piece = self.remove_piece(move_start);
                let removed_piece = self.remove_piece(move_end);
                self.update_rook_castling_state(color.flip(), removed_piece.abs(), move_end as i8);
                self.nn_eval.remove_remove_add_piece(move_end, removed_piece, move_start, own_piece, move_end, own_piece);
                self.add_piece(color, target_piece_id, move_end);

                self.reset_half_move_clock();

                self.set_king_moved(color);

                (own_piece, removed_piece.abs())
            }
            MoveType::KingQSCastling => {
                let own_piece = self.remove_piece(move_start);
                self.remove_piece(move_end);
                self.set_has_castled(color);
                self.add_piece(color, K, CastlingRules::qs_king_end(color) as usize);
                self.add_piece(color, R, CastlingRules::qs_rook_end(color) as usize);
                self.set_rook_moved(QUEEN_SIDE_CASTLING[color.idx()]);

                self.nn_eval.remove_add_piece(move_end, color.piece(R), CastlingRules::qs_rook_end(color) as usize, color.piece(R));
                self.nn_eval.remove_add_piece(move_start, own_piece, CastlingRules::qs_king_end(color) as usize, own_piece);

                (own_piece, EMPTY)
            }
            MoveType::KingKSCastling => {
                let own_piece = self.remove_piece(move_start);
                self.remove_piece(move_end);
                self.set_has_castled(color);
                self.add_piece(color, K, CastlingRules::ks_king_end(color) as usize);
                self.add_piece(color, R, CastlingRules::ks_rook_end(color) as usize);
                self.set_rook_moved(KING_SIDE_CASTLING[color.idx()]);

                self.nn_eval.remove_add_piece(move_end, color.piece(R), CastlingRules::ks_rook_end(color) as usize, color.piece(R));
                self.nn_eval.remove_add_piece(move_start, own_piece, CastlingRules::ks_king_end(color) as usize, own_piece);

                (own_piece, EMPTY)
            }

            _ => {
                (EMPTY, EMPTY)
            }
        }
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

        let move_start = m.start() as usize;
        let move_end = m.end() as usize;

        let color = Color::from_piece(piece);

        self.halfmove_count -= 1;
        self.restore_state();

        match m.move_type() {
            MoveType::PawnQuiet | MoveType::PawnDoubleQuiet | MoveType::KnightQuiet | MoveType::BishopQuiet | MoveType::RookQuiet | MoveType::QueenQuiet => {
                self.nn_eval.remove_add_piece(move_end , piece, move_start , piece);
                self.move_piece_without_state(color, piece, move_end , move_start );
            }
            MoveType::PawnCapture | MoveType::KnightCapture | MoveType::BishopCapture | MoveType::RookCapture | MoveType::QueenCapture => {
                self.nn_eval.remove_add_add_piece(move_end , piece, move_start , piece, move_end , color.flip().piece(removed_piece_id));

                self.remove_piece_without_inc_update(move_end );
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
            }
            MoveType::PawnEnPassant => {
                self.remove_piece_without_inc_update(move_end );
                self.add_piece_without_inc_update(color, piece, move_start);
                if (move_start as i32 - move_end as i32).abs() == 7 {
                    let capture_pos = if color.is_white() { move_start + 1 } else { move_start - 1 };
                    self.nn_eval.remove_add_add_piece( move_end , piece, move_start , piece, capture_pos, color.flip().piece(P));
                    self.add_piece_without_inc_update(color.flip(), color.flip().piece(P), capture_pos);
                } else {
                    let capture_pos = if color.is_white() { move_start - 1 } else { move_start + 1 };
                    self.nn_eval.remove_add_add_piece( move_end , piece, move_start , piece, capture_pos, color.flip().piece(P));
                    self.add_piece_without_inc_update(color.flip(), color.flip().piece(P), capture_pos);
                }
            }
            MoveType::KnightQuietPromotion | MoveType::BishopQuietPromotion | MoveType::RookQuietPromotion | MoveType::QueenQuietPromotion => {
                self.remove_piece_without_inc_update(move_end );
                self.add_piece_without_inc_update(color, piece, move_start);
                self.nn_eval.remove_add_piece(move_end , color.piece(m.move_type().piece_id()), move_start , piece);
            }
            MoveType::KnightCapturePromotion | MoveType::BishopCapturePromotion | MoveType::RookCapturePromotion | MoveType::QueenCapturePromotion => {
                self.remove_piece_without_inc_update(move_end );
                self.add_piece_without_inc_update(color, piece, move_start);
                self.nn_eval.remove_add_add_piece(move_end , color.piece(m.move_type().piece_id()), move_start , piece, move_end , color.flip().piece(removed_piece_id));
                self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
            }
            MoveType::KingQuiet => {
                self.nn_eval.remove_add_piece(move_end , piece, move_start , piece);
                self.move_piece_without_state(color, piece, move_end, move_start);
            }
            MoveType::KingCapture => {
                self.nn_eval.remove_add_add_piece(move_end , piece, move_start , piece, move_end , color.flip().piece(removed_piece_id));
                self.remove_piece_without_inc_update(move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
                self.add_piece_without_inc_update(color.flip(), color.flip().piece(removed_piece_id), move_end);
            }
            MoveType::KingQSCastling => {
                self.nn_eval.remove_add_piece(CastlingRules::qs_king_end(color) as usize, piece, move_start , piece);
                self.nn_eval.remove_add_piece(CastlingRules::qs_rook_end(color) as usize, color.piece(R), move_end , color.piece(R));

                self.remove_piece_without_inc_update(CastlingRules::qs_king_end(color) as usize);
                self.remove_piece_without_inc_update(CastlingRules::qs_rook_end(color) as usize);

                self.add_piece_without_inc_update(color, color.piece(R), move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
            }
            MoveType::KingKSCastling => {
                self.nn_eval.remove_add_piece(CastlingRules::ks_king_end(color) as usize, piece, move_start , piece);
                self.nn_eval.remove_add_piece(CastlingRules::ks_rook_end(color) as usize, color.piece(R), move_end , color.piece(R));

                self.remove_piece_without_inc_update(CastlingRules::ks_king_end(color) as usize);
                self.remove_piece_without_inc_update(CastlingRules::ks_rook_end(color) as usize);

                self.add_piece_without_inc_update(color, color.piece(R), move_end);
                self.add_piece_without_inc_update(color, piece, move_start);
            }

            _ => {}
        }
    }

    #[inline(always)]
    pub fn piece_hashes(&self) -> PieceHashes {
        self.state.piece_hashes 
    }

    pub fn undo_null_move(&mut self) {
        self.halfmove_count -= 1;
        self.restore_state();
    }

    fn add_piece_without_inc_update(&mut self, color: Color, piece: i8, pos: usize) {
        unsafe {
            *self.items.get_unchecked_mut(pos) = piece;
        }

        self.bitboards.flip(color, piece, pos as u32);
    }

    pub fn add_piece(&mut self, color: Color, piece_id: i8, pos: usize) {
        let piece = color.piece(piece_id);

        unsafe {
            *self.items.get_unchecked_mut(pos) = piece;
        }

        self.update_piece_hashes(piece, piece_zobrist_key(piece, pos));

        self.bitboards.flip(color, piece, pos as u32);
    }

    #[inline(always)]
    fn update_piece_hashes(&mut self, piece: i8, key: u64) {
        self.state.hash ^= key;

        let key16 = (key & 0xFFFF) as u16;

        if piece.abs() == 1 {
            self.state.piece_hashes.pawn ^= key16;
            return;
        }

        if piece < -1 {
            self.state.piece_hashes.black_non_pawn ^= key16;
        } else if piece > 1 {
            self.state.piece_hashes.white_non_pawn ^= key16;
        }
    }

    #[inline(always)]
    fn move_piece(&mut self, color: Color, piece: i8, start: usize, end: usize) {
        unsafe {
            *self.items.get_unchecked_mut(start) = EMPTY;
            *self.items.get_unchecked_mut(end) = piece;
        }

        self.update_piece_hashes(piece, piece_zobrist_key(piece, start));
        self.update_piece_hashes(piece, piece_zobrist_key(piece, end));

        self.bitboards.flip2(color, piece, start as u32, end as u32);
    }

    #[inline]
    fn move_piece_without_state(&mut self, color: Color, piece: i8, start: usize, end: usize) {
        unsafe {
            *self.items.get_unchecked_mut(start) = EMPTY;
            *self.items.get_unchecked_mut(end) = piece;
        }

        self.bitboards.flip2(color, piece, start as u32, end as u32);
    }

    fn clear_en_passant(&mut self) {
        let previous_state = self.state.en_passant;

        if previous_state != 0 {
            self.state.en_passant = 0;
            self.update_hash_for_enpassant(previous_state);
        }
    }

    #[inline(always)]
    pub fn remove_piece(&mut self, pos: usize) -> i8 {
        let piece = self.get_item(pos);

        self.update_piece_hashes(piece, piece_zobrist_key(piece, pos));
        let color = Color::from_piece(piece);

        self.remove(piece, color, pos)
    }

    #[inline(always)]
    fn update_rook_castling_state(&mut self, color: Color, piece_id: i8, pos: i8) {
        if piece_id == R {
            if self.castling_rules.is_ks_castling(color, pos) {
                self.set_rook_moved(KING_SIDE_CASTLING[color.idx()]);
            } else if self.castling_rules.is_qs_castling(color, pos) {
                self.set_rook_moved(QUEEN_SIDE_CASTLING[color.idx()]);
            }
        }
    }

    fn remove_piece_without_inc_update(&mut self, pos: usize) {
        let piece = self.get_item(pos);
        let color = Color::from_piece(piece);
        self.remove(piece, color, pos);
    }

    #[inline]
    fn remove(&mut self, piece: i8, color: Color, pos: usize) -> i8 {
        unsafe {
            self.bitboards.flip(color, piece, pos as u32);
            *self.items.get_unchecked_mut(pos) = EMPTY;
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
        self.is_attacked(color.flip(), self.king_pos(color) as usize)
    }

    #[inline(always)]
    pub fn is_left_in_check(&self, color: Color, was_in_check: bool, m: Move) -> bool {
        if was_in_check || m.move_type().piece_id() == K || m.move_type().is_en_passant() {
            return self.is_in_check(color);
        }

        let opp = color.flip();
        let empty_bb = !self.occupancy_bb();
        let queens = self.get_bitboard(opp.piece(Q));
        let king_pos = self.king_pos(color) as usize;
        let diagonal_attacks = (self.get_bitboard(opp.piece(B)) | queens) & get_bishop_attacks(empty_bb.0, king_pos);
        let orthogonal_attacks = (self.get_bitboard(opp.piece(R)) | queens) & get_rook_attacks(empty_bb.0, king_pos);

        (diagonal_attacks | orthogonal_attacks).is_occupied()
    }

    pub fn get_all_piece_bitboard(&self, color: Color) -> BitBoard {
        self.bitboards.by_color(color)
    }

    pub fn occupancy_bb(&self) -> BitBoard {
        self.bitboards.occupancy()
    }

    #[inline(always)]
    pub fn is_attacked(&self, opp: Color, pos: usize) -> bool {
        let empty_bb = !self.occupancy_bb();
        let target_bb = BitBoard(1 << pos as u64);

        let knight_attacks = self.get_bitboard(opp.piece(N)) & get_knight_attacks(pos);

        let queens = self.get_bitboard(opp.piece(Q));
        let diagonal_attacks = (self.get_bitboard(opp.piece(B)) | queens) & get_bishop_attacks(empty_bb.0, pos);

        let orthogonal_attacks = (self.get_bitboard(opp.piece(R)) | queens) & get_rook_attacks(empty_bb.0, pos);

        let pawns = self.get_bitboard(opp.piece(P));
        let pawn_attacks = get_pawn_attacks(pawns, opp) & target_bb;

        let king_attacks = get_king_attacks(self.king_pos(opp) as usize) & target_bb;
        (knight_attacks | diagonal_attacks | orthogonal_attacks | pawn_attacks | king_attacks).is_occupied()
    }

    pub fn get_bitboard(&self, piece: i8) -> BitBoard {
        self.bitboards.by_piece(piece)
    }

    pub fn is_legal_move(&mut self, color: Color, m: Move) -> bool {
        let (previous_piece, move_state) = self.perform_move(m);
        let is_legal = !self.is_in_check(color);
        self.undo_move(m, previous_piece, move_state);
        is_legal
    }

    pub fn is_repetition_draw(&self) -> bool {
        self.pos_history.is_repetition_draw(self.state.hash, self.state.history_start)
    }
    
    pub fn has_upcoming_repetition(&self) -> bool {
        self.pos_history.has_upcoming_repetition(self.occupancy_bb(), self.state.hash, self.state.history_start)
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
       after applying all immediate and revealed re-capture attacks.
    */
    pub fn has_negative_see(
        &self, mut opp_color: Color, start: usize, end: usize, own_piece_id: i8, captured_piece_id: i8, mut occupied: BitBoard,
    ) -> bool {
        let mut score = see_piece_value(captured_piece_id);
        occupied = occupied & !(1 << start as u64);
        let mut potential_gain = see_piece_value(own_piece_id);

        let mut attackers = self.find_attackers(!occupied, occupied, end);
        let all_bishops = self.get_bitboard(B) | self.get_bitboard(-B);
        let all_rooks = self.get_bitboard(R) | self.get_bitboard(-R);
        let all_queens = self.get_bitboard(Q) | self.get_bitboard(-Q);
        let all_diagonal = all_bishops | all_queens;
        let all_orthogonal = all_rooks | all_queens;

        let mut own_turn = false;

        loop {
            let (attacker_piece_id, attacker_bb) =
                if let Some(bb) = (self.get_bitboard(opp_color.piece(P)) & attackers).first() {
                    (P, bb)
                } else if let Some(bb) = (self.get_bitboard(opp_color.piece(N)) & attackers).first() {
                    (N, bb)
                } else if let Some(bb) = (self.get_bitboard(opp_color.piece(B)) & attackers).first() {
                    (B, bb)
                } else if let Some(bb) = (self.get_bitboard(opp_color.piece(R)) & attackers).first() {
                    (R, bb)
                } else if let Some(bb) = (self.get_bitboard(opp_color.piece(Q)) & attackers).first() {
                    (Q, bb)
                } else if let Some(bb) = (self.get_bitboard(opp_color.piece(K)) & attackers).first() {
                    (K, bb)
                } else {
                    break;
                };

            score -= potential_gain;
            potential_gain = see_piece_value(attacker_piece_id);
            if score + potential_gain < 0 {
                break;
            }
            occupied ^= attacker_bb;
            attackers ^= attacker_bb;
            
            // Add new revealed attackers
            attackers |= occupied & all_diagonal & get_bishop_attacks((!occupied).0, end);
            attackers |= occupied & all_orthogonal & get_rook_attacks((!occupied).0, end);

            own_turn = !own_turn;
            score = -score;
            opp_color = opp_color.flip();
        }

        if own_turn {
            -score < 0
        } else {
            score < 0
        }
    }

    #[inline(always)]
    fn find_attackers(&self, empty_bb: BitBoard, occupied_bb: BitBoard, pos: usize) -> BitBoard {
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
    fn find_ray_attackers(&self, empty_bb: BitBoard, occupied_bb: BitBoard, pos: usize) -> BitBoard {
        let bishop_attacks = get_bishop_attacks(empty_bb.0, pos);
        let rook_attacks = get_rook_attacks(empty_bb.0, pos);

        let attackers = ((self.get_bitboard(B) | self.get_bitboard(-B)) & bishop_attacks) |
                                 ((self.get_bitboard(R) | self.get_bitboard(-R)) & rook_attacks) |
                                 ((self.get_bitboard(Q) | self.get_bitboard(-Q)) & (bishop_attacks | rook_attacks));

        attackers & occupied_bb
    }

    #[inline(always)]
    pub fn king_pos(&self, color: Color) -> i8 {
        self.get_bitboard(color.piece(K)).piece_pos() as i8
    }

    pub fn reset_nn_eval(&mut self) {
        self.nn_eval.init_pos(&self.bitboards, self.king_pos(WHITE), self.king_pos(BLACK));
    }

    pub fn eval(&mut self) -> i16 {
        self.nn_eval.eval(
            self.active_player(),
            &self.bitboards,
            self.king_pos(WHITE),
            self.king_pos(BLACK),
        )
    }

    pub fn clock_scaled_eval(&mut self) -> i16 {
        clock_scaled_eval(self.halfmove_clock(), self.eval())
    }
}


#[cfg(test)]
mod tests {
    use crate::init::init;
    use crate::moves::MoveType;
    use super::*;

    #[test]
    fn update_hash_when_piece_moves() {
        init();
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

        let m = Move::new(MoveType::KingQuiet, 59, 60);
        let (previous, state) = board.perform_move(m);
        let hash_perform_move = board.get_hash();
        assert_ne!(initial_hash, hash_perform_move);

        board.undo_move(m, previous, state);
        let hash_undo_move = board.get_hash();
        assert_eq!(initial_hash, hash_undo_move);
    }

    #[test]
    fn incrementally_updates_hash() {
        init();
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
        init();
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
            MoveType::KingKSCastling,
            board.castling_rules.king_start(WHITE),
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
        init();
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
            MoveType::KingKSCastling,
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
    fn recognizes_white_in_check() {
        init();
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
        init();
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
        init();
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

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert!(!board.has_negative_see(BLACK, 52, 44, R, P, board.occupancy_bb()));
    }

    #[test]
    fn see_black_discovered_attacks() {
        init();
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

        let board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
        assert!(!board.has_negative_see(WHITE, 52, 44, R, P, board.occupancy_bb()));
    }

    #[test]
    fn updates_hash_for_piece_movements() {
        init();
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

        let mut board = Board::new(&items, BLACK, CastlingState::default(), None, 0, 1, CastlingRules::default());
        let initial_hash = board.get_hash();

        board.perform_move(Move::new(MoveType::KingQuiet, 59, 60));
        let hash_after_move = board.get_hash();
        assert_ne!(initial_hash, hash_after_move);
        board.perform_null_move();

        board.perform_move(Move::new(MoveType::KingQuiet, 60, 59));
        board.perform_null_move();
        let hash_reverted_move = board.get_hash();
        assert_eq!(initial_hash, hash_reverted_move);
    }

    #[test]
    fn updates_hash_for_en_passant_changes() {
        init();
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
