/*
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

use crate::bitboard::{BLACK_KING_SIDE_CASTLING_BIT_PATTERN, BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN, PAWN_DOUBLE_MOVE_LINES, WHITE_KING_SIDE_CASTLING_BIT_PATTERN, WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN, BitBoard, get_knight_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks, get_king_attacks};
use crate::board::{Board, interpolate_score};
use crate::boardpos::{BlackBoardPos, WhiteBoardPos};
use crate::castling::Castling;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, K, N, P, Q, R, EMPTY};
use crate::moves::{Move, MoveType, NO_MOVE};
use crate::history_heuristics::{HistoryHeuristics};
use crate::score_util::{unpack_score, unpack_eg_score};
use crate::transposition_table::MAX_DEPTH;

const CAPTURE_ORDER_SIZE: usize = 5 + 5 * 6 + 1;

const PRIMARY_KILLER_SCORE: i32 = -2267;
const SECONDARY_KILLER_SCORE: i32 = -3350;

pub const NEGATIVE_HISTORY_SCORE: i32 = -5000;

const CAPTURE_ORDER_SCORES: [i32; CAPTURE_ORDER_SIZE] = calc_capture_order_scores();

pub struct MoveGenerator {
    entries: Vec<MoveList>,
    ply: usize,
}

impl MoveGenerator {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(MAX_DEPTH + 1);
        for _ in 0..MAX_DEPTH + 1 {
            entries.push(MoveList::new());
        }

        MoveGenerator {
            entries,
            ply: 0
        }
    }

    #[inline(always)]
    pub fn enter_ply(&mut self, active_player: Color, scored_hash_move: Move, primary_killer: Move, secondary_killer: Move) {
        self.ply += 1;
        self.entries[self.ply].init(active_player, scored_hash_move, primary_killer, secondary_killer);
    }

    #[inline(always)]
    pub fn leave_ply(&mut self) {
        self.ply -= 1;
    }

    pub fn reset(&mut self) {
        self.entries[self.ply].reset();
    }

    pub fn resort(&mut self, best_move: Move) {
        self.entries[self.ply].resort(best_move);
    }

    #[inline(always)]
    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_move(hh, board)
    }

    pub fn next_legal_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_legal_move(hh, board)
    }

    #[inline(always)]
    pub fn next_capture_move(&mut self, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_capture_move(board)
    }

    pub fn update_root_move(&mut self, m: Move) {
        self.entries[self.ply].update_root_move(m);
    }

}

enum Stage {
    HashMove,
    CaptureMoves,
    QuietMoves
}

pub struct MoveList {
    scored_hash_move: Move,
    primary_killer: Move,
    secondary_killer: Move,
    moves: Vec<Move>, // contains all moves on root level, but only quiet moves in all other cases
    capture_moves: Vec<Move>, // not used on root level
    stage: Stage,
    move_index: usize,
    capture_index: usize,
    moves_generated: bool,
    quiets_scored: bool,
    quiets_sorted: bool,
    active_player: Color,
    phase: i32,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList{
            scored_hash_move: NO_MOVE,
            primary_killer: NO_MOVE,
            secondary_killer: NO_MOVE,
            moves: Vec::with_capacity(64),
            capture_moves: Vec::with_capacity(16),
            stage: Stage::HashMove,
            move_index: 0,
            capture_index: 0,
            moves_generated: false,
            quiets_scored: false,
            quiets_sorted: false,
            active_player: WHITE,
            phase: 0,
        }
    }

    pub fn init(&mut self, active_player: Color, scored_hash_move: Move, primary_killer: Move, secondary_killer: Move) {
        self.scored_hash_move = scored_hash_move;
        self.primary_killer = primary_killer;
        self.secondary_killer = secondary_killer;

        self.moves.clear();
        self.capture_moves.clear();
        self.moves_generated = false;
        self.active_player = active_player;
        self.move_index = 0;
        self.capture_index = 0;
        self.quiets_scored = false;
        self.quiets_sorted = false;
        self.stage = Stage::HashMove;
    }

    pub fn reset(&mut self) {
        self.stage = Stage::HashMove;
        self.move_index = 0;
        self.capture_index = 0;
    }

    pub fn resort(&mut self, best_move: Move) {
        let mut best_move_idx = 0;
        for i in 0..self.moves.len() {
            if self.moves[i].is_same_move(best_move) {
                best_move_idx = i;
                break;
            }
        }

        self.moves.remove(best_move_idx);
        self.moves.insert(0, best_move);
    }

    #[inline]
    pub fn add_moves(&mut self, typ: MoveType, piece: i8, pos: i32, target_bb: u64) {
        for end in BitBoard(target_bb) {
            self.add_move(typ, piece, pos, end as i32);
        }
    }

    #[inline]
    pub fn add_move(&mut self, typ: MoveType, piece: i8, start: i32, end: i32) {
        self.moves.push(Move::new(typ, piece, start, end));
    }

    pub fn update_root_move(&mut self, scored_move: Move) {
        self.moves[self.move_index - 1] = scored_move;
    }

    #[inline]
    pub fn add_capture_moves(&mut self, board: &Board, typ: MoveType, piece: i8, pos: i32, target_bb: u64) {
        for end in BitBoard(target_bb) {
            self.add_capture_move(board, typ, piece, pos, end as i32);
        }
    }

    #[inline]
    pub fn add_capture_move(&mut self, board: &Board, typ: MoveType, piece: i8, start: i32, end: i32) {
        let m = Move::new(typ, piece, start, end);
        let score = evaluate_capture_move_order(&board, m);
        self.capture_moves.push(m.with_score(score));
    }

    #[inline(always)]
    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        loop {
            match self.stage {
                Stage::HashMove => {
                    self.capture_index = 0;
                    self.stage = Stage::CaptureMoves;

                    if self.scored_hash_move != NO_MOVE {
                        return Some(self.scored_hash_move);
                    }
                },

                Stage::CaptureMoves => {
                    if !self.moves_generated {
                        self.gen_moves(board);
                        sort_by_score_desc(&mut self.capture_moves);

                        self.moves_generated = true;
                    }

                    match self.find_next_capture_move_except_hash_move() {
                        Some(m) => {
                            return Some(m);
                        },

                        None => {
                            self.stage = Stage::QuietMoves;
                            self.move_index = 0;

                            if !self.quiets_scored {
                                self.score_quiets(hh, board);
                            }

                            return self.find_next_quiet_move_except_hash_move();
                        }
                    }
                }

                Stage::QuietMoves => {
                    return self.find_next_quiet_move_except_hash_move();
                }
            }
        }
    }

    fn score_quiets(&mut self, hh: &HistoryHeuristics, board: &Board) {
        for m in self.moves.iter_mut() {
            let score = if *m == self.primary_killer {
                PRIMARY_KILLER_SCORE
            } else if *m == self.secondary_killer {
                SECONDARY_KILLER_SCORE
            } else {
                evaluate_move_order(self.phase, hh, board, self.active_player, *m)
            };

            *m = m.with_score(score);
        }

        self.quiets_scored = true;
    }

    #[inline]
    fn find_next_capture_move_except_hash_move(&mut self) -> Option<Move> {
        while self.capture_index < self.capture_moves.len() {
            let m = unsafe { *self.capture_moves.get_unchecked(self.capture_index) };
            self.capture_index += 1;
            if !m.is_same_move(self.scored_hash_move) {
                return Some(m);
            }
        }

        None
    }

    #[inline]
    fn find_next_quiet_move_except_hash_move(&mut self) -> Option<Move> {
        while self.move_index < self.moves.len() {
            if !self.quiets_sorted {
                self.quiets_sorted = sort_partial_by_score_desc(self.move_index, &mut self.moves);
            }

            let m = unsafe { *self.moves.get_unchecked(self.move_index) };
            self.move_index += 1;
            if !m.is_same_move(self.scored_hash_move) {
                return Some(m);
            }
        }

        None
    }

    pub fn next_legal_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        if !self.moves_generated {
            self.gen_moves(board);
            self.score_quiets(hh, board);
            sort_by_score_desc(&mut self.moves);
            self.moves.append(&mut self.capture_moves);

            let active_player = board.active_player();
            self.moves.retain(|&m| board.is_legal_move(active_player, m));

            sort_by_score_desc(&mut self.moves);
            self.moves_generated = true;
            self.quiets_sorted = true;
        }

        if self.move_index < self.moves.len() {
            self.move_index += 1;
            Some(unsafe { *self.moves.get_unchecked(self.move_index - 1) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn next_capture_move(&mut self, board: &mut Board) -> Option<Move> {
        if !self.moves_generated {
            self.gen_capture_moves(board);
            sort_by_score_desc(&mut self.capture_moves);
            self.moves_generated = true;
        }

        if self.capture_index < self.capture_moves.len() {
            self.capture_index += 1;
            Some(unsafe { *self.capture_moves.get_unchecked(self.capture_index - 1) })
        } else {
            None
        }
    }

    fn gen_moves(&mut self, board: &Board) {
        self.phase = board.calc_phase_value();

        let active_player = self.active_player;
        let opponent_bb = board.get_all_piece_bitboard(-active_player);
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        for pos in BitBoard(board.get_bitboard(N * active_player)) {
            let attacks = get_knight_attacks(pos as i32);
            self.gen_piece_moves(board, N, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in BitBoard(board.get_bitboard(B * active_player)) {
            let attacks = get_bishop_attacks(occupied, pos as i32);
            self.gen_piece_moves(board, B, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in BitBoard(board.get_bitboard(R * active_player)) {
            let attacks = get_rook_attacks(occupied, pos as i32);
            self.gen_piece_moves(board, R, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in BitBoard(board.get_bitboard(Q * active_player)) {
            let attacks = get_queen_attacks(occupied, pos as i32);
            self.gen_piece_moves(board, Q, pos as i32, attacks, opponent_bb, empty_bb);
        }

        if active_player == WHITE {
            self.gen_white_king_moves(board, board.king_pos(WHITE), opponent_bb, empty_bb);

            let pawns = board.get_bitboard(P);
            self.gen_white_attack_pawn_moves(board, pawns, opponent_bb);
            self.gen_white_straight_pawn_moves(pawns, empty_bb);
            self.gen_white_en_passant_moves(board, pawns);

        } else {
            self.gen_black_king_moves(board, board.king_pos(BLACK), opponent_bb, empty_bb);

            let pawns = board.get_bitboard(-P);
            self.gen_black_straight_pawn_moves(pawns, empty_bb);
            self.gen_black_attack_pawn_moves(board, pawns, opponent_bb);
            self.gen_black_en_passant_moves(board, pawns);
        }
    }

    fn gen_capture_moves(&mut self, board: &Board) {
        let active_player = self.active_player;

        let opponent_bb = board.get_all_piece_bitboard(-active_player);
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);

        if active_player == WHITE {
            self.gen_white_attack_pawn_moves(board, board.get_bitboard(P), opponent_bb);

        } else {
            self.gen_black_attack_pawn_moves(board, board.get_bitboard(-P), opponent_bb);

        }

        for pos in BitBoard(board.get_bitboard(N * active_player)) {
            let attacks = get_knight_attacks(pos as i32);
            self.add_capture_moves(board, MoveType::Capture, N, pos as i32, attacks & opponent_bb);
        }

        for pos in BitBoard(board.get_bitboard(B * active_player)) {
            let attacks = get_bishop_attacks(occupied, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, B, pos as i32, attacks & opponent_bb);
        }

        for pos in BitBoard(board.get_bitboard(R * active_player)) {
            let attacks = get_rook_attacks(occupied, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, R, pos as i32, attacks & opponent_bb);
        }

        for pos in BitBoard(board.get_bitboard(Q * active_player)) {
            let attacks = get_queen_attacks(occupied, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, Q, pos as i32, attacks & opponent_bb);
        }

        let king_pos = board.king_pos(active_player);
        let king_targets = get_king_attacks(king_pos);
        self.add_capture_moves(board, MoveType::KingCapture, K, king_pos, king_targets & opponent_bb);
    }

    fn gen_white_straight_pawn_moves(&mut self, pawns: u64, empty_bb: u64) {
        // Single move
        let mut target_bb = (pawns >> 8) & empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnQuiet, target_bb, 8);

        // Double move
        target_bb &= unsafe{ *PAWN_DOUBLE_MOVE_LINES.get_unchecked(WHITE as usize + 1) };
        target_bb >>= 8;

        target_bb &= empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnDoubleQuiet, target_bb, 16);
    }

    fn gen_white_attack_pawn_moves(&mut self, board: &Board, pawns: u64, opponent_bb: u64) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks >>= 9;

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, left_attacks, 9);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks >>= 7;

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, right_attacks, 7);
    }

    fn gen_white_en_passant_moves(&mut self, board: &Board, pawns: u64) {
        let en_passant = board.get_enpassant_state();
        if en_passant == 0 {
            return;
        }

        let end = 16 + en_passant.trailing_zeros();
        if en_passant != 0b10000000 {
            let start = end + 9;
            if (pawns & (1 << start)) != 0 {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }

        if en_passant != 0b00000001 {
            let start = end + 7;
            if (pawns & (1 << start)) != 0 {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }
    }

    fn gen_black_straight_pawn_moves(&mut self, pawns: u64, empty_bb: u64) {
        // Single move
        let mut target_bb = (pawns << 8) & empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnQuiet, target_bb, -8);

        // Double move
        target_bb &= unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(((BLACK as i32) + 1) as usize) };
        target_bb <<= 8;

        target_bb &= empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnDoubleQuiet, target_bb, -16);
    }

    fn gen_black_attack_pawn_moves(&mut self, board: &Board, pawns: u64, opponent_bb: u64) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks <<= 7;

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, left_attacks, -7);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks <<= 9;

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, right_attacks, -9);
    }

    fn gen_black_en_passant_moves(&mut self, board: &Board, pawns: u64) {
        let en_passant = board.get_enpassant_state() >> 8;
        if en_passant == 0 {
            return;
        }

        let end = 40 + en_passant.trailing_zeros();
        if en_passant != 0b00000001 {
            let start = end - 9;
            if (pawns & (1 << start)) != 0 {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }

        if en_passant != 0b10000000 {
            let start = end - 7;
            if (pawns & (1 << start)) != 0 {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }
    }

    fn add_pawn_quiet_moves(&mut self, typ: MoveType, target_bb: u64, direction: i32) {
        for end in BitBoard(target_bb) {
            let start = end as i32 + direction;

            if end <= 7 || end >= 56 {
                // Promotion
                self.add_move(MoveType::PawnSpecial, Q, start, end as i32);
                self.add_move(MoveType::PawnSpecial, N, start, end as i32);
                self.add_move(MoveType::PawnSpecial, R, start, end as i32);
                self.add_move(MoveType::PawnSpecial, B, start, end as i32);
            } else {
                // Normal move
                self.add_move(typ, P, start, end as i32);
            }
        }
    }

    fn add_pawn_capture_moves(&mut self, board: &Board, typ: MoveType, target_bb: u64, direction: i32) {
        for end in BitBoard(target_bb) {
            let start = end as i32 + direction;

            if end <= 7 || end >= 56 {
                // Promotion
                self.add_capture_move(board, MoveType::PawnSpecial, Q, start, end as i32);
                self.add_capture_move(board, MoveType::PawnSpecial, N, start, end as i32);
                self.add_capture_move(board, MoveType::PawnSpecial, R, start, end as i32);
                self.add_capture_move(board, MoveType::PawnSpecial, B, start, end as i32);
            } else {
                // Normal move
                self.add_capture_move(board, typ, P, start, end as i32);
            }
        }
    }

    fn gen_white_king_moves(&mut self, board: &Board, pos: i32, opponent_bb: u64, empty_bb: u64) {
        let king_targets = get_king_attacks(pos);

        // Captures
        self.add_capture_moves(board, MoveType::KingCapture, K, pos, king_targets & opponent_bb);

        // Normal moves
        self.add_moves(MoveType::KingQuiet, K, pos, king_targets & empty_bb);

        // // Castling moves
        if pos != WhiteBoardPos::KingStart as i32 {
            return;
        }

        if board.can_castle(Castling::WhiteKingSide) && is_kingside_castling_valid_for_white(board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, pos + 2);
        }

        if board.can_castle(Castling::WhiteQueenSide) && is_queenside_castling_valid_for_white(board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, pos - 2);
        }
    }

    fn gen_black_king_moves(&mut self, board: &Board, pos: i32, opponent_bb: u64, empty_bb: u64) {
        let king_targets = get_king_attacks(pos);

        // Captures
        self.add_capture_moves(board, MoveType::KingCapture, K, pos, king_targets & opponent_bb);

        // Normal moves
        self.add_moves(MoveType::KingQuiet, K, pos, king_targets & empty_bb);

        // Castling moves
        if pos != BlackBoardPos::KingStart as i32 {
            return;
        }

        if board.can_castle(Castling::BlackKingSide) && is_kingside_castling_valid_for_black(board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, pos + 2);
        }

        if board.can_castle(Castling::BlackQueenSide) && is_queenside_castling_valid_for_black(board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, pos - 2);
        }
    }

    fn gen_piece_moves(&mut self, board: &Board, piece: i8, pos: i32, targets: u64, opponent_bb: u64, empty_bb: u64) {
        self.add_capture_moves(board, MoveType::Capture, piece, pos, targets & opponent_bb);
        self.add_moves(MoveType::Quiet, piece, pos, targets & empty_bb);
    }
}


pub fn is_likely_valid_move(board: &Board, active_player: Color, m: Move) -> bool {
    let previous_piece = board.get_item(m.start());

    if previous_piece.signum() != active_player {
        return false;
    }

    let removed_piece = board.get_item(m.end());

    if removed_piece == K || removed_piece == -K {
        return false;
    }

    if removed_piece != EMPTY && removed_piece.signum() == active_player {
        return false;
    }

    match previous_piece.abs() {
        P => {
            if m.is_en_passant() {
                return removed_piece == EMPTY && board.can_enpassant(active_player, m.end() as u8);
            }

            let direction = (m.start() - m.end()).abs();
            if (direction == 7) || (direction == 9) {
                // Invalid capture?
                removed_piece != EMPTY
            } else {
                // Invalid quiet move?
                removed_piece == EMPTY
            }
        }

        N => {
            let attacks = get_knight_attacks(m.start());
            (attacks & (1 << m.end())) != 0
        }

        B => {
            let opponent_bb = board.get_all_piece_bitboard(-active_player);
            let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
            let attacks = get_bishop_attacks(occupied, m.start());
            let empty = !occupied;
            ((attacks & (empty | opponent_bb)) & (1 << m.end())) != 0
        }

        R => {
            let opponent_bb = board.get_all_piece_bitboard(-active_player);
            let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
            let attacks = get_rook_attacks(occupied, m.start());
            let empty = !occupied;
            ((attacks & (empty | opponent_bb)) & (1 << m.end())) != 0
        }

        Q => {
            let opponent_bb = board.get_all_piece_bitboard(-active_player);
            let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
            let attacks = get_queen_attacks(occupied, m.start());
            let empty = !occupied;
            ((attacks & (empty | opponent_bb)) & (1 << m.end())) != 0
        }

        _ => {
            true
        }
    }
}

fn is_kingside_castling_valid_for_white(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & WHITE_KING_SIDE_CASTLING_BIT_PATTERN) == WHITE_KING_SIDE_CASTLING_BIT_PATTERN
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32)
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 + 1)
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 + 2)
}

fn is_queenside_castling_valid_for_white(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN) == WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32)
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 - 1)
        && !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 - 2)
}

fn is_kingside_castling_valid_for_black(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & BLACK_KING_SIDE_CASTLING_BIT_PATTERN) == BLACK_KING_SIDE_CASTLING_BIT_PATTERN
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32)
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 + 1)
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 + 2)
}

fn is_queenside_castling_valid_for_black(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN) == BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32)
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 - 1)
        && !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 - 2)
}

// Move evaluation heuristic for initial move ordering (high values are better for the active player)
pub fn evaluate_move_order(phase: i32, hh: &HistoryHeuristics, board: &Board, active_player: Color, m: Move) -> i32 {
    match m.typ() {
        MoveType::PawnSpecial => {
            // Promotion
            if m.piece_id() == Q {
                400
            } else if m.piece_id() == N {
                0

            } else {
                -5000
            }
        }

        _ => {
            let end = m.end();

            let history_score = hh.get_history_score(active_player, m);
            if history_score == 0 {
                NEGATIVE_HISTORY_SCORE

            } else if history_score == -1 {
                // No history score -> use difference between piece square scores
                let original_piece = m.piece_id() * active_player;

                let start_packed_score = board.pst.get_packed_score(original_piece, m.start() as usize);
                let end_packed_score = board.pst.get_packed_score(original_piece, end as usize);

                let mg_diff = (unpack_score(end_packed_score) - unpack_score(start_packed_score)) as i32;
                let eg_diff = (unpack_eg_score(end_packed_score) - unpack_eg_score(start_packed_score)) as i32;

                let diff = interpolate_score(phase, mg_diff, eg_diff) * active_player as i32;

                -4096 + diff
            } else {

                -3600 + history_score
            }

        }
    }
}

// Evaluate score for capture move ordering
fn evaluate_capture_move_order(board: &Board, m: Move) -> i32 {
    match m.typ() {
        MoveType::PawnSpecial => {
            if m.is_promotion() {
                if m.piece_id() == Q {
                    1000
                } else if m.piece_id() == N {
                    0
                } else {
                    NEGATIVE_HISTORY_SCORE - 1
                }
            } else {
                // En Passant
                get_capture_order_score(P as i32, P as i32)
            }
        }

        _ => {
            let captured_piece = board.get_item(m.end());
            let original_piece_id = m.piece_id();
            let captured_piece_id = captured_piece.abs();

            get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
        }
    }
}

fn sort_by_score_desc(moves: &mut Vec<Move>) {
    // Basic insertion sort
    for i in 1..moves.len() {
        let x = unsafe { *moves.get_unchecked(i) };
        let x_score = x.score();
        let mut j = i as i32 - 1;
        while j >= 0 {
            let y = unsafe { *moves.get_unchecked(j as usize) };
            if y.score() >= x_score {
                break;
            }

            unsafe { *moves.get_unchecked_mut(j as usize + 1) = y };
            j -= 1;
        }
        unsafe { *moves.get_unchecked_mut((j + 1) as usize) = x };
    }
}


fn sort_partial_by_score_desc(movenum: usize, moves: &mut Vec<Move>) -> bool {
    let mut max_score = unsafe { (*moves.get_unchecked(movenum)).score() };
    let mut max_index = movenum;

    let mut is_sorted = true;

    for i in (movenum + 1)..moves.len() {
        let x = unsafe { *moves.get_unchecked(i) };
        let x_score = x.score();

        if x_score <= max_score {
            continue;
        }

        is_sorted = false;

        max_score = x_score;
        max_index = i;
    }

    if max_index != movenum {
        moves.swap(movenum, max_index);
    }

    is_sorted
}


#[inline]
fn get_capture_order_score(attacker_id: i32, victim_id: i32) -> i32 {
    unsafe { *CAPTURE_ORDER_SCORES.get_unchecked(((attacker_id - 1) * 6 + (victim_id - 1)) as usize) }
}

const fn calc_capture_order_scores() -> [i32; CAPTURE_ORDER_SIZE] {
    let mut scores: [i32; CAPTURE_ORDER_SIZE] = [0; CAPTURE_ORDER_SIZE];
    let mut score: i32 = 0;

    let mut victim = 0;
    while victim <= 5 {

        let mut attacker = 5;
        while attacker >= 0 {
            scores[(victim + attacker * 6) as usize] = score * 16;
            score += 1;

            attacker -= 1;
        }

        victim += 1;
    }

    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const ONLY_KINGS: [i8; 64] = [
        0, 0, 0, 0, 0, 0, -K, 0, // 0 - 7
        0, 0, 0, 0, 0, 0, 0, 0, // 8 - 15
        0, 0, 0, 0, 0, 0, 0, 0, // 16 - 23
        0, 0, 0, 0, 0, 0, 0, 0, // 24 - 31
        0, 0, 0, 0, 0, 0, 0, 0, // 32 - 39
        0, 0, 0, 0, 0, 0, 0, 0, // 40 - 47
        0, 0, 0, 0, 0, 0, 0, 0, // 48 - 55
        0, 0, 0, 0, 0, 0, K, 0, // 56 - 63
    ];

    #[test]
    pub fn white_pawn_moves_blocked() {
        let mut board: Board = board_with_one_piece(WHITE, P, 52);
        board.add_piece(WHITE, P, 44);

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(0, moves.len());
    }

    #[test]
    pub fn white_queen_moves() {
        let mut board: Board = board_with_one_piece(WHITE, Q, 28);

        let moves = generate_moves_for_pos(&mut board, WHITE, 28);

        assert_eq!(27, moves.len());
    }

    #[test]
    pub fn exclude_illegal_moves() {
        let mut board: Board = board_with_one_piece(WHITE, Q, 52);
        board.perform_move(Move::new(MoveType::KingQuiet, K, board.king_pos(WHITE), 53));
        board.add_piece(BLACK, R, 51);

        board.perform_null_move(); // so WHITE is the active player

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(
            1,
            moves.len(),
            "There must be only one legal move for the white queen"
        );
    }

    #[test]
    fn partial_sorting() {
        let m1 = Move::new(MoveType::Quiet, Q, 1, 2).with_score(1);
        let m2 = Move::new(MoveType::Quiet, Q, 1, 3).with_score(2);
        let m3 = Move::new(MoveType::Quiet, Q, 1, 5).with_score(3);
        let m4 = Move::new(MoveType::Quiet, Q, 1, 4).with_score(4);

        let mut moves = vec!(m1, m2, m3, m4);

        sort_partial_by_score_desc(0, &mut moves);
        assert_eq!(moves[0], m4);

        sort_partial_by_score_desc(1, &mut moves);
        assert_eq!(moves[1], m3);

        sort_partial_by_score_desc(2, &mut moves);
        assert_eq!(moves[2], m2);

        sort_partial_by_score_desc(3, &mut moves);
        assert_eq!(moves[3], m1);
    }

    fn board_with_one_piece(color: Color, piece_id: i8, pos: i32) -> Board {
        let mut items = ONLY_KINGS;
        items[pos as usize] = piece_id * color;
        Board::new(&items, color, 0, None, 0, 1)
    }

    fn generate_moves_for_pos(board: &mut Board, color: Color, pos: i32) -> Vec<Move> {
        let mut hh = HistoryHeuristics::new();
        let mut ml = MoveList::new();
        ml.init(color, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut moves = Vec::new();

        loop {
            let m = ml.next_move(&mut hh, board);

            if let Some(m) = m {
                moves.push(m);

            } else {
                break;
            }
        };

        moves.into_iter()
            .filter(|&m| m.start() == pos)
            .filter(|&m| board.is_legal_move(color, m))
            .collect()
    }


}
