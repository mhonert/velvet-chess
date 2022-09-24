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

use crate::bitboard::{get_king_attacks, get_knight_attacks, BitBoard, PAWN_DOUBLE_MOVE_LINES};
use crate::board::Board;
use crate::colors::{Color, BLACK, WHITE};
use crate::history_heuristics::{HistoryHeuristics, MIN_HISTORY_SCORE};
use crate::moves::{Move, MoveType, NO_MOVE};
use crate::pieces::{B, EMPTY, K, N, P, Q, R};
use crate::transposition_table::MAX_DEPTH;
use std::cmp::Reverse;

const CAPTURE_ORDER_SIZE: usize = 6 + 6 * 7 + 1;

const PRIMARY_KILLER_SCORE: i32 = -2200;
const SECONDARY_KILLER_SCORE: i32 = -2250;
const COUNTER_MOVE_SCORE: i32 = -2275;

pub const QUIET_BASE_SCORE: i32 = -3600;
pub const NEGATIVE_HISTORY_SCORE: i32 = QUIET_BASE_SCORE + MIN_HISTORY_SCORE;

const CAPTURE_ORDER_SCORES: [i32; CAPTURE_ORDER_SIZE] = calc_capture_order_scores();

#[derive(Clone)]
pub struct MoveGenerator {
    entries: Vec<MoveList>,
    ply: usize,
}

impl MoveGenerator {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(MAX_DEPTH + 2);
        for _ in 0..MAX_DEPTH + 2 {
            entries.push(MoveList::new());
        }

        MoveGenerator { entries, ply: 0 }
    }

    pub fn enter_ply(
        &mut self, active_player: Color, scored_hash_move: Move, primary_killer: Move, secondary_killer: Move,
        counter_move: Move, prev_own_move: Move, opp_move: Move,
    ) {
        self.ply += 1;
        self.entries[self.ply].init(
            active_player,
            scored_hash_move,
            primary_killer,
            secondary_killer,
            counter_move,
            prev_own_move,
            opp_move,
        );
    }

    pub fn leave_ply(&mut self) {
        self.ply -= 1;
    }

    pub fn next_root_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_root_move(hh, board)
    }

    pub fn reset_root_moves(&mut self) {
        self.entries[self.ply].reset_root_moves();
    }

    pub fn reorder_root_moves(&mut self, best_move: Move, sort_other_moves: bool) {
        self.entries[self.ply].reorder_root_moves(best_move, sort_other_moves);
    }

    #[inline(always)]
    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_move(hh, board)
    }

    #[inline(always)]
    pub fn next_capture_move(&mut self, board: &mut Board) -> Option<Move> {
        self.entries[self.ply].next_capture_move(board)
    }

    pub fn update_root_move(&mut self, m: Move) {
        self.entries[self.ply].update_root_move(m);
    }

    #[inline(always)]
    pub fn skip_bad_capture(
        &mut self, m: Move, captured_piece_id: i8, occupied_bb: BitBoard, board: &mut Board,
    ) -> bool {
        self.entries[self.ply].skip_bad_capture(m, captured_piece_id, occupied_bb, board)
    }

    pub fn sanitize_move(&mut self, board: &Board, active_player: Color, untyped_move: Move) -> Move {
        self.enter_ply(active_player, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);
        let m = self.entries[self.ply].sanitize_move(board, active_player, untyped_move);
        self.leave_ply();

        m
    }
}

#[derive(Clone)]
enum Stage {
    HashMove,
    GenerateMoves,
    CaptureMoves,
    PrimaryKillerMove,
    SecondaryKillerMove,
    CounterMove,
    PostponedBadCaptureMoves,
    SortQuietMoves,
    QuietMoves,
}

#[derive(Clone)]
pub struct MoveList {
    scored_hash_move: Move,
    primary_killer: Move,
    secondary_killer: Move,
    counter_move: Move,
    prev_own_move: Move,
    opp_move: Move,
    moves: Vec<Move>, // contains all moves on root level, but only quiet moves in all other cases
    capture_moves: Vec<Move>, // not used on root level
    bad_capture_moves: Vec<Move>, // not used on root level
    stage: Stage,
    root_move_index: usize,
    moves_generated: bool,
    active_player: Color,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            scored_hash_move: NO_MOVE,
            primary_killer: NO_MOVE,
            secondary_killer: NO_MOVE,
            counter_move: NO_MOVE,
            prev_own_move: NO_MOVE,
            opp_move: NO_MOVE,
            moves: Vec::with_capacity(64),
            capture_moves: Vec::with_capacity(16),
            bad_capture_moves: Vec::with_capacity(16),
            stage: Stage::HashMove,
            root_move_index: 0,
            moves_generated: false,
            active_player: WHITE,
        }
    }

    pub fn init(
        &mut self, active_player: Color, scored_hash_move: Move, primary_killer: Move, secondary_killer: Move,
        counter_move: Move, prev_own_move: Move, opp_move: Move,
    ) {
        self.scored_hash_move = scored_hash_move;
        self.primary_killer = primary_killer;
        self.secondary_killer = secondary_killer;
        self.counter_move = counter_move;
        self.prev_own_move = prev_own_move;
        self.opp_move = opp_move;

        self.moves.clear();
        self.capture_moves.clear();
        self.bad_capture_moves.clear();
        self.moves_generated = false;
        self.active_player = active_player;
        self.root_move_index = 0;
        self.stage = Stage::HashMove;
    }

    pub fn reset_root_moves(&mut self) {
        self.stage = Stage::HashMove;
        self.root_move_index = 0;
    }

    pub fn reorder_root_moves(&mut self, best_move: Move, sort_other_moves: bool) {
        if let Some(i) = self.moves.iter().position(|m| m.is_same_move(best_move)) {
            self.moves.remove(i);
            if sort_other_moves {
                self.moves.sort_by_key(|m| Reverse(m.score()));
            }
            self.moves.insert(0, best_move);
        }
    }

    #[inline]
    pub fn add_moves(&mut self, typ: MoveType, piece: i8, pos: i32, target_bb: BitBoard) {
        for end in target_bb {
            self.add_move(typ, piece, pos, end as i32);
        }
    }

    #[inline]
    pub fn add_move(&mut self, typ: MoveType, piece: i8, start: i32, end: i32) {
        let m = Move::new(typ, piece, start, end);
        if !m.is_same_move(self.scored_hash_move) {
            self.moves.push(m);
        }
    }

    pub fn update_root_move(&mut self, scored_move: Move) {
        self.moves[self.root_move_index - 1] = scored_move;
    }

    #[inline]
    pub fn add_capture_moves(&mut self, board: &Board, typ: MoveType, piece: i8, pos: i32, target_bb: BitBoard) {
        for end in target_bb {
            self.add_capture_move(board, typ, piece, pos, end as i32);
        }
    }

    #[inline]
    pub fn add_capture_move(&mut self, board: &Board, typ: MoveType, piece: i8, start: i32, end: i32) {
        let m = Move::new(typ, piece, start, end);
        if !m.is_same_move(self.scored_hash_move) {
            let score = evaluate_capture_move_order(board, m);
            self.capture_moves.push(m.with_score(score));
        }
    }

    #[inline(always)]
    pub fn next_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        loop {
            match self.stage {
                Stage::HashMove => {
                    self.stage = Stage::GenerateMoves;

                    if self.scored_hash_move != NO_MOVE {
                        return Some(self.scored_hash_move);
                    }
                }

                Stage::GenerateMoves => {
                    self.stage = Stage::CaptureMoves;
                    self.gen_moves(board);
                    self.capture_moves.sort_unstable_by_key(Move::score);
                    self.moves_generated = true;
                }

                Stage::CaptureMoves => match self.capture_moves.pop() {
                    Some(m) => return Some(m),
                    None => self.stage = Stage::PrimaryKillerMove,
                },

                Stage::PrimaryKillerMove => {
                    self.stage = Stage::SecondaryKillerMove;

                    if self.primary_killer != NO_MOVE && remove_move(&mut self.moves, self.primary_killer) != NO_MOVE {
                        return Some(self.primary_killer.with_score(PRIMARY_KILLER_SCORE));
                    }
                }

                Stage::SecondaryKillerMove => {
                    self.stage = Stage::CounterMove;

                    if self.secondary_killer != NO_MOVE
                        && remove_move(&mut self.moves, self.secondary_killer) != NO_MOVE
                    {
                        return Some(self.secondary_killer.with_score(SECONDARY_KILLER_SCORE));
                    }
                }

                Stage::CounterMove => {
                    self.stage = Stage::PostponedBadCaptureMoves;

                    if self.counter_move != NO_MOVE && remove_move(&mut self.moves, self.counter_move) != NO_MOVE {
                        return Some(self.counter_move.with_score(COUNTER_MOVE_SCORE));
                    }
                }

                Stage::PostponedBadCaptureMoves => {
                    if self.bad_capture_moves.is_empty() {
                        self.stage = Stage::SortQuietMoves;
                    } else {
                        return Some(self.bad_capture_moves.swap_remove(0));
                    }
                }

                Stage::SortQuietMoves => {
                    self.stage = Stage::QuietMoves;
                    self.score_quiets(hh);
                    self.moves.sort_unstable_by_key(Move::score)
                }

                Stage::QuietMoves => {
                    return self.moves.pop();
                }
            }
        }
    }

    fn score_quiets(&mut self, hh: &HistoryHeuristics) {
        for m in self.moves.iter_mut() {
            *m = m.with_score(evaluate_move_order(hh, self.active_player, self.prev_own_move, self.opp_move, *m));
        }
    }

    pub fn next_root_move(&mut self, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        if !self.moves_generated {
            self.gen_moves(board);
            self.score_quiets(hh);
            self.moves.append(&mut self.capture_moves);

            let active_player = board.active_player();
            self.moves.retain(|&m| board.is_legal_move(active_player, m));

            self.moves.sort_by_key(|m| Reverse(m.score()));
            self.moves_generated = true;
        }

        if self.root_move_index >= self.moves.len() {
            return None;
        }

        self.root_move_index += 1;
        Some(unsafe { *self.moves.get_unchecked(self.root_move_index - 1) })
    }

    #[inline(always)]
    pub fn next_capture_move(&mut self, board: &mut Board) -> Option<Move> {
        if !self.moves_generated {
            self.moves_generated = true;
            self.gen_capture_moves(board);
            self.capture_moves.sort_unstable_by_key(Move::score)
        }

        self.capture_moves.pop()
    }

    fn gen_moves(&mut self, board: &Board) {
        let active_player = self.active_player;
        let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        if active_player.is_white() {
            let pawns = board.get_bitboard(P);
            self.gen_white_attack_pawn_moves(board, pawns, opponent_bb);
            self.gen_white_straight_pawn_moves(pawns, empty_bb);
            self.gen_white_en_passant_moves(board, pawns);
        } else {
            let pawns = board.get_bitboard(-P);
            self.gen_black_straight_pawn_moves(pawns, empty_bb);
            self.gen_black_attack_pawn_moves(board, pawns, opponent_bb);
            self.gen_black_en_passant_moves(board, pawns);
        }

        for pos in board.get_bitboard(active_player.piece(B)) {
            let attacks = board.get_bishop_attacks(empty_bb, pos as i32);
            self.gen_piece_moves(board, B, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(N)) {
            let attacks = get_knight_attacks(pos as i32);
            self.gen_piece_moves(board, N, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(R)) {
            let attacks = board.get_rook_attacks(empty_bb, pos as i32);
            self.gen_piece_moves(board, R, pos as i32, attacks, opponent_bb, empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(Q)) {
            let attacks = board.get_queen_attacks(empty_bb, pos as i32);
            self.gen_piece_moves(board, Q, pos as i32, attacks, opponent_bb, empty_bb);
        }

        self.gen_king_moves(active_player, board, opponent_bb, empty_bb);
    }

    fn gen_capture_moves(&mut self, board: &Board) {
        let active_player = self.active_player;

        let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        if active_player.is_white() {
            self.gen_white_attack_pawn_moves(board, board.get_bitboard(P), opponent_bb);
        } else {
            self.gen_black_attack_pawn_moves(board, board.get_bitboard(-P), opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(N)) {
            let attacks = get_knight_attacks(pos as i32);
            self.add_capture_moves(board, MoveType::Capture, N, pos as i32, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(B)) {
            let attacks = board.get_bishop_attacks(empty_bb, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, B, pos as i32, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(R)) {
            let attacks = board.get_rook_attacks(empty_bb, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, R, pos as i32, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(Q)) {
            let attacks = board.get_queen_attacks(empty_bb, pos as i32);
            self.add_capture_moves(board, MoveType::Capture, Q, pos as i32, attacks & opponent_bb);
        }

        let king_pos = board.king_pos(active_player);
        let king_targets = get_king_attacks(king_pos as i32);
        self.add_capture_moves(board, MoveType::KingCapture, K, king_pos as i32, king_targets & opponent_bb);
    }

    fn gen_white_straight_pawn_moves(&mut self, pawns: BitBoard, empty_bb: BitBoard) {
        // Single move
        let mut target_bb = (pawns >> 8) & empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnQuiet, target_bb, 8);

        // Double move
        target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(WHITE.idx()) });
        target_bb >>= BitBoard(8);

        target_bb &= empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnDoubleQuiet, target_bb, 16);
    }

    fn gen_white_attack_pawn_moves(&mut self, board: &Board, pawns: BitBoard, opponent_bb: BitBoard) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks >>= BitBoard(9);

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, left_attacks, 9);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks >>= BitBoard(7);

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, right_attacks, 7);
    }

    fn gen_white_en_passant_moves(&mut self, board: &Board, pawns: BitBoard) {
        let en_passant = board.get_enpassant_state();
        if en_passant == 0 {
            return;
        }

        let end = 16 + en_passant.trailing_zeros();
        if en_passant != 0b10000000 {
            let start = end + 9;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }

        if en_passant != 0b00000001 {
            let start = end + 7;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }
    }

    fn gen_black_straight_pawn_moves(&mut self, pawns: BitBoard, empty_bb: BitBoard) {
        // Single move
        let mut target_bb = (pawns << 8) & empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnQuiet, target_bb, -8);

        // Double move
        target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(BLACK.idx()) });
        target_bb <<= BitBoard(8);

        target_bb &= empty_bb;
        self.add_pawn_quiet_moves(MoveType::PawnDoubleQuiet, target_bb, -16);
    }

    fn gen_black_attack_pawn_moves(&mut self, board: &Board, pawns: BitBoard, opponent_bb: BitBoard) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks <<= BitBoard(7);

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, left_attacks, -7);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks <<= BitBoard(9);

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves(board, MoveType::Capture, right_attacks, -9);
    }

    fn gen_black_en_passant_moves(&mut self, board: &Board, pawns: BitBoard) {
        let en_passant = board.get_enpassant_state() >> 8;
        if en_passant == 0 {
            return;
        }

        let end = 40 + en_passant.trailing_zeros();
        if en_passant != 0b00000001 {
            let start = end - 9;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }

        if en_passant != 0b10000000 {
            let start = end - 7;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnSpecial, P, start as i32, end as i32);
            }
        }
    }

    fn add_pawn_quiet_moves(&mut self, typ: MoveType, target_bb: BitBoard, direction: i32) {
        for end in target_bb {
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

    fn add_pawn_capture_moves(&mut self, board: &Board, typ: MoveType, target_bb: BitBoard, direction: i32) {
        for end in target_bb {
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

    fn gen_king_moves(&mut self, color: Color, board: &Board, opponent_bb: BitBoard, empty_bb: BitBoard) {
        let pos = board.king_pos(color);
        let king_targets = get_king_attacks(pos as i32);

        // Captures
        self.add_capture_moves(board, MoveType::KingCapture, K, pos as i32, king_targets & opponent_bb);

        self.gen_quiet_king_moves(color, board, pos as i32, empty_bb, king_targets);
    }

    fn gen_quiet_king_moves(
        &mut self, color: Color, board: &Board, pos: i32, empty_bb: BitBoard, king_targets: BitBoard,
    ) {
        // Normal moves
        self.add_moves(MoveType::KingQuiet, K, pos, king_targets & empty_bb);

        // // Castling moves
        if !board.castling_rules.is_king_start(color, pos) {
            return;
        }

        if board.can_castle_king_side(color) && board.castling_rules.is_ks_castling_valid(color, board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, board.castling_rules.ks_rook_start(color));
        }

        if board.can_castle_queen_side(color) && board.castling_rules.is_qs_castling_valid(color, board, empty_bb) {
            self.add_move(MoveType::Castling, K, pos, board.castling_rules.qs_rook_start(color));
        }
    }

    fn gen_piece_moves(
        &mut self, board: &Board, piece: i8, pos: i32, targets: BitBoard, opponent_bb: BitBoard, empty_bb: BitBoard,
    ) {
        self.add_capture_moves(board, MoveType::Capture, piece, pos, targets & opponent_bb);
        self.add_moves(MoveType::Quiet, piece, pos, targets & empty_bb);
    }

    pub fn sanitize_move(&mut self, board: &Board, active_player: Color, untyped_move: Move) -> Move {
        let start = untyped_move.start();
        let end = untyped_move.end();
        if untyped_move.piece_id() == 0 {
            return NO_MOVE;
        }

        let piece = board.get_item(start);
        if piece == EMPTY {
            return NO_MOVE;
        }

        if active_player.is_opp_piece(piece) {
            return NO_MOVE;
        }

        let captured_piece = board.get_item(end);
        let piece_id = piece.abs();
        if captured_piece != EMPTY && piece_id != K && active_player.is_own_piece(captured_piece) {
            return NO_MOVE;
        }

        let target_piece_id = untyped_move.piece_id();

        let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        let start_bb = BitBoard(1u64 << start);
        let end_bb = BitBoard(1u64 << end);

        match piece_id {
            P => {
                if captured_piece == EMPTY {
                    if active_player.is_white() {
                        self.gen_white_straight_pawn_moves(start_bb, empty_bb);
                        self.gen_white_en_passant_moves(board, start_bb);
                    } else {
                        self.gen_black_straight_pawn_moves(start_bb, empty_bb);
                        self.gen_black_en_passant_moves(board, start_bb);
                    }
                } else {
                    if active_player.is_white() {
                        self.gen_white_attack_pawn_moves(board, start_bb, opponent_bb);
                    } else {
                        self.gen_black_attack_pawn_moves(board, start_bb, opponent_bb);
                    }
                }
            }

            N => {
                if target_piece_id != N {
                    return NO_MOVE;
                }
                if (get_knight_attacks(start) & end_bb).is_empty() {
                    return NO_MOVE;
                }

                return untyped_move.with_typ(if captured_piece == EMPTY {
                    MoveType::Quiet
                } else {
                    MoveType::Capture
                });
            }

            B => {
                if target_piece_id != B {
                    return NO_MOVE;
                }
                if (board.get_bishop_attacks(empty_bb, start) & end_bb).is_empty() {
                    return NO_MOVE;
                }

                return untyped_move.with_typ(if captured_piece == EMPTY {
                    MoveType::Quiet
                } else {
                    MoveType::Capture
                });
            }

            R => {
                if target_piece_id != R {
                    return NO_MOVE;
                }
                if (board.get_rook_attacks(empty_bb, start) & end_bb).is_empty() {
                    return NO_MOVE;
                }

                return untyped_move.with_typ(if captured_piece == EMPTY {
                    MoveType::Quiet
                } else {
                    MoveType::Capture
                });
            }

            Q => {
                if target_piece_id != Q {
                    return NO_MOVE;
                }
                if (board.get_queen_attacks(empty_bb, start) & end_bb).is_empty() {
                    return NO_MOVE;
                }

                return untyped_move.with_typ(if captured_piece == EMPTY {
                    MoveType::Quiet
                } else {
                    MoveType::Capture
                });
            }

            K => {
                let king_targets = get_king_attacks(start) & end_bb;
                if captured_piece == EMPTY || active_player.is_own_piece(captured_piece) {
                    self.gen_quiet_king_moves(active_player, board, start, empty_bb, king_targets);
                } else {
                    if target_piece_id != K {
                        return NO_MOVE;
                    }
                    if (king_targets & opponent_bb).is_empty() {
                        return NO_MOVE;
                    }
                    return untyped_move.with_typ(MoveType::KingCapture);
                }
            }

            _ => {
                return NO_MOVE;
            }
        }

        for m in self.moves.iter() {
            if m.is_same_untyped_move(untyped_move) {
                return untyped_move.with_typ(m.typ());
            }
        }

        for m in self.capture_moves.iter() {
            if m.is_same_untyped_move(untyped_move) {
                return untyped_move.with_typ(m.typ());
            }
        }

        NO_MOVE
    }

    // If the given move is a bad capture (i.e. has a negative SEE value), the search can be skipped for now and the move will be stored in a separate "bad capture" list
    pub fn skip_bad_capture(
        &mut self, m: Move, captured_piece_id: i8, occupied_bb: BitBoard, board: &mut Board,
    ) -> bool {
        if !matches!(self.stage, Stage::CaptureMoves) {
            return false;
        }

        if !board.has_negative_see(
            board.active_player(),
            m.start(),
            m.end(),
            m.piece_id(),
            captured_piece_id,
            0,
            occupied_bb,
        ) {
            return false;
        }

        self.bad_capture_moves.push(m);

        true
    }
}

#[inline(always)]
fn remove_move(moves: &mut Vec<Move>, to_be_removed: Move) -> Move {
    if let Some(i) = moves.iter().position(|m| m.is_same_move(to_be_removed)) {
        return moves.swap_remove(i);
    }

    NO_MOVE
}

// Move evaluation heuristic for initial move ordering (high values are better for the active player)
#[inline(always)]
pub fn evaluate_move_order(
    hh: &HistoryHeuristics, active_player: Color, prev_own_m: Move, opp_m: Move, m: Move,
) -> i32 {
    let history_score = hh.get_history_score(active_player, prev_own_m, opp_m, m);
    QUIET_BASE_SCORE + history_score
}

// Evaluate score for capture move ordering
#[inline(always)]
fn evaluate_capture_move_order(board: &Board, m: Move) -> i32 {
    let captured_piece = board.get_item(m.end());
    let original_piece_id = m.piece_id();
    let captured_piece_id = captured_piece.abs();

    get_capture_order_score(original_piece_id as i32, captured_piece_id as i32)
}

#[inline]
pub fn is_killer(m: Move) -> bool {
    let score = m.score();
    score == PRIMARY_KILLER_SCORE || score == SECONDARY_KILLER_SCORE
}

#[inline(always)]
fn get_capture_order_score(attacker_id: i32, victim_id: i32) -> i32 {
    unsafe { *CAPTURE_ORDER_SCORES.get_unchecked((attacker_id * 7 + victim_id) as usize) }
}

const fn calc_capture_order_scores() -> [i32; CAPTURE_ORDER_SIZE] {
    let mut scores: [i32; CAPTURE_ORDER_SIZE] = [0; CAPTURE_ORDER_SIZE];
    let mut score: i32 = 0;

    let mut victim = 1;
    while victim <= 6 {
        let mut attacker = 6;
        while attacker >= 0 {
            scores[(attacker * 7 + victim) as usize] = score;
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
    use crate::board::castling::{CastlingRules, CastlingState};

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
        let mut board = board_with_one_piece(WHITE, P, 52);
        board.add_piece(WHITE, P, 44);

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(0, moves.len());
    }

    #[test]
    pub fn white_queen_moves() {
        let mut board = board_with_one_piece(WHITE, Q, 28);

        let moves = generate_moves_for_pos(&mut board, WHITE, 28);

        assert_eq!(27, moves.len());
    }

    #[test]
    pub fn exclude_illegal_moves() {
        let mut board = board_with_one_piece(WHITE, Q, 52);
        board.perform_move(Move::new(MoveType::KingQuiet, K, board.king_pos(WHITE) as i32, 53));
        board.add_piece(BLACK, R, 51);

        board.perform_null_move(); // so WHITE is the active player

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(1, moves.len(), "There must be only one legal move for the white queen");
    }

    fn board_with_one_piece(color: Color, piece_id: i8, pos: i32) -> Board {
        let mut items = ONLY_KINGS;
        items[pos as usize] = color.piece(piece_id);
        Board::new(&items, color, CastlingState::default(), None, 0, 1, CastlingRules::default())
    }

    fn generate_moves_for_pos(board: &mut Board, color: Color, pos: i32) -> Vec<Move> {
        let hh = HistoryHeuristics::new();
        let mut ml = MoveList::new();
        ml.init(color, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

        let mut moves = Vec::new();

        loop {
            let m = ml.next_move(&hh, board);

            if let Some(m) = m {
                moves.push(m);
            } else {
                break;
            }
        }

        moves.into_iter().filter(|&m| m.start() == pos).filter(|&m| board.is_legal_move(color, m)).collect()
    }
}
