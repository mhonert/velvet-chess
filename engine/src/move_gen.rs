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

use crate::bitboard::{get_king_attacks, get_knight_attacks, BitBoard, PAWN_DOUBLE_MOVE_LINES};
use crate::board::{BlackBoardPos, Board, WhiteBoardPos};
use crate::colors::{Color, BLACK, WHITE};
use crate::history_heuristics::{HistoryHeuristics, MIN_HISTORY_SCORE, MoveHistory};
use crate::moves::{Move, MoveType, NO_MOVE};
use crate::pieces::{B, N, P, Q, R};
use std::cmp::Reverse;
use std::mem::swap;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::magics::{get_bishop_attacks, get_queen_attacks, get_rook_attacks};
use crate::random::Random;
use crate::scores::MAX_SCORE;

const PRIMARY_KILLER_SCORE: i16 = -2200;
const SECONDARY_KILLER_SCORE: i16 = -2250;
const COUNTER_MOVE_SCORE: i16 = -2275;

pub const QUIET_BASE_SCORE: i16 = -3600;
pub const NEGATIVE_HISTORY_SCORE: i16 = QUIET_BASE_SCORE + MIN_HISTORY_SCORE;

#[derive(Clone)]
enum Stage {
    HashMove,
    GenerateMoves,
    CaptureMoves,
    PrimaryKillerMove,
    SecondaryKillerMove,
    CounterMove,
    PostponedBadCaptureMoves,
    GenerateQuietMoves,
    QuietMoves,
}

#[derive(Clone)]
pub struct MoveList {
    scored_hash_move: Move,
    move_history: MoveHistory,
    moves: Vec<Move>, // contains all moves on root level, but only quiet moves in all other cases
    capture_moves: Vec<Move>, // not used on root level
    bad_capture_moves: Vec<Move>, // not used on root level
    checked_priority_moves: [Move; 4],
    stage: Stage,
    root_move_index: usize,
    moves_generated: bool,
    active_player: Color,
    rnd: Random,
}

impl Default for MoveList {
    fn default() -> Self {
        MoveList {
            scored_hash_move: NO_MOVE,
            move_history: MoveHistory::default(),
            moves: Vec::with_capacity(64),
            capture_moves: Vec::with_capacity(16),
            bad_capture_moves: Vec::with_capacity(16),
            checked_priority_moves: [NO_MOVE; 4],
            stage: Stage::HashMove,
            root_move_index: 0,
            moves_generated: false,
            active_player: WHITE,
            rnd: Random::new_with_seed(SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()),
        }
    }
}

impl MoveList {
    pub fn init(
        &mut self, active_player: Color, scored_hash_move: Move, move_history: MoveHistory
    ) {
        self.scored_hash_move = scored_hash_move;
        self.move_history = move_history;

        self.moves.clear();
        self.capture_moves.clear();
        self.bad_capture_moves.clear();
        self.checked_priority_moves.fill(NO_MOVE);
        self.moves_generated = false;
        self.active_player = active_player;
        self.root_move_index = 0;
        self.stage = Stage::HashMove;
    }

    pub fn reset_root_moves(&mut self) {
        self.stage = Stage::HashMove;
        self.root_move_index = 0;
        self.checked_priority_moves.fill(NO_MOVE);
    }

    pub fn reorder_root_moves(&mut self, best_move: Move, sort_other_moves: bool) {
        if let Some(i) = self.moves.iter().position(|&m| m == best_move) {
            self.moves.remove(i);
            if sort_other_moves {
                self.moves.sort_by_key(|m| Reverse(m.score()));
            }
            self.moves.insert(0, best_move);
        }
    }

    pub fn root_move_count(&self) -> usize {
        self.moves.len()
    }

    #[inline]
    pub fn add_moves(&mut self, hh: &HistoryHeuristics, typ: MoveType, pos: i8, target_bb: BitBoard) {
        for end in target_bb {
            self.add_move(hh, typ, pos, end as i8);
        }
    }

    #[inline]
    pub fn add_move(&mut self, hh: &HistoryHeuristics, typ: MoveType, start: i8, end: i8) {
        let m = Move::new(typ, start, end);
        let score = QUIET_BASE_SCORE + hh.score(self.active_player, self.move_history, m) + i16::from(typ.is_queen_promotion());
        self.moves.push(m.with_initial_score(score));
    }
    
    #[inline]
    pub fn add_promotion_move(&mut self, hh: &HistoryHeuristics, typ: MoveType, start: i8, end: i8, underpromotion: bool) {
        let m = Move::new(typ, start, end);
        let mut score = QUIET_BASE_SCORE + hh.score(self.active_player, self.move_history, m);
        if underpromotion {
            score -= 1;
        }
        
        self.moves.push(m.with_initial_score(score));
    }

    pub fn update_root_move(&mut self, scored_move: Move) {
        self.moves[self.root_move_index - 1] = scored_move;
    }

    #[inline]
    pub fn add_capture_moves(&mut self, board: &Board, typ: MoveType, pos: i8, target_bb: BitBoard) {
        for end in target_bb {
            self.add_capture_move(board, typ, pos, end as i8);
        }
    }

    #[inline]
    pub fn add_capture_move(&mut self, board: &Board, typ: MoveType, start: i8, end: i8) {
        let m = Move::new(typ, start, end);
        let score = evaluate_capture_move_order(board, end, typ.piece_id());
        self.capture_moves.push(m.with_initial_score(score));
    }

    #[inline]
    pub fn add_capture_promotion_move(&mut self, board: &Board, typ: MoveType, start: i8, end: i8, underpromotion: bool) {
        let m = Move::new(typ, start, end);
        let score = evaluate_capture_move_order(board, end, P) + typ.piece_id() as i16 * 128;
        if underpromotion {
            self.bad_capture_moves.push(m.with_initial_score(score));
        } else {
            self.capture_moves.push(m.with_initial_score(score));
        }
    }

    #[inline(always)]
    fn add_checked_priority_move(&mut self, m: Move) {
        for entry in self.checked_priority_moves.iter_mut() {
            if *entry == NO_MOVE {
                *entry = m;
                return;
            }
        }
    }

    #[inline(always)]
    pub fn next_move(&mut self, ply: usize, hh: &HistoryHeuristics, board: &mut Board) -> Option<Move> {
        loop {
            match self.stage {
                Stage::HashMove => {
                    self.stage = Stage::GenerateMoves;

                    if self.scored_hash_move != NO_MOVE {
                        self.add_checked_priority_move(self.scored_hash_move);
                        return Some(self.scored_hash_move);
                    }
                }

                Stage::GenerateMoves => {
                    self.stage = Stage::CaptureMoves;
                    self.gen_capture_moves::<true>(board);
                    self.capture_moves.sort_unstable_by_key(Move::score);
                    self.moves_generated = true;
                }

                Stage::CaptureMoves => match self.capture_moves.pop() {
                    Some(m) => {
                        if self.checked_priority_moves.contains(&m) {
                            continue;
                        }
                        if is_bad_capture(m, board) {
                            self.bad_capture_moves.push(m);
                            continue;
                        }
                        return Some(m);
                    }
                    None => self.stage = Stage::PrimaryKillerMove,
                },

                Stage::PrimaryKillerMove => {
                    self.stage = Stage::SecondaryKillerMove;

                    let killer = hh.get_killer_moves(ply).0;
                    if killer != NO_MOVE && !self.checked_priority_moves.contains(&killer) && is_valid_move(board, board.active_player(), killer) {
                        self.add_checked_priority_move(killer);
                        return Some(killer.with_score(PRIMARY_KILLER_SCORE));
                    }
                }

                Stage::SecondaryKillerMove => {
                    self.stage = Stage::CounterMove;

                    let killer = hh.get_killer_moves(ply).1;
                    if killer != NO_MOVE && !self.checked_priority_moves.contains(&killer) && is_valid_move(board, board.active_player(), killer) {
                        self.add_checked_priority_move(killer);
                        return Some(killer.with_score(SECONDARY_KILLER_SCORE));
                    }
                }

                Stage::CounterMove => {
                    self.stage = Stage::PostponedBadCaptureMoves;
                    let counter = hh.get_counter_move(self.move_history.last_opp);

                    if counter != NO_MOVE && !self.checked_priority_moves.contains(&counter) && is_valid_move(board, board.active_player(), counter) {
                        self.add_checked_priority_move(counter);
                        return Some(counter.with_score(COUNTER_MOVE_SCORE));
                    }
                }

                Stage::PostponedBadCaptureMoves => {
                    if self.bad_capture_moves.is_empty() {
                        self.stage = Stage::GenerateQuietMoves;
                    } else {
                        let m = self.bad_capture_moves.swap_remove(0);
                        if !self.checked_priority_moves.contains(&m) {
                            return Some(m);
                        }
                    }
                }

                Stage::GenerateQuietMoves => {
                    self.stage = Stage::QuietMoves;
                    self.gen_quiet_moves(hh, board);
                    self.moves.sort_unstable_by_key(Move::score)
                }

                Stage::QuietMoves => {
                    return match self.moves.pop() {
                        Some(m) => {
                            if self.checked_priority_moves.contains(&m) {
                                continue;
                            }
                            Some(m)
                        }
                        None => {
                            None
                        },
                    }
                }
            }
        }
    }

    pub fn is_bad_capture_move(&self) -> bool {
        matches!(self.stage, Stage::PostponedBadCaptureMoves)
    }

    pub fn next_root_move(&mut self, hh: &HistoryHeuristics, board: &mut Board, randomize: bool) -> Option<Move> {
        if !self.moves_generated {
            self.gen_capture_moves::<true>(board);
            self.capture_moves.append(&mut self.bad_capture_moves);
            swap(&mut self.moves, &mut self.capture_moves);
            self.gen_quiet_moves(hh, board);

            let active_player = board.active_player();
            self.moves.retain(|&m| board.is_legal_move(active_player, m));
            if randomize {
                for m in self.moves.iter_mut() {
                    let r = (self.rnd.rand32() % 30) as i16 - 15;
                    let (previous_piece, removed_piece_id) = board.perform_move(*m);
                    let eval = (-board.eval() / 16) * 16;
                    board.undo_move(*m, previous_piece, removed_piece_id);
                    *m = m.with_score(eval + r);
                }
            }

            self.moves.sort_by_key(|&m| Reverse(if m == self.scored_hash_move { MAX_SCORE } else { m.score() }));
            self.moves_generated = true;
        }

        if self.root_move_index >= self.moves.len() {
            return None;
        }

        self.root_move_index += 1;
        Some(unsafe { *self.moves.get_unchecked(self.root_move_index - 1) })
    }

    #[inline(always)]
    pub fn generate_qs_captures(&mut self, board: &mut Board) {
        self.gen_capture_moves::<false>(board);
        self.capture_moves.sort_unstable_by_key(Move::score);
    }

    #[inline(always)]
    pub fn next_good_capture_move(&mut self, board: &mut Board) -> Option<Move> {
        while let Some(m) = self.capture_moves.pop() {
            if !is_bad_capture(m, board) {
                return Some(m);
            }
        }

        None
    }

    fn gen_quiet_moves(&mut self, hh: &HistoryHeuristics, board: &Board) {
        let active_player = self.active_player;
        let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        if active_player.is_white() {
            let pawns = board.get_bitboard(P);
            self.gen_white_straight_pawn_moves(hh, pawns, empty_bb);
        } else {
            let pawns = board.get_bitboard(-P);
            self.gen_black_straight_pawn_moves(hh, pawns, empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(B)) {
            let attacks = get_bishop_attacks(empty_bb.0, pos as usize);
            self.add_moves(hh, MoveType::BishopQuiet,  pos as i8, attacks & empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(N)) {
            let attacks = get_knight_attacks(pos as usize);
            self.add_moves(hh, MoveType::KnightQuiet,  pos as i8, attacks & empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(R)) {
            let attacks = get_rook_attacks(empty_bb.0, pos as usize);
            self.add_moves(hh, MoveType::RookQuiet,  pos as i8, attacks & empty_bb);
        }

        for pos in board.get_bitboard(active_player.piece(Q)) {
            let attacks = get_queen_attacks(empty_bb.0, pos as usize);
            self.add_moves(hh, MoveType::QueenQuiet,  pos as i8, attacks & empty_bb);
        }

        let pos = board.king_pos(active_player);
        let king_targets = get_king_attacks(pos as usize);
        self.gen_quiet_king_moves(hh, active_player, board, pos, empty_bb, king_targets);

    }

    fn gen_capture_moves<const MINOR_PROMOTIONS: bool>(&mut self, board: &Board) {
        let active_player = self.active_player;

        let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
        let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
        let empty_bb = !occupied;

        if active_player.is_white() {
            let pawns = board.get_bitboard(P);
            self.gen_white_attack_pawn_moves::<MINOR_PROMOTIONS>(board, pawns, opponent_bb);
            self.gen_white_en_passant_moves(board, pawns);
        } else {
            let pawns = board.get_bitboard(-P);
            self.gen_black_attack_pawn_moves::<MINOR_PROMOTIONS>(board, pawns, opponent_bb);
            self.gen_black_en_passant_moves(board, pawns);
        }

        for pos in board.get_bitboard(active_player.piece(N)) {
            let attacks = get_knight_attacks(pos as usize);
            self.add_capture_moves(board, MoveType::KnightCapture, pos as i8, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(B)) {
            let attacks = get_bishop_attacks(empty_bb.0, pos as usize);
            self.add_capture_moves(board, MoveType::BishopCapture, pos as i8, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(R)) {
            let attacks = get_rook_attacks(empty_bb.0, pos as usize);
            self.add_capture_moves(board, MoveType::RookCapture, pos as i8, attacks & opponent_bb);
        }

        for pos in board.get_bitboard(active_player.piece(Q)) {
            let attacks = get_queen_attacks(empty_bb.0, pos as usize);
            self.add_capture_moves(board, MoveType::QueenCapture, pos as i8, attacks & opponent_bb);
        }

        let king_pos = board.king_pos(active_player);
        let king_targets = get_king_attacks(king_pos as usize);
        self.add_capture_moves(board, MoveType::KingCapture, king_pos, king_targets & opponent_bb);
    }

    fn gen_white_straight_pawn_moves(&mut self, hh: &HistoryHeuristics, pawns: BitBoard, empty_bb: BitBoard) {
        // Single move
        let mut target_bb = (pawns >> 8) & empty_bb;
        self.add_pawn_quiet_moves(hh, target_bb, 8);

        // Double move
        target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(WHITE.idx()) });
        target_bb >>= BitBoard(8);

        target_bb &= empty_bb;
        for end in target_bb {
            let start = end as i8 + 16;
            self.add_move(hh, MoveType::PawnDoubleQuiet, start, end as i8);
        }
    }

    #[inline(always)]
    fn gen_white_attack_pawn_moves<const MINOR_PROMOTIONS: bool>(&mut self, board: &Board, pawns: BitBoard, opponent_bb: BitBoard) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks >>= BitBoard(9);

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves::<MINOR_PROMOTIONS>(board, left_attacks, 9);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks >>= BitBoard(7);

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves::<MINOR_PROMOTIONS>(board, right_attacks, 7);
    }

    fn gen_white_en_passant_moves(&mut self, board: &Board, pawns: BitBoard) {
        let en_passant = board.enpassant_target();
        if en_passant < WhiteBoardPos::EnPassantLineStart as u8 || en_passant > WhiteBoardPos::EnPassantLineEnd as u8  {
            return;
        }

        let end = en_passant as i8;
        if en_passant != WhiteBoardPos::EnPassantLineEnd as u8 {
            let start = end + 9;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnEnPassant, start, end);
            }
        }

        if en_passant != WhiteBoardPos::EnPassantLineStart as u8 {
            let start = end + 7;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnEnPassant, start, end);
            }
        }
    }

    fn gen_black_straight_pawn_moves(&mut self, hh: &HistoryHeuristics, pawns: BitBoard, empty_bb: BitBoard) {
        // Single move
        let mut target_bb = (pawns << 8) & empty_bb;
        self.add_pawn_quiet_moves(hh, target_bb, -8);

        // Double move
        target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(BLACK.idx()) });
        target_bb <<= BitBoard(8);

        target_bb &= empty_bb;
        for end in target_bb {
            let start = end as i8 - 16;
            self.add_move(hh, MoveType::PawnDoubleQuiet, start, end as i8);
        }
    }

    #[inline(always)]
    fn gen_black_attack_pawn_moves<const MINOR_PROMOTIONS: bool>(&mut self, board: &Board, pawns: BitBoard, opponent_bb: BitBoard) {
        let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
        left_attacks <<= BitBoard(7);

        left_attacks &= opponent_bb;
        self.add_pawn_capture_moves::<MINOR_PROMOTIONS>(board, left_attacks, -7);

        let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
        right_attacks <<= BitBoard(9);

        right_attacks &= opponent_bb;
        self.add_pawn_capture_moves::<MINOR_PROMOTIONS>(board, right_attacks, -9);
    }

    fn gen_black_en_passant_moves(&mut self, board: &Board, pawns: BitBoard) {
        let en_passant = board.enpassant_target();
        if en_passant < BlackBoardPos::EnPassantLineStart as u8 || en_passant > BlackBoardPos::EnPassantLineEnd as u8 {
            return;
        }

        let end = en_passant as i8;
        if en_passant != BlackBoardPos::EnPassantLineStart as u8 {
            let start = end - 9;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnEnPassant, start, end);
            }
        }

        if en_passant != BlackBoardPos::EnPassantLineEnd as u8 {
            let start = end - 7;
            if (pawns & (1 << start)).is_occupied() {
                self.add_capture_move(board, MoveType::PawnEnPassant, start, end);
            }
        }
    }

    fn add_pawn_quiet_moves(&mut self, hh: &HistoryHeuristics, target_bb: BitBoard, direction: i8) {
        // Promotions
        for end in target_bb & 0xFF000000000000FF {
            let start = end as i8 + direction;
            self.add_promotion_move(hh, MoveType::QueenQuietPromotion, start, end as i8, false);
            self.add_promotion_move(hh, MoveType::KnightQuietPromotion, start, end as i8, true);
            self.add_promotion_move(hh, MoveType::RookQuietPromotion, start, end as i8, true);
            self.add_promotion_move(hh, MoveType::BishopQuietPromotion, start, end as i8, true);
        }

        // Normal quiet moves
        for end in target_bb & !0xFF000000000000FF {
            let start = end as i8 + direction;
            self.add_move(hh, MoveType::PawnQuiet, start, end as i8);
        }
    }

    fn add_pawn_capture_moves<const MINOR_PROMOTIONS: bool>(&mut self, board: &Board, target_bb: BitBoard, direction: i8) {
        // Promotions
        for end in target_bb & 0xFF000000000000FF {
            let start = end as i8 + direction;
            self.add_capture_promotion_move(board, MoveType::QueenCapturePromotion, start, end as i8, false);
            if MINOR_PROMOTIONS {
                self.add_capture_promotion_move(board, MoveType::KnightCapturePromotion, start, end as i8, true);
                self.add_capture_promotion_move(board, MoveType::RookCapturePromotion, start, end as i8, true);
                self.add_capture_promotion_move(board, MoveType::BishopCapturePromotion, start, end as i8, true);
            }
        }

        // Normal captures
        for end in target_bb & !0xFF000000000000FF {
            let start = end as i8 + direction;
            self.add_capture_move(board, MoveType::PawnCapture, start, end as i8);
        }
    }

    fn gen_quiet_king_moves(
        &mut self, hh: &HistoryHeuristics, color: Color, board: &Board, pos: i8, empty_bb: BitBoard, king_targets: BitBoard,
    ) {
        // Normal moves
        self.add_moves(hh, MoveType::KingQuiet, pos, king_targets & empty_bb);

        // // Castling moves
        if !board.castling_rules.is_king_start(color, pos) {
            return;
        }

        if board.can_castle_king_side(color) && board.castling_rules.is_ks_castling_valid(color, board, empty_bb) {
            self.add_move(hh, MoveType::KingKSCastling, pos, board.castling_rules.ks_rook_start(color));
        }

        if board.can_castle_queen_side(color) && board.castling_rules.is_qs_castling_valid(color, board, empty_bb) {
            self.add_move(hh, MoveType::KingQSCastling, pos, board.castling_rules.qs_rook_start(color));
        }
    }

}

// If the given move is a bad capture (i.e. has a negative SEE value), the search can be skipped for now and the move will be stored in a separate "bad capture" list
#[inline(always)]
fn is_bad_capture(m: Move, board: &Board) -> bool {
    if matches!(m.move_type(), MoveType::PawnEnPassant) {
        return false;
    }
    let captured_piece_id = board.get_item(m.end() as usize).abs();
    let own_piece_id = m.move_type().piece_id();
    captured_piece_id < own_piece_id
        && board.has_negative_see(
        board.active_player().flip(),
        m.start() as usize,
        m.end() as usize,
        own_piece_id,
        captured_piece_id,
        board.occupancy_bb(),
    )
}

pub fn is_valid_move(board: &Board, active_player: Color, m: Move) -> bool {
    let opponent_bb = board.get_all_piece_bitboard(active_player.flip());
    let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
    let empty_bb = !occupied;

    let start = m.start();
    let end = m.end();

    match m.move_type() {
        MoveType::PawnQuiet => {
            if !(8..=55).contains(&end) {
                return false;
            }

            let pawns = board.get_bitboard(active_player.piece(P)) & 1 << start;
            if pawns.is_empty() {
                return false;
            }

            if active_player.is_white() {
                let target_bb = (pawns >> 8) & empty_bb;
                target_bb.is_set(end as usize)

            } else {
                let target_bb = (pawns << 8) & empty_bb;
                target_bb.is_set(end as usize)
            }
        }
        MoveType::PawnCapture => {
            let pawns = board.get_bitboard(active_player.piece(P)) & 1 << start;
            if pawns.is_empty() {
                return false;
            }

            if active_player.is_white() {
                let mut left_attacks = pawns & 0xfefefefefefefefe;
                left_attacks >>= BitBoard(9);
                left_attacks &= opponent_bb;
                if left_attacks.is_set(end as usize) {
                    return true;
                }

                let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
                right_attacks >>= BitBoard(7);
                right_attacks &= opponent_bb;
                if right_attacks.is_set(end as usize) {
                    return true;
                }

                false
            } else {
                let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
                left_attacks <<= BitBoard(7);
                left_attacks &= opponent_bb;
                if left_attacks.is_set(end as usize) {
                    return true;
                }

                let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
                right_attacks <<= BitBoard(9);
                right_attacks &= opponent_bb;
                if right_attacks.is_set(end as usize) {
                    return true;
                }

                false
            }
        }
        MoveType::PawnDoubleQuiet => {
            if !(8..=55).contains(&end) {
                return false;
            }

            let pawns = board.get_bitboard(active_player.piece(P)) & 1 << start;
            if pawns.is_empty() {
                return false;
            }

            if active_player.is_white() {
                let mut target_bb = (pawns >> 8) & empty_bb;
                target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(WHITE.idx()) });
                target_bb >>= BitBoard(8);
                target_bb &= empty_bb;

                target_bb.is_set(end as usize)

            } else {
                let mut target_bb = (pawns << 8) & empty_bb;
                target_bb &= BitBoard(unsafe { *PAWN_DOUBLE_MOVE_LINES.get_unchecked(BLACK.idx()) });
                target_bb <<= BitBoard(8);
                target_bb &= empty_bb;
                target_bb.is_set(end as usize)
            }
        }
        MoveType::PawnEnPassant => {
            let pawns = board.get_bitboard(active_player.piece(P)) & 1 << start;
            if pawns.is_empty() {
                return false;
            }

            if active_player.is_white() {
                let en_passant = board.enpassant_target();
                if en_passant == 0 {
                    return false;
                }
                let calc_end = en_passant;
                if calc_end as i8 != end {
                    return false;
                }

                if en_passant != WhiteBoardPos::EnPassantLineEnd as u8 {
                    let start = calc_end + 9;
                    if (pawns & (1 << start)).is_occupied() {
                        return true;
                    }
                }

                if en_passant != WhiteBoardPos::EnPassantLineStart as u8 {
                    let start = calc_end + 7;
                    if (pawns & (1 << start)).is_occupied() {
                        return true;
                    }
                }
                false

            } else {
                let en_passant = board.enpassant_target();
                if en_passant == 0 {
                    return false;
                }
                
                let calc_end = en_passant as i8;
                if calc_end != end {
                    return false;
                }
                if en_passant != BlackBoardPos::EnPassantLineStart as u8 {
                    let start = calc_end - 9;
                    if (pawns & (1 << start)).is_occupied() {
                        return true;
                    }
                }

                if en_passant != BlackBoardPos::EnPassantLineEnd as u8 {
                    let start = calc_end - 7;
                    if (pawns & (1 << start)).is_occupied() {
                        return true;
                    }
                }
                false
            }
        }
        MoveType::KnightQuiet => {
            if !board.get_bitboard(active_player.piece(N)).is_set(start as usize) {
                return false;
            }

            let attacks = get_knight_attacks(start as usize) & empty_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KnightCapture => {
            if !board.get_bitboard(active_player.piece(N)).is_set(start as usize) {
                return false;
            }

            let attacks = get_knight_attacks(start as usize) & opponent_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KnightQuietPromotion | MoveType::BishopQuietPromotion | MoveType::RookQuietPromotion | MoveType::QueenQuietPromotion => {
            let direction = end - start;
            if direction.signum() != active_player.piece(-1) {
                return false;
            }

            if !board.get_bitboard(active_player.piece(P)).is_set(start as usize) {
                return false;
            }

            if (start - end).abs() != 8 {
                return false;
            }

            if (8..=55).contains(&end) {
                return false;
            }

            if occupied.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KnightCapturePromotion | MoveType::BishopCapturePromotion | MoveType::RookCapturePromotion | MoveType::QueenCapturePromotion => {
            let direction = end - start;
            if direction.signum() != active_player.piece(-1) {
                return false;
            }

            if (start - end).abs() != 9 && (start - end).abs() != 7 {
                return false;
            }

            if (8..=55).contains(&end) {
                return false;
            }

            if !opponent_bb.is_set(end as usize) {
                return false;
            }

            if !board.get_bitboard(active_player.piece(P)).is_set(start as usize) {
                return false;
            }

            true
        }
        MoveType::BishopQuiet => {
            if !board.get_bitboard(active_player.piece(B)).is_set(start as usize) {
                return false;
            }

            let attacks = get_bishop_attacks(empty_bb.0, start as usize) & empty_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::BishopCapture => {
            if !board.get_bitboard(active_player.piece(B)).is_set(start as usize) {
                return false;
            }

            let attacks = get_bishop_attacks(empty_bb.0, start as usize) & opponent_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::RookQuiet => {
            if !board.get_bitboard(active_player.piece(R)).is_set(start as usize) {
                return false;
            }

            let attacks = get_rook_attacks(empty_bb.0, start as usize) & empty_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::RookCapture => {
            if !board.get_bitboard(active_player.piece(R)).is_set(start as usize) {
                return false;
            }

            let attacks = get_rook_attacks(empty_bb.0, start as usize) & opponent_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::QueenQuiet => {
            if !board.get_bitboard(active_player.piece(Q)).is_set(start as usize) {
                return false;
            }

            let attacks = get_queen_attacks(empty_bb.0, start as usize) & empty_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::QueenCapture => {
            if !board.get_bitboard(active_player.piece(Q)).is_set(start as usize) {
                return false;
            }

            let attacks = get_queen_attacks(empty_bb.0, start as usize) & opponent_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KingQuiet => {
            if board.king_pos(active_player) != start {
                return false;
            }

            let attacks = get_king_attacks(start as usize) & empty_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KingCapture => {
            if board.king_pos(active_player) != start {
                return false;
            }

            let attacks = get_king_attacks(start as usize) & opponent_bb;
            if !attacks.is_set(end as usize) {
                return false;
            }

            true
        }
        MoveType::KingQSCastling => {
            if board.king_pos(active_player) != start {
                return false;
            }

            if end != board.castling_rules.qs_rook_start(active_player) {
                return false;
            }

            if !board.can_castle_queen_side(active_player) || !board.castling_rules.is_qs_castling_valid(active_player, board, empty_bb) {
                return false;
            }

            true
        }
        MoveType::KingKSCastling => {
            if board.king_pos(active_player) != start {
                return false;
            }

            if end != board.castling_rules.ks_rook_start(active_player) {
                return false;
            }

            if !board.can_castle_king_side(active_player) || !board.castling_rules.is_ks_castling_valid(active_player, board, empty_bb) {
                return false;
            }

            true
        }

        _ => false
    }
}

// Evaluate score for capture move ordering
#[inline(always)]
fn evaluate_capture_move_order(board: &Board, end: i8, piece_id: i8) -> i16 {
    let captured_piece_id = board.get_item(end as usize).abs();
    let original_piece_id = piece_id;

    captured_piece_id as i16 * 16 - original_piece_id as i16
}

#[inline]
pub fn is_killer(m: Move) -> bool {
    m.score() == PRIMARY_KILLER_SCORE || m.score() == SECONDARY_KILLER_SCORE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::castling::{CastlingRules, CastlingState};
    use crate::history_heuristics::EMPTY_HISTORY;
    use crate::init::init;
    use crate::pieces::K;

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
        let mut board = setup(WHITE, P, 52);
        board.add_piece(WHITE, P, 44);

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(0, moves.len());
    }

    #[test]
    pub fn white_queen_moves() {
        let mut board = setup(WHITE, Q, 28);

        let moves = generate_moves_for_pos(&mut board, WHITE, 28);

        assert_eq!(27, moves.len());
    }

    #[test]
    pub fn exclude_illegal_moves() {
        let mut board = setup(WHITE, Q, 52);
        board.perform_move(Move::new(MoveType::KingQuiet, board.king_pos(WHITE), 53));
        board.add_piece(BLACK, R, 51);

        board.perform_null_move(); // so WHITE is the active player

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(1, moves.len(), "There must be only one legal move for the white queen");
    }

    fn setup(color: Color, piece_id: i8, pos: i32) -> Board {
        init();

        let mut items = ONLY_KINGS;
        items[pos as usize] = color.piece(piece_id);
        Board::new(&items, color, CastlingState::default(), None, 0, 1, CastlingRules::default())
    }

    fn generate_moves_for_pos(board: &mut Board, color: Color, pos: i32) -> Vec<Move> {
        let hh = HistoryHeuristics::default();
        let mut ml = MoveList::default();
        ml.init(color, NO_MOVE, EMPTY_HISTORY);

        let mut moves = Vec::new();

        loop {
            let m = ml.next_move(0, &hh, board);

            if let Some(m) = m {
                moves.push(m);
            } else {
                break;
            }
        }

        moves.into_iter()
            .filter(|&m| m.start() == pos as i8)
            .filter(|&m| board.is_legal_move(color, m))
            .collect()
    }
}
