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

use crate::bitboard::{BLACK_KING_SIDE_CASTLING_BIT_PATTERN, BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN, PAWN_DOUBLE_MOVE_LINES, WHITE_KING_SIDE_CASTLING_BIT_PATTERN, WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN, BitBoard, get_knight_attacks, get_bishop_attacks, get_rook_attacks, get_queen_attacks, get_king_attacks};
use crate::board::Board;
use crate::boardpos::{BlackBoardPos, WhiteBoardPos};
use crate::castling::Castling;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, K, N, P, Q, R};


pub fn generate_moves(board: &Board, active_player: Color) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::with_capacity(64);

    let opponent_bb = board.get_all_piece_bitboard(-active_player);
    let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
    let empty_bb = !occupied;

    if active_player == WHITE {
        gen_white_king_moves(
            &mut moves,
            board.king_pos(WHITE),
            board,
            opponent_bb,
            empty_bb,
        );
        gen_white_pawn_moves(&mut moves, board, opponent_bb, empty_bb);
    } else {
        gen_black_king_moves(
            &mut moves,
            board.king_pos(BLACK),
            board,
            opponent_bb,
            empty_bb,
        );
        gen_black_pawn_moves(&mut moves, board, opponent_bb, empty_bb);
    }

    for pos in BitBoard(board.get_bitboard(N * active_player)) {
        let attacks = get_knight_attacks(pos as i32);
        gen_piece_moves(&mut moves, N, pos as i32, attacks, opponent_bb, empty_bb);
    }

    for pos in BitBoard(board.get_bitboard(B * active_player)) {
        let attacks = get_bishop_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, B, pos as i32, attacks, opponent_bb, empty_bb);
    }

    for pos in BitBoard(board.get_bitboard(R * active_player)) {
        let attacks = get_rook_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, R, pos as i32, attacks, opponent_bb, empty_bb);
    }

    for pos in BitBoard(board.get_bitboard(Q * active_player)) {
        let attacks = get_queen_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, Q, pos as i32, attacks, opponent_bb, empty_bb);
    }

    moves
}

pub fn generate_capture_moves(board: &Board, active_player: Color) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::with_capacity(16);

    let opponent_bb = board.get_all_piece_bitboard(-active_player);
    let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);

    if active_player == WHITE {
        gen_white_attack_pawn_moves(&mut moves, board.get_bitboard(P), opponent_bb);

        let king_pos = board.king_pos(WHITE);
        let king_targets = get_king_attacks(king_pos);
        add_moves(&mut moves, K, king_pos, king_targets & opponent_bb);
    } else {
        gen_black_attack_pawn_moves(&mut moves, board.get_bitboard(-P), opponent_bb);

        let king_pos = board.king_pos(BLACK);
        let king_targets = get_king_attacks(king_pos);
        add_moves(&mut moves, K, king_pos, king_targets & opponent_bb);
    }

    for pos in BitBoard(board.get_bitboard(N * active_player)) {
        let attacks = get_knight_attacks(pos as i32);
        gen_piece_capture_moves(&mut moves, N, pos as i32, attacks, opponent_bb);
    }

    for pos in BitBoard(board.get_bitboard(B * active_player)) {
        let attacks = get_bishop_attacks(occupied, pos as i32);
        gen_piece_capture_moves(&mut moves, B, pos as i32, attacks, opponent_bb);
    }

    for pos in BitBoard(board.get_bitboard(R * active_player)) {
        let attacks = get_rook_attacks(occupied, pos as i32);
        gen_piece_capture_moves(&mut moves, R, pos as i32, attacks, opponent_bb);
    }

    for pos in BitBoard(board.get_bitboard(Q * active_player)) {
        let attacks = get_queen_attacks(occupied, pos as i32);
        gen_piece_capture_moves(&mut moves, Q, pos as i32, attacks, opponent_bb);
    }

    moves
}

pub fn has_valid_moves(board: &mut Board, active_player: Color) -> bool {
    let mut moves: Vec<Move> = Vec::with_capacity(27);

    let opponent_bb = board.get_all_piece_bitboard(-active_player);
    let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
    let empty_bb = !occupied;

    if active_player == WHITE {
        gen_white_king_moves(
            &mut moves,
            board.king_pos(WHITE),
            board,
            opponent_bb,
            empty_bb,
        );
        if any_moves_allow_check_evasion(board, &mut moves, active_player) {
            return true;
        }
    } else {
        gen_black_king_moves(
            &mut moves,
            board.king_pos(BLACK),
            board,
            opponent_bb,
            empty_bb,
        );
        if any_moves_allow_check_evasion(board, &mut moves, active_player) {
            return true;
        }
    }

    for pos in BitBoard(board.get_bitboard(N * active_player)) {
        let attacks = get_knight_attacks(pos as i32);
        gen_piece_moves(&mut moves, N, pos as i32, attacks, opponent_bb, empty_bb);
    }
    if any_moves_allow_check_evasion(board, &mut moves, active_player) {
        return true;
    }

    for pos in BitBoard(board.get_bitboard(B * active_player)) {
        let attacks = get_bishop_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, B, pos as i32, attacks, opponent_bb, empty_bb);
    }
    if any_moves_allow_check_evasion(board, &mut moves, active_player) {
        return true;
    }

    for pos in BitBoard(board.get_bitboard(R * active_player)) {
        let attacks = get_rook_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, R, pos as i32, attacks, opponent_bb, empty_bb);
    }
    if any_moves_allow_check_evasion(board, &mut moves, active_player) {
        return true;
    }

    for pos in BitBoard(board.get_bitboard(Q * active_player)) {
        let attacks = get_queen_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, Q, pos as i32, attacks, opponent_bb, empty_bb);
    }
    if any_moves_allow_check_evasion(board, &mut moves, active_player) {
        return true;
    }

    if active_player == WHITE {
        gen_white_pawn_moves(&mut moves, board, opponent_bb, empty_bb);
        if any_moves_allow_check_evasion(board, &mut moves, active_player) {
            return true;
        }
    } else {
        gen_black_pawn_moves(&mut moves, board, opponent_bb, empty_bb);
        if any_moves_allow_check_evasion(board, &mut moves, active_player) {
            return true;
        }
    }
    false
}

fn any_moves_allow_check_evasion(
    board: &mut Board,
    moves: &mut Vec<Move>,
    active_player: Color,
) -> bool {
    for m in moves.iter() {
        if !move_results_in_check(board, m, active_player) {
            return true;
        }
    }
    moves.clear();
    false
}

fn move_results_in_check(board: &mut Board, m: &Move, active_player: Color) -> bool {
    let start = m.start();
    let end = m.end();

    let previous_piece = board.get_item(start);
    let move_state = board.perform_move(m.piece_id() as i8, start, end);
    let check = board.is_in_check(active_player);
    board.undo_move(previous_piece, start, end, move_state);

    check
}

fn gen_piece_moves(
    moves: &mut Vec<Move>,
    piece: i8,
    pos: i32,
    targets: u64,
    opponent_bb: u64,
    empty_bb: u64,
) {
    // Captures
    add_moves(moves, piece, pos, targets & opponent_bb);

    // Normal moves
    add_moves(moves, piece, pos, targets & empty_bb);
}

fn gen_piece_capture_moves(
    moves: &mut Vec<Move>,
    piece: i8,
    pos: i32,
    targets: u64,
    opponent_bb: u64,
) {
    // Captures
    add_moves(moves, piece, pos, targets & opponent_bb);
}

fn gen_white_pawn_moves(moves: &mut Vec<Move>, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let pawns = board.get_bitboard(P);

    gen_white_straight_pawn_moves(moves, pawns, empty_bb);
    gen_white_attack_pawn_moves(moves, pawns, opponent_bb);
    gen_white_en_passant_moves(moves, board, pawns);
}

fn gen_white_straight_pawn_moves(moves: &mut Vec<Move>, pawns: u64, empty_bb: u64) {
    // Single move
    let mut target_bb = (pawns >> 8) & empty_bb;
    add_pawn_moves(moves, target_bb, 8);

    // Double move
    target_bb &= PAWN_DOUBLE_MOVE_LINES[WHITE as usize + 1];
    target_bb >>= 8;

    target_bb &= empty_bb;
    add_pawn_moves(moves, target_bb, 16);
}

fn gen_white_attack_pawn_moves(moves: &mut Vec<Move>, pawns: u64, opponent_bb: u64) {
    let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
    left_attacks >>= 9;

    left_attacks &= opponent_bb;
    add_pawn_moves(moves, left_attacks, 9);

    let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
    right_attacks >>= 7;

    right_attacks &= opponent_bb;
    add_pawn_moves(moves, right_attacks, 7);
}

fn gen_white_en_passant_moves(moves: &mut Vec<Move>, board: &Board, pawns: u64) {
    let en_passant = board.get_enpassant_state();
    if en_passant == 0 {
        return;
    }

    let end = 16 + en_passant.trailing_zeros();
    if en_passant != 0b10000000 {
        let start = end + 9;
        if (pawns & (1 << start)) != 0 {
            moves.push(Move::new(P, start as i32, end as i32));
        }
    }

    if en_passant != 0b00000001 {
        let start = end + 7;
        if (pawns & (1 << start)) != 0 {
            moves.push(Move::new(P, start as i32, end as i32));
        }
    }
}

fn gen_black_pawn_moves(moves: &mut Vec<Move>, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let pawns = board.get_bitboard(-P);

    gen_black_straight_pawn_moves(moves, pawns, empty_bb);
    gen_black_attack_pawn_moves(moves, pawns, opponent_bb);
    gen_black_en_passant_moves(moves, board, pawns);
}

fn gen_black_straight_pawn_moves(moves: &mut Vec<Move>, pawns: u64, empty_bb: u64) {
    // Single move
    let mut target_bb = (pawns << 8) & empty_bb;
    add_pawn_moves(moves, target_bb, -8);

    // Double move
    target_bb &= PAWN_DOUBLE_MOVE_LINES[((BLACK as i32) + 1) as usize];
    target_bb <<= 8;

    target_bb &= empty_bb;
    add_pawn_moves(moves, target_bb, -16);
}

fn gen_black_attack_pawn_moves(moves: &mut Vec<Move>, pawns: u64, opponent_bb: u64) {
    let mut left_attacks = pawns & 0xfefefefefefefefe; // mask right column
    left_attacks <<= 7;

    left_attacks &= opponent_bb;
    add_pawn_moves(moves, left_attacks, -7);

    let mut right_attacks = pawns & 0x7f7f7f7f7f7f7f7f; // mask left column
    right_attacks <<= 9;

    right_attacks &= opponent_bb;
    add_pawn_moves(moves, right_attacks, -9);
}

fn gen_black_en_passant_moves(moves: &mut Vec<Move>, board: &Board, pawns: u64) {
    let en_passant = board.get_enpassant_state() >> 8;
    if en_passant == 0 {
        return;
    }

    let end = 40 + en_passant.trailing_zeros();
    if en_passant != 0b00000001 {
        let start = end - 9;
        if (pawns & (1 << start)) != 0 {
            moves.push(Move::new(P, start as i32, end as i32));
        }
    }

    if en_passant != 0b10000000 {
        let start = end - 7;
        if (pawns & (1 << start)) != 0 {
            moves.push(Move::new(P, start as i32, end as i32));
        }
    }
}

fn add_pawn_moves(moves: &mut Vec<Move>, target_bb: u64, direction: i32) {
    for end in BitBoard(target_bb) {
        let start = end as i32 + direction;

        if end <= 7 || end >= 56 {
            // Promotion
            moves.push(Move::new(Q, start, end as i32));
            moves.push(Move::new(R, start, end as i32));
            moves.push(Move::new(B, start, end as i32));
            moves.push(Move::new(N, start, end as i32));
        } else {
            // Normal move
            moves.push(Move::new(P, start, end as i32));
        }
    }
}

fn gen_white_king_moves(moves: &mut Vec<Move>, pos: i32, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let king_targets = get_king_attacks(pos);

    // Captures
    add_moves(moves, K, pos, king_targets & opponent_bb);

    // Normal moves
    add_moves(moves, K, pos, king_targets & empty_bb);

    // // Castling moves
    if pos != WhiteBoardPos::KingStart as i32 {
        return;
    }

    if board.can_castle(Castling::WhiteKingSide) && is_kingside_castling_valid_for_white(board, empty_bb) {
        moves.push(Move::new(K, pos, pos + 2));
    }

    if board.can_castle(Castling::WhiteQueenSide) && is_queenside_castling_valid_for_white(board, empty_bb) {
        moves.push(Move::new(K, pos, pos - 2));
    }
}

fn gen_black_king_moves(moves: &mut Vec<Move>, pos: i32, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let king_targets = get_king_attacks(pos);

    // Captures
    add_moves(moves, K, pos, king_targets & opponent_bb);

    // Normal moves
    add_moves(moves, K, pos, king_targets & empty_bb);

    // Castling moves
    if pos != BlackBoardPos::KingStart as i32 {
        return;
    }

    if board.can_castle(Castling::BlackKingSide) && is_kingside_castling_valid_for_black(board, empty_bb) {
        moves.push(Move::new(K, pos, pos + 2));
    }

    if board.can_castle(Castling::BlackQueenSide)
        && is_queenside_castling_valid_for_black(board, empty_bb)
    {
        moves.push(Move::new(K, pos, pos - 2));
    }
}

fn add_moves(moves: &mut Vec<Move>, piece: i8, pos: i32, target_bb: u64) {
    for end in BitBoard(target_bb) {
        moves.push(Move::new(piece, pos, end as i32));
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

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub struct Move(u32);

pub const NO_MOVE: Move = Move(0);

impl Move {
    #[inline]
    pub fn new(piece: i8, start: i32, end: i32) -> Self {
        Move((piece.abs() as u32) | ((start as u32) << 3) | ((end as u32) << 10))
    }

    #[inline]
    pub fn with_score(&self, score: i32) -> Move {
        if score < 0 {
            Move(self.0 & 0x1FFFF | 0x80000000 | ((-score as u32) << 17))
        } else {
            Move(self.0 & 0x1FFFF | (score as u32) << 17)
        }
    }

    #[inline]
    pub fn to_u32(&self) -> u32 {
        self.0
    }

    #[inline]
    pub fn from_u32(packed_move: u32) -> Move {
        Move(packed_move)
    }

    #[inline]
    pub fn without_score(&self) -> Move {
        self.with_score(0)
    }

    /// Checks, whether the two moves are the same (except for the score)
    #[inline]
    pub fn is_same_move(&self, m: Move) -> bool {
        (self.0 & 0x1FFFF) == (m.0 & 0x1FFFF)
    }

    #[inline]
    pub fn piece_id(&self) -> i8 {
        (self.0 & 0x7) as i8
    }

    #[inline]
    pub fn start(&self) -> i32 {
        ((self.0 >> 3) & 0x7F) as i32
    }

    #[inline]
    pub fn end(&self) -> i32 {
        ((self.0 >> 10) & 0x7F) as i32
    }

    #[inline]
    pub fn score(&self) -> i32 {
        if self.0 & 0x80000000 != 0 {
            -(((self.0 & 0x7FFE0000) >> 17) as i32)
        } else {
            (self.0 >> 17) as i32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score_util::{MAX_SCORE, MIN_SCORE};

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
        board.perform_move(K, board.king_pos(WHITE), 53);
        board.add_piece(BLACK, R, 51);

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(
            1,
            moves.len(),
            "There must be only one legal move for the white queen"
        );
    }

    #[test]
    fn scored_move_for_max_score() {
        let m = Move::new(Q, 2, 63).with_score(MAX_SCORE);

        assert_eq!(m.piece_id(), Q);
        assert_eq!(m.start(), 2);
        assert_eq!(m.end(), 63);
        assert_eq!(MAX_SCORE, m.score());
    }

    #[test]
    fn scored_move_for_min_score() {
        let m = Move::new(K, 0, 1).with_score(MIN_SCORE);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 0);
        assert_eq!(m.end(), 1);
        assert_eq!(MIN_SCORE, m.score());
    }

    #[test]
    fn scored_move() {
        let score = -1037;
        let m = Move::new(K, 4, 12).with_score(score);

        assert_eq!(m.piece_id(), K);
        assert_eq!(m.start(), 4);
        assert_eq!(m.end(), 12);
        assert_eq!(score, m.score());
    }

    fn board_with_one_piece(color: Color, piece_id: i8, pos: i32) -> Board {
        let mut items = ONLY_KINGS;
        items[pos as usize] = piece_id * color;
        Board::new(&items, color, 0, None, 0, 1)
    }

    fn generate_moves_for_pos(board: &mut Board, color: Color, pos: i32) -> Vec<Move> {
        generate_moves(board, color)
            .into_iter()
            .filter(|&m| m.start() == pos)
            .filter(|&m| board.is_legal_move(color, m))
            .collect()
    }

}
