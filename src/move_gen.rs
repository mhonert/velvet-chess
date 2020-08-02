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

use crate::board::{Board};
use crate::colors::{Color, WHITE, BLACK};
use crate::pieces::{K, P, Q, R, B, N};
use crate::boardpos::{BlackBoardPos, WhiteBoardPos};
use crate::castling::Castling;
use crate::bitboard::{WHITE_KING_SIDE_CASTLING_BIT_PATTERN, WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN, BLACK_KING_SIDE_CASTLING_BIT_PATTERN, BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN, PAWN_DOUBLE_MOVE_LINES};
use core::fmt;
use std::fmt::Debug;

pub fn generate_moves(board: &Board, active_player: Color) -> Vec<Move> {
    let mut moves: Vec<Move> = Vec::with_capacity(128);

    let opponent_bb = board.get_all_piece_bitboard(-active_player);
    let occupied = opponent_bb | board.get_all_piece_bitboard(active_player);
    let empty_bb = !occupied;

    if active_player == WHITE {
        gen_white_king_moves(&mut moves, board.king_pos(WHITE), board, opponent_bb, empty_bb);
        gen_white_pawn_moves(&mut moves, board, opponent_bb, empty_bb);

    } else {
        gen_black_king_moves(&mut moves, board.king_pos(BLACK), board, opponent_bb, empty_bb);
        gen_black_pawn_moves(&mut moves, board, opponent_bb, empty_bb);

    }

    let mut knights = board.get_bitboard(N * active_player);
    while knights != 0 {
        let pos = knights.trailing_zeros();
        knights ^= 1 << pos as u64;
        let attacks = board.bb.get_knight_attacks(pos as i32);
        gen_piece_moves(&mut moves, N, pos as i32, attacks, opponent_bb, empty_bb);
    }

    let mut bishops = board.get_bitboard(B * active_player);
    while bishops != 0 {
        let pos = bishops.trailing_zeros();
        bishops ^= 1 << pos as u64;
        let attacks = board.bb.get_diagonal_attacks(occupied, pos as i32)
            | board.bb.get_anti_diagonal_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, B, pos as i32, attacks, opponent_bb, empty_bb);
    }

    let mut rooks = board.get_bitboard(R * active_player);
    while rooks != 0 {
        let pos = rooks.trailing_zeros();
        rooks ^= 1 << pos as u64;
        let attacks = board.bb.get_horizontal_attacks(occupied, pos as i32)
            | board.bb.get_vertical_attacks(occupied, pos as i32);
        gen_piece_moves(&mut moves, R, pos as i32, attacks, opponent_bb, empty_bb);
    }

    let mut queens = board.get_bitboard(Q * active_player);
    while queens != 0 {
        let pos = queens.trailing_zeros();
        queens ^= 1 << pos as u64;
        let attacks = board.bb.get_horizontal_attacks(occupied, pos as i32)
            | board.bb.get_vertical_attacks(occupied, pos as i32)
            | board.bb.get_diagonal_attacks(occupied, pos as i32)
            | board.bb.get_anti_diagonal_attacks(occupied, pos as i32);

        gen_piece_moves(&mut moves, Q, pos as i32, attacks, opponent_bb, empty_bb);
    }

    moves
}

fn gen_piece_moves(moves: &mut Vec<Move>, piece: i8, pos: i32, targets: u64, opponent_bb: u64, empty_bb: u64) {
    // Captures
    add_moves(moves, piece, pos, targets & opponent_bb);

    // Normal moves
    add_moves(moves, piece, pos, targets & empty_bb);
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
            moves.push(encode_move(P, start as i32, end as i32));
        }
    }

    if en_passant != 0b00000001 {
        let start = end + 7;
        if (pawns & (1 << start)) != 0 {
            moves.push(encode_move(P, start as i32, end as i32));
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
            moves.push(encode_move(P, start as i32, end as i32));
        }
    }

    if en_passant != 0b10000000 {
        let start = end - 7;
        if (pawns & (1 << start)) != 0 {
            moves.push(encode_move(P, start as i32, end as i32));
        }
    }
}

fn add_pawn_moves(moves: &mut Vec<Move>, target_bb: u64, direction: i32) {
    let mut bb = target_bb;
    while bb != 0 {
        let end = bb.trailing_zeros();
        bb ^= 1 << (end as u64);
        let start = end as i32 + direction;

        if end <= 7 || end >= 56 {
            // Promotion
            moves.push(encode_move(Q, start, end as i32));
            moves.push(encode_move(R, start, end as i32));
            moves.push(encode_move(B, start, end as i32));
            moves.push(encode_move(N, start, end as i32));

        } else {
            // Normal move
            moves.push(encode_move(P, start, end as i32));
        }
    }
}

fn gen_white_king_moves(moves: &mut Vec<Move>, pos: i32, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let king_targets = board.bb.get_king_attacks(pos);

    // Captures
    add_moves(moves, K, pos, king_targets & opponent_bb);

    // Normal moves
    add_moves(moves, K, pos, king_targets & empty_bb);

    // // Castling moves
    if pos != WhiteBoardPos::KingStart as i32 {
        return;
    }

    if board.can_castle(Castling::WhiteKingSide) && is_kingside_castling_valid_for_white(board, empty_bb) {
        moves.push(encode_move(K, pos, pos + 2));
    }

    if board.can_castle(Castling::WhiteQueenSide) && is_queenside_castling_valid_for_white(board, empty_bb) {
        moves.push(encode_move(K, pos, pos - 2));
    }
}

fn gen_black_king_moves(moves: &mut Vec<Move>, pos: i32, board: &Board, opponent_bb: u64, empty_bb: u64) {
    let king_targets = board.bb.get_king_attacks(pos);

    // Captures
    add_moves(moves, K, pos, king_targets & opponent_bb);

    // Normal moves
    add_moves(moves, K, pos, king_targets & empty_bb);

    // Castling moves
    if pos != BlackBoardPos::KingStart as i32 {
        return;
    }

    if board.can_castle(Castling::BlackKingSide) && is_kingside_castling_valid_for_black(board, empty_bb) {
        moves.push(encode_move(K, pos, pos + 2));
    }

    if board.can_castle(Castling::BlackQueenSide) && is_queenside_castling_valid_for_black(board, empty_bb) {
        moves.push(encode_move(K, pos, pos - 2));
    }
}

fn add_moves(moves: &mut Vec<Move>, piece: i8, pos: i32, target_bb: u64) {
    let mut bb =  target_bb;
    while bb != 0 {
        let end = bb.trailing_zeros();
        bb ^= 1 << (end as u64);
        moves.push(encode_move(piece, pos, end as i32));
    }
}

fn is_kingside_castling_valid_for_white(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & WHITE_KING_SIDE_CASTLING_BIT_PATTERN) == WHITE_KING_SIDE_CASTLING_BIT_PATTERN &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32) &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 + 1) &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 + 2)
}

fn is_queenside_castling_valid_for_white(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN) == WHITE_QUEEN_SIDE_CASTLING_BIT_PATTERN &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32) &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 - 1) &&
        !board.is_attacked(BLACK, WhiteBoardPos::KingStart as i32 - 2)
}

fn is_kingside_castling_valid_for_black(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & BLACK_KING_SIDE_CASTLING_BIT_PATTERN) == BLACK_KING_SIDE_CASTLING_BIT_PATTERN &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32) &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 + 1) &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 + 2)
}

fn is_queenside_castling_valid_for_black(board: &Board, empty_bb: u64) -> bool {
    (empty_bb & BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN) == BLACK_QUEEN_SIDE_CASTLING_BIT_PATTERN &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32) &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 - 1) &&
        !board.is_attacked(WHITE, BlackBoardPos::KingStart as i32 - 2)
}

pub type Move = u32;

fn print_move(m: Move) {
    println!("Move {:?} from {:?} to {:?}", decode_piece_id(m), decode_start_index(m), decode_end_index(m));
}

pub fn encode_move(piece: i8, start: i32, end: i32) -> Move {
    (piece.abs() as u32) | ((start as u32) << 3) | ((end as u32) << 10)
}

pub fn decode_piece_id(m: Move) -> u32  {
    m & 0x7
}

pub fn decode_start_index(m: Move) -> i32  {
    ((m >> 3) & 0x7F) as i32
}

pub fn decode_end_index(m: Move) -> i32  {
    ((m >> 10) & 0x7F) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONLY_KINGS: [i8; 64] = [
        0,  0,  0,  0,  0,  0, -K,  0, // 0 - 7
        0,  0,  0,  0,  0,  0,  0,  0, // 8 - 15
        0,  0,  0,  0,  0,  0,  0,  0, // 16 - 23
        0,  0,  0,  0,  0,  0,  0,  0, // 24 - 31
        0,  0,  0,  0,  0,  0,  0,  0, // 32 - 39
        0,  0,  0,  0,  0,  0,  0,  0, // 40 - 47
        0,  0,  0,  0,  0,  0,  0,  0, // 48 - 55
        0,  0,  0,  0,  0,  0,  K,  0, // 56 - 63
    ];

    # [test]
    pub fn white_pawn_moves_blocked() {
        let mut board: Board = board_with_one_piece(WHITE, P, 52);
        board.add_piece(WHITE, P, 44);

        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(0, moves.len());

    }

    # [test]
    pub fn white_queen_moves() {
        let mut board: Board = board_with_one_piece(WHITE, Q, 28);

        let moves = generate_moves_for_pos(&mut board, WHITE, 28);

        assert_eq!(27, moves.len());
    }

    # [test]
    pub fn exclude_illegal_moves() {
        let mut board: Board = board_with_one_piece(WHITE, Q, 52);
        board.perform_move(K, board.king_pos(WHITE), 53);
        board.add_piece(BLACK, R, 51);


        let moves = generate_moves_for_pos(&mut board, WHITE, 52);
        assert_eq!(1, moves.len(), "There must be only one legal move for the white queen");
    }

    fn board_with_one_piece(color: Color, piece_id: i8, pos: i32) -> Board {
        let mut items = ONLY_KINGS;
        items[pos as usize] = piece_id * color;
        Board::new(&items, color, 0, None, 0, 1)
    }

    fn generate_moves_for_pos(board: &mut Board, color: Color, pos: i32) -> Vec<Move> {
        generate_moves(board, color)
            .into_iter()
            .filter(|&m| decode_start_index(m) == pos)
            .filter(|&m| board.is_legal_move(color, decode_piece_id(m) as i8, decode_start_index(m), decode_end_index(m)))
            .collect()
    }
}