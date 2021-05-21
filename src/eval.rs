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

use crate::pieces::{P, N, B, R, Q};
use crate::colors::{WHITE, BLACK};
use crate::bitboard::{black_left_pawn_attacks, black_right_pawn_attacks, white_left_pawn_attacks, white_right_pawn_attacks, BitBoard, get_king_danger_zone, get_knight_attacks, get_white_pawn_freepath, get_white_pawn_freesides, get_black_pawn_freepath, get_black_pawn_freesides, get_column_mask, mirror};
use crate::magics::{get_bishop_attacks, get_rook_attacks, get_queen_attacks};
use crate::engine::Engine;

pub trait Eval {
    fn get_score(&mut self) -> i32;
}

impl Eval for Engine {
    fn get_score(&mut self) -> i32 {
        let board = &self.board;
        let white_pawns = board.get_bitboard(P);
        let black_pawns = board.get_bitboard(-P);

        let white_king = board.king_pos(WHITE);
        let black_king = board.king_pos(BLACK);

        let white_pieces = board.get_all_piece_bitboard(WHITE);
        let black_pieces = board.get_all_piece_bitboard(BLACK);

        let occupied = white_pieces | black_pieces;
        let empty_board = !occupied;
        let empty_or_black = empty_board | black_pieces;

        let white_pawn_attacks = white_left_pawn_attacks(white_pawns) | white_right_pawn_attacks(white_pawns);
        let black_pawn_attacks = black_left_pawn_attacks(black_pawns) | black_right_pawn_attacks(black_pawns);

        // Knights
        let white_knights = board.get_bitboard(N);
        let mut white_knight_attacks = 0;
        for pos in BitBoard(white_knights) {
            white_knight_attacks |= get_knight_attacks(pos as i32);
        }
        white_knight_attacks &= empty_or_black;

        let empty_or_white = empty_board | white_pieces;

        let black_knights = board.get_bitboard(-N);
        let mut black_knight_attacks = 0;
        for pos in BitBoard(black_knights) {
            black_knight_attacks |= get_knight_attacks(pos as i32);
        }
        black_knight_attacks &= empty_or_white;

        // Bishops
        let white_bishops = board.get_bitboard(B);
        let mut white_bishop_attacks = 0;
        for pos in BitBoard(white_bishops) {
            white_bishop_attacks |= get_bishop_attacks(empty_board, pos as i32);
        }
        white_bishop_attacks &= empty_or_black;

        let black_bishops = board.get_bitboard(-B);
        let mut black_bishop_attacks = 0;
        for pos in BitBoard(black_bishops) {
            black_bishop_attacks |= get_bishop_attacks(empty_board, pos as i32);
        }
        black_bishop_attacks &= empty_or_white;

        // Rooks
        let white_rooks = board.get_bitboard(R);
        let mut white_rook_attacks = 0;
        for pos in BitBoard(white_rooks) {
            white_rook_attacks |= get_rook_attacks(empty_board, pos as i32);
        }
        white_rook_attacks &= empty_or_black;

        let black_rooks = board.get_bitboard(-R);
        let mut black_rook_attacks = 0;
        for pos in BitBoard(black_rooks) {
            black_rook_attacks |= get_rook_attacks(empty_board, pos as i32);
        }
        black_rook_attacks &= empty_or_white;

        // Queens
        let white_queens = board.get_bitboard(Q);
        let mut white_queen_attacks = 0;
        for pos in BitBoard(white_queens) {
            white_queen_attacks |= get_queen_attacks(empty_board, pos as i32);
        }
        white_queen_attacks &= empty_or_black;

        let black_queens = board.get_bitboard(-Q);
        let mut black_queen_attacks = 0;
        for pos in BitBoard(black_queens) {
            black_queen_attacks |= get_queen_attacks(empty_board, pos as i32);
        }
        black_queen_attacks &= empty_or_white;

        // Genetic eval
        let white_king_half = king_half_bb(white_king);
        let black_king_half = king_half_bb(black_king);

        let white_king_bb = 1u64 << white_king;
        let black_king_bb = 1u64 << black_king;

        let mut score = 0;
        score += self.genetic_eval.eval(white_pawns, black_pawns, white_knights, black_knights, white_bishops, black_bishops,
                                        white_rooks, black_rooks, white_queens, black_queens, white_king_bb, black_king_bb,
                                        white_pawn_attacks, black_pawn_attacks, white_knight_attacks, black_knight_attacks, white_bishop_attacks, black_bishop_attacks,
                                        white_rook_attacks, black_rook_attacks, white_queen_attacks, black_queen_attacks,
                                        black_king_half, white_king_half);

        score -= self.genetic_eval.eval(mirror(black_pawns), mirror(white_pawns), mirror(black_knights), mirror(white_knights), mirror(black_bishops), mirror(white_bishops),
                                        mirror(black_rooks), mirror(white_rooks), mirror(black_queens), mirror(white_queens), mirror(black_king_bb), mirror(white_king_bb),
                                        mirror(black_pawn_attacks), mirror(white_pawn_attacks), mirror(black_knight_attacks), mirror(white_knight_attacks), mirror(black_bishop_attacks),
                                        mirror(white_bishop_attacks), mirror(black_rook_attacks), mirror(white_rook_attacks), mirror(black_queen_attacks), mirror(white_queen_attacks),
                                        mirror(white_king_half), mirror(black_king_half));

        score as i32
    }
}

fn king_half_bb(king_pos: i32) -> u64 {
    if king_pos & 7 <= 3 {
        0b00001111_00001111_00001111_00001111_00001111_00001111_00001111_00001111
    } else {
        0b11110000_11110000_11110000_11110000_11110000_11110000_11110000_11110000
    }
}