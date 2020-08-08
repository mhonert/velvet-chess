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

use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, EG_PIECE_VALUES, K, N, P, PIECE_VALUES, Q, R};
use crate::score_util::{pack_scores};

pub struct PieceSquareTables {
    white_scores: [u32; 64 * 7],
    black_scores: [u32; 64 * 7],
}

impl PieceSquareTables {
    pub fn new() -> Self {
        let white_pawns = combine(WHITE, P, PAWN_SCORES, EG_PAWN_SCORES);
        let black_pawns = combine(BLACK, P, mirror(PAWN_SCORES), mirror(EG_PAWN_SCORES));

        let white_knights = combine(WHITE, N, KNIGHT_SCORES, EG_KNIGHT_SCORES);
        let black_knights = combine(BLACK, N, mirror(KNIGHT_SCORES), mirror(EG_KNIGHT_SCORES));

        let white_bishops = combine(WHITE, B, BISHOP_SCORES, EG_BISHOP_SCORES);
        let black_bishops = combine(BLACK, B, mirror(BISHOP_SCORES), mirror(EG_BISHOP_SCORES));

        let white_rooks = combine(WHITE, R, ROOK_SCORES, EG_ROOK_SCORES);
        let black_rooks = combine(BLACK, R, mirror(ROOK_SCORES), mirror(EG_ROOK_SCORES));

        let white_queens = combine(WHITE, Q, QUEEN_SCORES, EG_QUEEN_SCORES);
        let black_queens = combine(BLACK, Q, mirror(QUEEN_SCORES), mirror(EG_QUEEN_SCORES));

        let white_kings = combine(WHITE, K, KING_SCORES, EG_KING_SCORES);
        let black_kings = combine(BLACK, K, mirror(KING_SCORES), mirror(EG_KING_SCORES));

        let white_scores: [u32; 64 * 7] = concat(
            white_pawns,
            white_knights,
            white_bishops,
            white_rooks,
            white_queens,
            white_kings,
        );
        let black_scores: [u32; 64 * 7] = concat(
            black_pawns,
            black_knights,
            black_bishops,
            black_rooks,
            black_queens,
            black_kings,
        );

        PieceSquareTables {
            white_scores,
            black_scores,
        }
    }

    pub fn get_packed_score(&self, piece: i8, pos: usize) -> u32 {
        if piece < 0 {
            return self.black_scores[-piece as usize * 64 + pos];
        }

        self.white_scores[piece as usize * 64 + pos as usize]
    }
}

fn concat(
    pawns: [u32; 64],
    knights: [u32; 64],
    bishops: [u32; 64],
    rooks: [u32; 64],
    queens: [u32; 64],
    kings: [u32; 64],
) -> [u32; 64 * 7] {
    let mut all: [u32; 64 * 7] = [0; 64 * 7];

    copy(pawns, &mut all, 64);
    copy(knights, &mut all, 64 * 2);
    copy(bishops, &mut all, 64 * 3);
    copy(rooks, &mut all, 64 * 4);
    copy(queens, &mut all, 64 * 5);
    copy(kings, &mut all, 64 * 6);

    all
}

fn copy(source: [u32; 64], target: &mut [u32; 64 * 7], start: usize) {
    for i in 0..64 {
        target[i + start] = source[i]
    }
}

fn combine(color: Color, piece: i8, scores: [i16; 64], eg_scores: [i16; 64]) -> [u32; 64] {
    let mut combined_scores: [u32; 64] = [0; 64];
    let piece_value = PIECE_VALUES[piece as usize];
    let eg_piece_value = EG_PIECE_VALUES[piece as usize];

    for i in 0..64 {
        let score = (scores[i] + piece_value) * (color as i16);
        let eg_score = (eg_scores[i] + eg_piece_value) * (color as i16);
        combined_scores[i] = pack_scores(score, eg_score);
    }

    combined_scores
}

fn mirror(scores: [i16; 64]) -> [i16; 64] {
    let mut output: [i16; 64] = scores.clone();

    for col in 0..8 {
        for row in 0..4 {
            let opposite_row = 7 - row;

            let pos = col + row * 8;
            let opposite_pos = col + opposite_row * 8;

            output.swap(pos, opposite_pos);
        }
    }

    output
}

const PAWN_SCORES: [i16; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 98, 94, 33, 98, 70, 123, 50, 2, -6, -13, 23, 22, 69, 87, 33, -9, -19,
    13, 3, 29, 31, 23, 17, -17, -22, -9, -3, 18, 24, 10, 9, -28, -25, -10, -12, -22, -10, -17, 24,
    -24, -24, 0, -12, -19, -15, 18, 34, -18, 0, 0, 0, 0, 0, 0, 0, 0,
];

const EG_PAWN_SCORES: [i16; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 134, 127, 110, 68, 92, 66, 120, 145, 81, 92, 56, 28, 6, 15, 60, 64, 26,
    7, -6, -30, -30, -15, 3, 7, 5, 3, -16, -35, -32, -23, -8, -9, -1, 3, -9, 0, -1, 0, -14, -13,
    14, 11, 10, 12, 24, 3, 2, -12, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_SCORES: [i16; 64] = [
    -178, -109, -69, -41, 99, -101, -14, -118, -116, -57, 89, 31, -4, 60, 9, -40, -67, 31, 10, 31,
    101, 121, 46, 42, -21, 7, 1, 48, 22, 65, 14, 31, -5, 7, 7, 12, 24, 17, 24, -2, -28, -9, 8, 11,
    27, 12, 24, -23, -10, -38, 4, 22, 18, 33, 3, 8, -98, -4, -41, -24, 12, -3, -7, 5,
];

const EG_KNIGHT_SCORES: [i16; 64] = [
    17, 14, 35, 2, -27, 8, -42, -44, 38, 32, -27, 16, 15, -10, 1, -10, 18, 0, 29, 28, -9, -6, -1,
    -16, 30, 29, 42, 36, 47, 27, 34, 2, 8, 11, 38, 35, 33, 34, 26, 16, 16, 15, 10, 32, 21, 15, -3,
    26, -20, 16, 8, 2, 26, -6, 3, -17, 28, -9, 22, 38, 14, 15, 9, -35,
];

const BISHOP_SCORES: [i16; 64] = [
    -54, -14, -99, -66, -13, -24, -34, -13, -58, 1, -26, -30, 44, 78, 27, -68, -33, 13, 46, 28, 37,
    57, 28, -2, -23, 17, 26, 63, 39, 40, 19, 1, 4, 28, 17, 40, 61, 15, 19, 18, 7, 38, 29, 27, 23,
    53, 29, 10, 35, 31, 39, 21, 33, 44, 51, 16, -13, 20, 13, -3, 18, 11, -10, -12,
];

const EG_BISHOP_SCORES: [i16; 64] = [
    -12, -46, -17, -25, -29, -36, -32, -33, -12, -30, -21, -38, -46, -57, -40, -18, -10, -33, -44,
    -38, -41, -34, -31, -18, -13, -29, -26, -35, -28, -25, -34, -23, -39, -36, -16, -22, -45, -26,
    -42, -34, -35, -39, -28, -27, -16, -43, -36, -30, -57, -43, -47, -35, -30, -48, -39, -55, -44,
    -36, -29, -19, -33, -31, -36, -36,
];

const ROOK_SCORES: [i16; 64] = [
    23, 20, 31, 72, 69, 8, 5, 5, 18, 40, 86, 92, 82, 97, 53, 47, -15, 39, 42, 43, 23, 73, 85, 22,
    -13, -7, 21, 30, 16, 34, 14, 1, -33, -15, 6, 15, 18, 1, 46, -5, -43, -11, -4, -11, 1, 2, 17,
    -11, -39, -9, -7, 7, 7, 10, 13, -46, -21, -1, 18, 18, 18, 3, -1, -8,
];

const EG_ROOK_SCORES: [i16; 64] = [
    18, 18, 14, 3, 9, 17, 18, 15, 18, 13, 1, -2, -6, -2, 8, 8, 21, 11, 7, 11, 9, 0, 0, 5, 16, 16,
    20, 14, 15, 19, 8, 12, 21, 17, 17, 8, 6, 6, -9, 2, 13, 11, 4, 8, 6, 3, -2, -8, 8, 4, 8, 6, 2,
    1, -5, 8, 5, 5, 2, 5, 4, 15, 1, -26,
];

const QUEEN_SCORES: [i16; 64] = [
    -25, 3, 47, 48, 62, 44, 40, 38, -31, -49, -3, 20, -23, 53, 7, 19, 8, -2, 16, -3, 12, 40, 1, 23,
    -39, -33, -33, -27, -21, 7, -24, -9, 0, -32, -10, -16, -3, 3, -5, 4, -11, 10, -10, 0, -5, 2,
    16, 16, -25, -9, 22, 13, 25, 34, 14, 32, 4, 2, 11, 28, 2, -18, -23, -52,
];

const EG_QUEEN_SCORES: [i16; 64] = [
    44, 77, 50, 44, 56, 51, 58, 88, 45, 75, 60, 73, 73, 36, 81, 81, 21, 52, 35, 98, 86, 64, 89, 83,
    84, 89, 80, 96, 124, 91, 147, 132, 32, 89, 68, 101, 78, 84, 123, 104, 58, 26, 67, 53, 65, 78,
    86, 85, 40, 37, 18, 24, 25, 14, 10, 11, 29, 18, 20, 7, 51, 38, 58, 34,
];

const KING_SCORES: [i16; 64] = [
    -6, 293, 247, 145, 35, -28, 41, 55, 159, 114, 203, 245, 153, 108, -49, 16, 69, 223, 180, 232,
    238, 227, 225, 16, 46, 120, 136, 155, 180, 102, 54, -47, -119, 75, 66, 56, 78, 38, 5, -83, -19,
    -3, 11, -21, -19, 8, -10, -52, -2, 9, -31, -97, -76, -40, 2, 12, -27, 20, 8, -69, -54, -55, 36,
    30,
];

const EG_KING_SCORES: [i16; 64] = [
    -87, -79, -54, -42, -11, 33, 14, -13, -28, 6, -9, -17, 4, 33, 43, 13, 10, -4, 8, -11, -9, 30,
    30, 16, -10, 11, 18, 17, 11, 35, 36, 25, 11, -7, 28, 33, 33, 38, 24, 14, -12, 9, 23, 42, 47,
    33, 23, 9, -31, -10, 24, 47, 48, 26, 3, -23, -66, -47, -23, 3, -7, 5, -47, -77,
];
