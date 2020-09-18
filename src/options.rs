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

// Auto-generated file (see tools/tuning/gencode.py)

use std::sync::mpsc::Sender;
use crate::engine::Message;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces::{B, K, N, P, PIECE_VALUES, Q, R};
use crate::score_util::{pack_scores};

const CASTLING_BONUS: i32 = 28;
const LOST_QUEENSIDE_CASTLING_PENALTY: i32 = 24;
const LOST_KINGSIDE_CASTLING_PENALTY: i32 = 51;
const DOUBLED_PAWN_PENALTY: i32 = 18;
const KING_SHIELD_BONUS: i32 = 12;
const PAWN_COVER_BONUS: i32 = 12;
const EG_PASSED_PAWN_BONUS: [i32; 4] = [186, 149, 71, 21];
const PASSED_PAWN_BONUS: [i32; 4] = [32, 18, 8, 3];
const PASSED_PAWN_KING_DEFENSE_BONUS: [i32; 8] = [0, 69, 50, 21, 10, 6, 7, 2];
const PASSED_PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [0, 100, 69, 31, 9, 0, 0, 1];
const KING_DANGER_PIECE_PENALTY: [i32; 16] = [0, -4, -3, 9, 27, 46, 79, 116, 1914, 1773, 1773, 1773, 1773, 1773, 1773, 1773];
const EG_KNIGHT_MOB_BONUS: [i32; 9] = [-121, -75, -56, -42, -43, -25, -26, -27, -54];
const EG_BISHOP_MOB_BONUS: [i32; 14] = [-71, -35, -25, -17, -11, -8, -3, -8, -7, -8, -18, -33, 3, -30];
const EG_ROOK_MOB_BONUS: [i32; 15] = [-85, -46, -8, -2, 12, 24, 28, 29, 33, 32, 30, 41, 49, 49, 44];
const EG_QUEEN_MOB_BONUS: [i32; 28] = [-129, -38, -42, -23, -31, -20, 0, 15, 35, 38, 65, 77, 67, 71, 76, 82, 50, 56, 34, 71, 91, 115, 20, 94, -2, 62, 117, 108];
const KNIGHT_MOB_BONUS: [i32; 9] = [-35, -22, -13, -11, -1, 0, 5, 7, 46];
const BISHOP_MOB_BONUS: [i32; 14] = [-9, 2, 13, 18, 21, 27, 30, 32, 39, 35, 48, 113, 35, 83];
const ROOK_MOB_BONUS: [i32; 15] = [-24, -16, -14, -6, -4, 3, 10, 20, 28, 41, 66, 57, 54, 58, 41];
const QUEEN_MOB_BONUS: [i32; 28] = [-12, -17, -12, -9, -1, 3, 3, 6, 8, 14, 13, 11, 31, 29, 44, 52, 114, 95, 160, 126, 92, 50, 262, 130, 359, 159, 111, 8];
const EG_PAWN_PST: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 74, 70, 79, 27, 63, 19, 73, 104, 43, 50, 28, 9, -9, -5, 31, 37, 23, 8, 0, -28, -20, -9, 10, 8, 7, 4, -16, -27, -26, -17, -3, -4, -9, 0, -13, -3, -1, -5, -12, -15, 3, -2, 5, 3, 15, -1, -7, -15, 0, 0, 0, 0, 0, 0, 0, 0];
const PAWN_PST: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 122, 113, 25, 120, 88, 167, 67, 4, 6, -1, 28, 19, 70, 91, 38, -3, -21, 3, -6, 24, 19, 15, 2, -18, -25, -14, -6, 7, 16, 2, -6, -31, -24, -17, -15, -23, -14, -12, 12, -19, -21, 1, -16, -15, -12, 19, 31, -16, 0, 0, 0, 0, 0, 0, 0, 0];
const EG_KNIGHT_PST: [i32; 64] = [3, -33, -6, -19, -41, -3, -75, -53, -1, 5, -64, -6, -12, -39, -23, -37, -18, -36, 7, 2, -20, -27, -32, -48, -10, -6, 19, 10, 16, -2, 5, -16, -5, -21, 10, 17, 9, 12, 2, -6, -19, -8, -17, 8, -2, -9, -32, -6, -50, -32, -22, -21, -11, -31, -16, -44, -19, -42, -19, -3, -17, -12, -23, -61];
const KNIGHT_PST: [i32; 64] = [-198, -74, -21, -48, 67, -130, -3, -143, -101, -62, 84, 6, -3, 42, -5, -45, -49, 31, -4, 17, 62, 99, 47, 38, -10, 5, -8, 41, 19, 56, 9, 24, -10, 6, 6, 2, 19, 12, 11, -5, -25, -14, 6, 4, 17, 10, 19, -19, -5, -13, 0, 22, 22, 28, 1, 7, -81, 0, -26, -8, 17, -3, -1, 2];
const EG_BISHOP_PST: [i32; 64] = [6, -27, -4, -7, -7, -13, -19, -18, 12, -9, 3, -12, -4, -25, -13, 4, 8, -16, -11, -2, -13, -11, -2, 6, -3, 0, 9, 1, 15, 3, -7, 3, -15, -7, 9, 17, -5, 5, -10, -13, -17, -12, -4, 0, 13, -12, -6, -20, -45, -31, -26, -8, -6, -29, -25, -51, -39, -37, -9, -12, -19, -9, -22, -28];
const BISHOP_PST: [i32; 64] = [-56, -7, -78, -53, -29, -31, -15, -13, -63, 0, -34, -39, 3, 44, 16, -76, -35, 22, 30, 5, 29, 52, 20, -11, -9, 10, 5, 46, 21, 26, 12, -6, 7, 14, 14, 21, 38, 8, 8, 13, 9, 26, 29, 26, 19, 46, 19, 20, 45, 35, 37, 17, 31, 48, 53, 27, 0, 44, 14, 13, 24, 11, 2, 3];
const EG_ROOK_PST: [i32; 64] = [29, 31, 36, 35, 40, 34, 33, 27, 26, 24, 14, 20, 2, 15, 33, 22, 24, 19, 16, 20, 21, 14, 15, 13, 29, 18, 28, 16, 17, 17, 21, 23, 28, 25, 26, 26, 17, 21, 11, 13, 25, 20, 15, 18, 11, 14, 17, 7, 14, 13, 14, 13, 10, 7, 9, 23, 10, 7, 6, 4, -2, 3, 14, -8];
const ROOK_PST: [i32; 64] = [13, 3, -1, 11, 2, -12, -9, 5, 15, 22, 63, 51, 74, 58, 10, 23, 3, 34, 39, 38, 7, 44, 44, 12, -16, 10, 21, 28, 26, 45, 0, -8, -28, -6, 6, -2, 6, -10, 17, -10, -44, -14, -9, -11, 0, -2, 0, -21, -37, -11, -5, 5, 4, 12, -1, -53, -18, 3, 17, 23, 26, 15, -11, -30];
const EG_QUEEN_PST: [i32; 64] = [32, 58, 24, 48, 33, 6, 12, 63, 35, 69, 58, 67, 82, 41, 65, 52, -2, 28, 28, 89, 86, 46, 64, 70, 55, 64, 54, 80, 107, 78, 141, 117, 13, 71, 46, 91, 73, 75, 114, 100, 35, 3, 46, 25, 48, 59, 78, 69, 21, 12, -1, 1, -1, 2, -9, -1, -19, -13, -7, -5, 20, 6, 13, 0];
const QUEEN_PST: [i32; 64] = [-37, 1, 59, 27, 95, 121, 94, 62, -42, -64, -4, 15, 1, 83, 45, 61, 0, -7, 11, -2, 32, 103, 57, 58, -34, -31, -19, -27, -12, 16, -16, -9, -3, -34, -6, -20, -13, -6, -9, -6, -12, 9, -10, 3, -9, 1, 7, 8, -22, -4, 19, 16, 28, 32, 7, 26, 21, 10, 15, 28, 10, -1, -5, -50];
const EG_KING_PST: [i32; 64] = [-94, -61, -30, -35, -11, 23, 2, -16, -25, -21, -12, -3, -10, 26, 18, 3, 0, -22, -8, -26, -29, 8, 2, 2, -18, -1, 9, -14, -18, 20, 15, 2, -5, -29, 10, 13, 20, 20, 7, 1, -18, 1, 7, 16, 20, 16, 9, -2, -24, -7, 5, 27, 20, 6, -1, -8, -60, -37, -26, -1, -15, -4, -36, -54];
const KING_PST: [i32; 64] = [-8, 133, 56, 38, 7, -19, 22, 9, 58, 128, 106, 92, 128, 27, 2, 12, 18, 158, 122, 163, 202, 170, 185, 1, -1, 65, 53, 148, 167, 44, 35, -24, -69, 83, 44, 40, 40, 35, 9, -57, -6, -1, 13, 4, 7, 3, 3, -31, -14, -1, -2, -69, -32, -8, 1, -12, -27, 27, 21, -55, -48, -37, 25, 6];
    
pub struct Options {}
    
impl Options {
    pub fn new() -> Self {
        Options{}
    }
    
    pub fn set_option(&mut self, name: String, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    
    pub fn set_array_option(&mut self, name: String, _: usize, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    

    #[inline]
    pub fn get_castling_bonus(&self) -> i32 {
        CASTLING_BONUS
    }


    #[inline]
    pub fn get_lost_queenside_castling_penalty(&self) -> i32 {
        LOST_QUEENSIDE_CASTLING_PENALTY
    }


    #[inline]
    pub fn get_lost_kingside_castling_penalty(&self) -> i32 {
        LOST_KINGSIDE_CASTLING_PENALTY
    }


    #[inline]
    pub fn get_doubled_pawn_penalty(&self) -> i32 {
        DOUBLED_PAWN_PENALTY
    }


    #[inline]
    pub fn get_king_shield_bonus(&self) -> i32 {
        KING_SHIELD_BONUS
    }


    #[inline]
    pub fn get_pawn_cover_bonus(&self) -> i32 {
        PAWN_COVER_BONUS
    }

    #[inline]
    pub fn get_eg_passed_pawn_bonus(&self, index: usize) -> i32 {
        EG_PASSED_PAWN_BONUS[index]
    }
    
    #[inline]
    pub fn get_passed_pawn_bonus(&self, index: usize) -> i32 {
        PASSED_PAWN_BONUS[index]
    }
    
    #[inline]
    pub fn get_passed_pawn_king_defense_bonus(&self, index: usize) -> i32 {
        PASSED_PAWN_KING_DEFENSE_BONUS[index]
    }
    
    #[inline]
    pub fn get_passed_pawn_king_attacked_penalty(&self, index: usize) -> i32 {
        PASSED_PAWN_KING_ATTACKED_PENALTY[index]
    }
    
    #[inline]
    pub fn get_king_danger_piece_penalty(&self, index: usize) -> i32 {
        KING_DANGER_PIECE_PENALTY[index]
    }
    
    #[inline]
    pub fn get_eg_knight_mob_bonus(&self, index: usize) -> i32 {
        EG_KNIGHT_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_eg_bishop_mob_bonus(&self, index: usize) -> i32 {
        EG_BISHOP_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_eg_rook_mob_bonus(&self, index: usize) -> i32 {
        EG_ROOK_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_eg_queen_mob_bonus(&self, index: usize) -> i32 {
        EG_QUEEN_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_knight_mob_bonus(&self, index: usize) -> i32 {
        KNIGHT_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_bishop_mob_bonus(&self, index: usize) -> i32 {
        BISHOP_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_rook_mob_bonus(&self, index: usize) -> i32 {
        ROOK_MOB_BONUS[index]
    }
    
    #[inline]
    pub fn get_queen_mob_bonus(&self, index: usize) -> i32 {
        QUEEN_MOB_BONUS[index]
    }
    
    #[inline]
    pub const fn get_eg_pawn_pst(&self) -> [i32; 64] {
        EG_PAWN_PST
    }
            
    #[inline]
    pub const fn get_pawn_pst(&self) -> [i32; 64] {
        PAWN_PST
    }
            
    #[inline]
    pub const fn get_eg_knight_pst(&self) -> [i32; 64] {
        EG_KNIGHT_PST
    }
            
    #[inline]
    pub const fn get_knight_pst(&self) -> [i32; 64] {
        KNIGHT_PST
    }
            
    #[inline]
    pub const fn get_eg_bishop_pst(&self) -> [i32; 64] {
        EG_BISHOP_PST
    }
            
    #[inline]
    pub const fn get_bishop_pst(&self) -> [i32; 64] {
        BISHOP_PST
    }
            
    #[inline]
    pub const fn get_eg_rook_pst(&self) -> [i32; 64] {
        EG_ROOK_PST
    }
            
    #[inline]
    pub const fn get_rook_pst(&self) -> [i32; 64] {
        ROOK_PST
    }
            
    #[inline]
    pub const fn get_eg_queen_pst(&self) -> [i32; 64] {
        EG_QUEEN_PST
    }
            
    #[inline]
    pub const fn get_queen_pst(&self) -> [i32; 64] {
        QUEEN_PST
    }
            
    #[inline]
    pub const fn get_eg_king_pst(&self) -> [i32; 64] {
        EG_KING_PST
    }
            
    #[inline]
    pub const fn get_king_pst(&self) -> [i32; 64] {
        KING_PST
    }
            
}

pub fn parse_set_option(_: &Sender<Message>, _: &str, _: &str) {}
    
const SCORES: [u32; 64 * 13] = calc_scores();

pub struct PieceSquareTables {
}

impl PieceSquareTables {
    pub fn new(_: &Options) -> Self {
        PieceSquareTables {}
    }

    #[inline]
    pub fn get_packed_score(&self, piece: i8, pos: usize) -> u32 {
        unsafe { *SCORES.get_unchecked((piece + 6) as usize * 64 + pos) }
    }

    pub fn recalculate(&mut self, _: &Options) {}
}

const fn calc_scores() -> [u32; 64 * 13] {
    concat(
        combine(BLACK, P, mirror(PAWN_PST), mirror(EG_PAWN_PST)),
        combine(BLACK, N, mirror(KNIGHT_PST), mirror(EG_KNIGHT_PST)),
        combine(BLACK, B, mirror(BISHOP_PST), mirror(EG_BISHOP_PST)),
        combine(BLACK, R, mirror(ROOK_PST), mirror(EG_ROOK_PST)),
        combine(BLACK, Q, mirror(QUEEN_PST), mirror(EG_QUEEN_PST)),
        combine(BLACK, K, mirror(KING_PST), mirror(EG_KING_PST)),
        combine(WHITE, P, PAWN_PST, EG_PAWN_PST),
        combine(WHITE, N, KNIGHT_PST, EG_KNIGHT_PST),
        combine(WHITE, B, BISHOP_PST, EG_BISHOP_PST),
        combine(WHITE, R, ROOK_PST, EG_ROOK_PST),
        combine(WHITE, Q, QUEEN_PST, EG_QUEEN_PST),
        combine(WHITE, K, KING_PST, EG_KING_PST)
    )
}

const fn concat(
    black_pawns: [u32; 64],
    black_knights: [u32; 64],
    black_bishops: [u32; 64],
    black_rooks: [u32; 64],
    black_queens: [u32; 64],
    black_kings: [u32; 64],
    white_pawns: [u32; 64],
    white_knights: [u32; 64],
    white_bishops: [u32; 64],
    white_rooks: [u32; 64],
    white_queens: [u32; 64],
    white_kings: [u32; 64],
) -> [u32; 64 * 13] {
    let mut all: [u32; 64 * 13] = [0; 64 * 13];

    let mut i = 0;
    while i < 64 {
        all[i] = black_kings[i];
        all[i + 1 * 64] = black_queens[i];
        all[i + 2 * 64] = black_rooks[i];
        all[i + 3 * 64] = black_bishops[i];
        all[i + 4 * 64] = black_knights[i];
        all[i + 5 * 64] = black_pawns[i];

        all[i + 7 * 64] = white_pawns[i];
        all[i + 8 * 64] = white_knights[i];
        all[i + 9 * 64] = white_bishops[i];
        all[i + 10 * 64] = white_rooks[i];
        all[i + 11 * 64] = white_queens[i];
        all[i + 12 * 64] = white_kings[i];
        i += 1;
    }

    all
}

const fn combine(color: Color, piece: i8, scores: [i32; 64], eg_scores: [i32; 64]) -> [u32; 64] {
    let mut combined_scores: [u32; 64] = [0; 64];
    let piece_value = PIECE_VALUES[piece as usize];

    let mut i = 0;
    while i < 64 {
        let score = (scores[i] as i16 + piece_value) * (color as i16);
        let eg_score = (eg_scores[i] as i16 + piece_value) * (color as i16);
        combined_scores[i] = pack_scores(score, eg_score);

        i += 1;
    }

    combined_scores
}

const fn mirror(scores: [i32; 64]) -> [i32; 64] {
    let mut output: [i32; 64] = clone(scores);

    let mut col = 0;
    while col < 8 {

        let mut row = 0;
        while row < 4 {
            let opposite_row = 7 - row;

            let pos = col + row * 8;
            let opposite_pos = col + opposite_row * 8;

            let tmp = output[pos];
            output[pos] = output[opposite_pos];
            output[opposite_pos] = tmp;


            row += 1;
        }

        col += 1;
    }

    output
}

const fn clone(input: [i32; 64]) -> [i32; 64] {
    let mut output: [i32; 64] = [0; 64];

    let mut i = 0;
    while i < 64 {
        output[i] = input[i];
        i += 1;
    }

    output
}

