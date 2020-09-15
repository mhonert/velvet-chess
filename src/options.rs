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
const LOST_KINGSIDE_CASTLING_PENALTY: i32 = 49;
const DOUBLED_PAWN_PENALTY: i32 = 18;
const KING_SHIELD_BONUS: i32 = 12;
const PAWN_COVER_BONUS: i32 = 12;
const EG_PASSED_PAWN_BONUS: [i32; 4] = [180, 147, 67, 21];
const PASSED_PAWN_BONUS: [i32; 4] = [90, 16, 7, 3];
const PASSED_PAWN_KING_DEFENSE_BONUS: [i32; 8] = [0, 69, 49, 21, 10, 6, 5, 0];
const PASSED_PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [0, 97, 69, 31, 9, 0, 0, 4];
const KING_DANGER_PIECE_PENALTY: [i32; 16] = [0, -4, -3, 9, 28, 47, 80, 117, 1773, 1773, 1773, 1773, 1773, 1773, 1773, 1773];
const EG_KNIGHT_MOB_BONUS: [i32; 9] = [-92, -73, -56, -42, -43, -25, -24, -27, -53];
const EG_BISHOP_MOB_BONUS: [i32; 14] = [-44, -33, -23, -17, -11, -5, -3, -7, -7, -8, -18, -32, 2, -29];
const EG_ROOK_MOB_BONUS: [i32; 15] = [-83, -45, -25, -7, 9, 19, 22, 25, 32, 30, 27, 39, 46, 49, 45];
const EG_QUEEN_MOB_BONUS: [i32; 28] = [-23, -38, -40, -24, -32, -21, -21, 7, 16, 23, 39, 45, 51, 52, 59, 70, 52, 56, 47, 71, 84, 93, 54, 93, 51, 70, 115, 82];
const KNIGHT_MOB_BONUS: [i32; 9] = [-35, -21, -13, -12, 0, -3, 1, 6, 47];
const BISHOP_MOB_BONUS: [i32; 14] = [-12, 1, 11, 18, 21, 24, 30, 31, 39, 29, 47, 111, 32, 81];
const ROOK_MOB_BONUS: [i32; 15] = [-18, -13, -11, -7, -8, 1, 8, 19, 21, 38, 67, 57, 48, 43, 26];
const QUEEN_MOB_BONUS: [i32; 28] = [-23, -20, -13, -12, -1, 1, 6, 1, 4, 10, 8, 10, 20, 27, 44, 37, 117, 95, 164, 128, 91, 46, 270, 115, 364, 163, 13, 3];
const EG_PAWN_PST: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 68, 62, 76, 19, 50, 16, 66, 97, 43, 53, 27, 8, -7, -3, 34, 37, 22, 8, 0, -27, -20, -9, 10, 6, 6, 3, -16, -27, -25, -17, -3, -4, -9, 0, -16, -3, -1, -6, -11, -15, -3, -2, 4, -1, 9, 2, -2, -19, 0, 0, 0, 0, 0, 0, 0, 0];
const PAWN_PST: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 114, 113, 5, 121, 89, 156, 67, 4, 4, -9, 19, 19, 70, 92, 36, -8, -21, 2, -7, 29, 25, 13, 1, -18, -27, -15, -5, 11, 20, 4, -7, -33, -21, -17, -12, -26, -16, -10, 13, -20, -17, 3, -21, -17, -14, 18, 30, -12, 0, 0, 0, 0, 0, 0, 0, 0];
const EG_KNIGHT_PST: [i32; 64] = [-7, -37, -11, -20, -45, -2, -75, -50, -2, 4, -64, -6, -14, -38, -25, -38, -22, -35, 1, 3, -19, -24, -31, -48, -11, -7, 12, 6, 16, 1, 8, -16, -6, -21, 6, 17, 10, 11, 0, -8, -13, -4, -16, 6, -2, -9, -24, -3, -52, -33, -23, -19, -11, -29, -17, -41, -23, -42, -21, -1, -17, -14, -24, -59];
const KNIGHT_PST: [i32; 64] = [-200, -70, -22, -48, 68, -143, 0, -159, -105, -62, 89, 5, -4, 41, -6, -43, -49, 29, 1, 16, 60, 86, 45, 39, -11, 0, -8, 45, 16, 51, 2, 25, -15, 6, 4, -4, 14, 7, 10, -6, -36, -21, 5, 1, 14, 8, 17, -28, 2, -10, 0, 24, 22, 30, 2, 14, -82, 3, -26, -13, 18, -4, -3, 1];
const EG_BISHOP_PST: [i32; 64] = [4, -30, -8, -13, -9, -20, -21, -21, 9, -11, 1, -15, -5, -25, -12, 6, 7, -17, -11, -3, -14, -9, 2, 5, -4, 2, 6, 3, 14, 1, -3, -1, -16, -7, 9, 16, -4, 6, -10, -14, -16, -12, -4, 6, 13, -9, -2, -19, -43, -27, -24, -8, -5, -26, -20, -54, -40, -34, -8, -12, -20, -2, -24, -30];
const BISHOP_PST: [i32; 64] = [-67, -9, -77, -43, -31, -23, -14, -16, -67, -1, -34, -37, 2, 43, 14, -98, -44, 22, 29, 4, 30, 52, 12, -17, -10, 1, 5, 46, 18, 24, 6, -8, 6, 12, 8, 21, 38, 5, 4, 10, 6, 26, 26, 18, 11, 44, 13, 13, 45, 38, 37, 15, 29, 49, 53, 32, 0, 42, 11, 10, 24, 2, 4, 1];
const EG_ROOK_PST: [i32; 64] = [28, 28, 34, 33, 37, 30, 29, 24, 25, 24, 14, 20, 2, 18, 26, 23, 21, 17, 13, 19, 18, 12, 13, 12, 24, 14, 24, 15, 18, 14, 18, 20, 23, 21, 22, 22, 16, 18, 12, 11, 19, 11, 9, 13, 10, 11, 15, 7, 10, 9, 8, 11, 7, 7, 8, 22, -4, 7, 4, 4, -1, 3, 19, -19];
const ROOK_PST: [i32; 64] = [14, 5, -4, 11, 1, -10, -13, 4, 15, 20, 61, 47, 74, 41, 10, 13, 4, 36, 38, 33, 6, 43, 40, 12, -11, 10, 20, 27, 20, 46, 1, -8, -27, -6, 8, -2, 5, -11, 11, -13, -46, -6, -6, -11, 1, -4, -5, -30, -41, -10, -5, 4, 3, 9, -2, -66, -16, 1, 18, 22, 27, 18, -27, -26];
const EG_QUEEN_PST: [i32; 64] = [20, 45, 31, 43, 48, 34, 39, 63, 16, 44, 39, 57, 68, 52, 63, 59, -5, 18, 24, 69, 84, 63, 57, 69, 44, 45, 41, 67, 90, 68, 117, 97, 8, 56, 35, 72, 54, 59, 87, 84, 22, 0, 34, 22, 30, 47, 59, 57, 14, 3, 0, 3, 3, 4, -9, 2, -14, -12, -5, 2, 14, -5, 6, -2];
const QUEEN_PST: [i32; 64] = [-38, 0, 61, 31, 96, 122, 93, 63, -45, -86, -4, 8, -13, 83, 47, 61, -2, -8, 11, -7, 18, 106, 71, 59, -57, -45, -28, -48, -28, 5, -45, -27, -8, -55, -17, -40, -18, -11, -26, -20, -13, 8, -14, -2, -7, -6, 0, -2, -29, -6, 20, 15, 30, 33, 6, 25, 23, 13, 17, 29, 11, -2, -6, -73];
const EG_KING_PST: [i32; 64] = [-93, -31, -26, -28, -12, 20, 5, -15, -17, -15, -8, 2, -6, 26, 16, 1, 0, -18, 6, -22, -23, 9, 6, 1, -19, 0, 9, -12, -16, 20, 15, -4, -9, -27, 10, 13, 20, 20, 7, -4, -17, 1, 6, 15, 20, 16, 8, -2, -25, -5, 5, 26, 17, 4, -1, -8, -61, -31, -26, -3, -16, -6, -31, -51];
const KING_PST: [i32; 64] = [-2, 5, 57, 15, 7, -17, -14, -2, 24, 125, 104, 76, 127, 24, 1, 7, 18, 153, 66, 163, 186, 170, 183, 0, -2, 65, 52, 149, 168, 42, 34, -6, -59, 82, 41, 38, 33, 36, 7, -41, -5, -2, 12, 5, 8, 2, 2, -32, -14, -3, -3, -71, -32, -8, 0, -13, -23, 26, 24, -57, -48, -39, 27, 8];
    
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

