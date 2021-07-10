/*
 * Velvet Chess Engine
 * Copyright (C) 2021 mhonert (https://github.com/mhonert)
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
use crate::pieces::{B, K, N, P, Q, R, get_piece_value};
use crate::score_util::{pack_scores};

const QS_SEE_THRESHOLD: i32 = 104;
const QS_PRUNE_MARGIN: i32 = 650;
const TIMEEXT_SCORE_DROP_THRESHOLD: i32 = 20;
const TIMEEXT_HISTORY_SIZE: i32 = 6;
const RAZOR_MARGIN: i32 = 130;
const KING_SHIELD_BONUS: i32 = 23;
const PASSED_PAWN_BONUS: [i32; 4] = [83, 99, 63, 26];
const PAWN_KING_DEFENSE_BONUS: [i32; 8] = [1, 24, 18, 12, 7, 3, 1, 0];
const PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [1, 24, 18, 12, 8, 3, 1, 0];
const PASSED_PAWN_KING_DEFENSE_BONUS: [i32; 8] = [3, 54, 37, 15, 2, 0, 12, 6];
const PASSED_PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [4, 63, 42, 21, 12, 1, 2, 9];
const HALF_OPEN_FILE_BONUS: [i32; 4] = [0, 23, 22, 1];
const EG_HALF_OPEN_FILE_BONUS: [i32; 4] = [57, 20, 10, 6];
const KING_THREAT_BY_PIECE_COMBO: [i32; 128] = [7, 5, 8, 9, 11, 0, 5, 0, 4, 3, 9, 20, 10, 10, 14, 30, 0, 0, 3, 8, 25, 34, 19, 38, 6, 6, 20, 41, 32, 34, 33, 54, 7, 1, 17, 41, 13, 22, 21, 56, 16, 27, 34, 67, 30, 46, 49, 123, 2, 0, 29, 93, 36, 69, 51, 111, 44, 81, 65, 153, 62, 74, 71, 135, 0, 0, 3, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 26, 62, 80, 43, 51, 79, 153, 50, 71, 94, 115, 83, 94, 126, 221, 0, 0, 77, 163, 83, 68, 168, 122, 91, 131, 142, 218, 131, 138, 108, 443];
const EG_KNIGHT_MOB_BONUS: [i32; 9] = [-121, -80, -55, -47, -41, -33, -36, -40, -48];
const EG_BISHOP_MOB_BONUS: [i32; 14] = [-100, -71, -57, -44, -34, -28, -19, -19, -14, -16, -17, -24, -14, -34];
const EG_ROOK_MOB_BONUS: [i32; 15] = [-19, -6, -11, -13, -5, -1, -1, -3, 7, 11, 19, 25, 28, 26, 11];
const EG_QUEEN_MOB_BONUS: [i32; 28] = [-231, -72, -61, -48, -52, -34, -32, -20, -22, -13, -12, -2, -1, 10, 13, 18, 13, 10, 14, 3, -10, -36, -41, -74, -41, -44, -86, -53];
const KNIGHT_MOB_BONUS: [i32; 9] = [-14, 8, 17, 26, 32, 36, 42, 48, 58];
const BISHOP_MOB_BONUS: [i32; 14] = [15, 22, 32, 36, 42, 48, 53, 56, 59, 68, 72, 94, 87, 148];
const ROOK_MOB_BONUS: [i32; 15] = [-24, -12, -5, -1, -2, 3, 8, 14, 13, 17, 17, 16, 14, 25, 53];
const QUEEN_MOB_BONUS: [i32; 28] = [6, -14, -10, -10, -3, -4, 0, 0, 4, 5, 8, 5, 7, 5, 2, 4, 7, 15, 15, 27, 40, 81, 85, 138, 92, 109, 159, 27];
const EG_PAWN_PST: [i32; 64] = [3, 3, 6, 2, -18, -7, -7, 0, 28, 60, 58, 19, 41, 38, 88, 72, 26, 31, 14, 9, -3, 5, 13, 17, 17, 10, 1, -16, -20, -16, 4, -8, 6, 7, -14, -21, -23, -16, -6, -16, -2, -3, -9, -9, -13, -7, -14, -17, 2, 7, -6, 3, -1, -8, -14, -15, -9, 3, -3, 0, 0, 0, 0, 1];
const PAWN_PST: [i32; 64] = [8, -2, 0, 1, -5, 0, 0, 0, 183, 133, 137, 122, 71, 44, -31, -40, 3, 11, 46, 34, 37, 43, 57, 5, -24, -16, -16, -6, 10, 3, -7, -9, -29, -23, -16, -14, -2, -1, -10, -26, -30, -25, -27, -18, -11, -23, 1, -27, -23, -19, -18, -26, -11, 8, 21, -21, 3, -6, -1, 13, 0, 0, 7, 0];
const EG_KNIGHT_PST: [i32; 64] = [-12, -47, -36, -58, -43, -48, -57, -66, -21, -13, -18, -21, -25, -54, -43, -35, -15, -27, -6, -15, -29, -41, -47, -42, -22, -14, -8, -12, -10, -13, -13, -39, -19, -16, -4, -6, -4, -5, -21, -19, -35, -22, -27, -14, -18, -32, -35, -28, -56, -27, -37, -35, -32, -31, -21, -25, -84, -40, -28, -37, -31, -41, -45, -59];
const KNIGHT_PST: [i32; 64] = [-112, 15, 13, 63, 47, 36, -41, -129, -47, -36, 0, 19, 36, 50, 18, -62, -41, 15, 5, 31, 66, 101, 33, 8, -1, -12, 8, 45, 17, 34, -8, 28, -14, -15, 2, 7, 12, -4, 1, -11, -34, -25, -10, -6, 5, -4, -8, -23, -22, -25, -17, 9, 4, -4, -16, -8, -25, -10, -33, -6, -14, 1, -4, -26];
const EG_BISHOP_PST: [i32; 64] = [-11, -5, -10, -17, -14, -29, -15, -23, 19, -6, -13, -15, -9, -29, -10, -33, -13, -17, -22, -26, -27, -36, -26, -26, -17, -20, -13, -27, -18, -23, -16, -23, -22, -18, -22, -23, -28, -19, -22, -32, -23, -24, -20, -34, -21, -27, -30, -28, -38, -46, -49, -32, -38, -38, -40, -55, -40, -42, -37, -26, -32, -38, -43, -51];
const BISHOP_PST: [i32; 64] = [-17, -21, -28, -23, -28, 21, -7, -11, -96, -48, -31, -29, -34, 15, -34, -7, -19, 8, -13, 3, 21, 72, 29, 14, -20, -1, -13, 34, 5, 2, -6, -11, -17, -15, 10, 18, 11, -3, -12, 2, -12, 7, 3, 20, 17, 11, 11, 4, 10, 13, 24, 5, 24, 25, 32, 19, -15, 23, 7, -12, 1, 18, 10, 4];
const EG_ROOK_PST: [i32; 64] = [-23, -1, 11, 13, 16, 17, 8, -6, 16, 22, 9, 9, 20, 1, 10, 2, 20, 9, 13, 10, 6, 4, 2, 1, 18, 18, 17, 8, 9, 8, 5, 3, 16, 14, 17, 13, 11, 15, 4, 10, -4, -1, 1, 0, -6, -5, -13, -14, -21, -12, -7, -16, -16, -21, -16, -35, -10, -17, -7, -15, -13, -10, -17, -10];
const ROOK_PST: [i32; 64] = [50, 26, 12, 5, 5, 19, 42, 47, -17, -18, 28, 28, -6, 52, 30, 23, -20, 14, 5, 11, 23, 39, 53, 13, -29, -19, -14, 5, -11, 1, 13, -11, -42, -37, -32, -29, -31, -36, -11, -50, -33, -31, -27, -30, -21, -17, 2, -25, -29, -27, -27, -16, -17, 1, -5, -12, -6, -3, -3, 3, 4, 11, 9, -22];
const EG_QUEEN_PST: [i32; 64] = [-4, -3, 14, 11, 0, 20, 12, 14, 70, 74, 52, 56, 70, 9, 77, 39, 20, 43, 69, 42, 60, 17, 35, 8, 21, 39, 57, 53, 52, 48, 63, 30, 12, 39, 36, 40, 47, 35, 26, 18, 3, 8, 10, 4, 0, 13, 10, -5, -6, 2, -5, -28, -18, -59, -47, -44, -51, -17, -39, -12, -51, -45, -66, -78];
const QUEEN_PST: [i32; 64] = [21, 30, 16, 25, 6, -8, 22, 41, -80, -85, -27, -27, -69, -4, -74, -15, -29, -32, -50, -14, -8, 22, -2, -2, -30, -23, -38, -20, -14, -8, -20, -1, -11, -34, -16, -11, -22, -11, 6, -4, -16, -4, -4, -2, 2, 8, 8, 5, -6, -8, 7, 14, 18, 36, 19, 15, 16, -3, 16, 23, 24, 8, 21, 14];
const EG_KING_PST: [i32; 64] = [-215, -124, -97, -81, -60, -66, -85, -150, -67, -41, -56, -37, -28, -16, -17, -39, -45, -9, -12, -13, -11, -10, -6, -26, -54, -14, -10, 3, 2, -1, 1, -28, -36, -8, 0, 9, 19, 14, 8, -3, -28, -8, 3, 13, 20, 13, 6, -3, -27, 5, 7, 13, 13, 12, -1, -14, -14, -18, -5, 7, -2, 9, -23, -37];
const KING_PST: [i32; 64] = [409, 376, 307, 266, 158, 255, 319, 188, 125, 127, 235, 138, 137, 108, 159, 89, 119, 82, 119, 134, 150, 167, 127, 101, 128, 99, 135, 107, 126, 145, 92, 85, 66, 56, 78, 74, 49, 52, 16, -33, -1, 13, 19, 14, 7, 9, -5, -33, 9, -23, -7, -27, -11, -11, 12, 1, -82, -1, -1, -64, -51, -51, 6, -7];
    
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
    pub fn get_qs_see_threshold(&self) -> i32 {
        QS_SEE_THRESHOLD
    }


    #[inline]
    pub fn get_qs_prune_margin(&self) -> i32 {
        QS_PRUNE_MARGIN
    }


    #[inline]
    pub fn get_timeext_score_drop_threshold(&self) -> i32 {
        TIMEEXT_SCORE_DROP_THRESHOLD
    }


    #[inline]
    pub fn get_timeext_history_size(&self) -> i32 {
        TIMEEXT_HISTORY_SIZE
    }


    #[inline]
    pub fn get_razor_margin(&self) -> i32 {
        RAZOR_MARGIN
    }


    #[inline]
    pub fn get_king_shield_bonus(&self) -> i32 {
        KING_SHIELD_BONUS
    }

    #[inline]
    pub fn get_passed_pawn_bonus(&self, index: usize) -> i32 {
        unsafe { *PASSED_PAWN_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_pawn_king_defense_bonus(&self, index: usize) -> i32 {
        unsafe { *PAWN_KING_DEFENSE_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_pawn_king_attacked_penalty(&self, index: usize) -> i32 {
        unsafe { *PAWN_KING_ATTACKED_PENALTY.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_passed_pawn_king_defense_bonus(&self, index: usize) -> i32 {
        unsafe { *PASSED_PAWN_KING_DEFENSE_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_passed_pawn_king_attacked_penalty(&self, index: usize) -> i32 {
        unsafe { *PASSED_PAWN_KING_ATTACKED_PENALTY.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_half_open_file_bonus(&self, index: usize) -> i32 {
        unsafe { *HALF_OPEN_FILE_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_eg_half_open_file_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_HALF_OPEN_FILE_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_king_threat_by_piece_combo(&self, index: usize) -> i32 {
        unsafe { *KING_THREAT_BY_PIECE_COMBO.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_eg_knight_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_KNIGHT_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_eg_bishop_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_BISHOP_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_eg_rook_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_ROOK_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_eg_queen_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_QUEEN_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_knight_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *KNIGHT_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_bishop_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *BISHOP_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_rook_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *ROOK_MOB_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_queen_mob_bonus(&self, index: usize) -> i32 {
        unsafe { *QUEEN_MOB_BONUS.get_unchecked(index) }
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
    let piece_value = get_piece_value(piece as usize);

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

