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
use crate::pieces::{B, K, N, P, Q, R, get_piece_value};
use crate::score_util::{pack_scores};

const FUTILITY_MARGIN_MULTIPLIER: i32 = 51;
const QS_SEE_THRESHOLD: i32 = 104;
const QS_PRUNE_MARGIN: i32 = 989;
const TIMEEXT_SCORE_CHANGE_THRESHOLD: i32 = 80;
const TIMEEXT_SCORE_FLUCTUATION_THRESHOLD: i32 = 130;
const TIMEEXT_SCORE_FLUCTUATION_REDUCTIONS: i32 = 90;
const RAZOR_MARGIN: i32 = 130;
const QUEEN_KING_THREAT: i32 = 3;
const ROOK_KING_THREAT: i32 = 2;
const BISHOP_KING_THREAT: i32 = 1;
const KNIGHT_KING_THREAT: i32 = 1;
const CASTLING_BONUS: i32 = 29;
const LOST_QUEENSIDE_CASTLING_PENALTY: i32 = 28;
const LOST_KINGSIDE_CASTLING_PENALTY: i32 = 53;
const DOUBLED_PAWN_PENALTY: i32 = 22;
const KING_SHIELD_BONUS: i32 = 17;
const PAWN_COVER_BONUS: i32 = 8;
const ROOK_ON_HALF_OPEN_FILE_BONUS: i32 = 11;
const ROOK_ON_OPEN_FILE_BONUS: i32 = 14;
const EG_ROOK_ON_HALF_OPEN_FILE_BONUS: i32 = -6;
const EG_ROOK_ON_OPEN_FILE_BONUS: i32 = 1;
const BISHOP_PIN_BONUS: i32 = 17;
const EG_BISHOP_PIN_BONUS: i32 = 10;
const ROOK_PIN_BONUS: i32 = 2;
const EG_ROOK_PIN_BONUS: i32 = 9;
const UNCOVERED_PIECE_PENALTY: i32 = 4;
const EG_PASSED_PAWN_BONUS: [i32; 4] = [115, 97, 56, 10];
const PASSED_PAWN_BONUS: [i32; 4] = [2, 23, 10, 0];
const PASSED_PAWN_KING_DEFENSE_BONUS: [i32; 8] = [3, 95, 72, 45, 27, 21, 31, 23];
const PASSED_PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [5, 100, 60, 34, 18, 0, 0, 4];
const KING_DANGER_PIECE_PENALTY: [i32; 20] = [0, 0, 5, 1, 11, 21, 34, 61, 75, 111, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170];
const KING_THREAT_ADJUSTMENT: [i32; 128] = [0, 1, 0, 0, -16, -12, -9, 4, 10, 16, 21, 25, -10, -6, -6, -8, -4, -7, -7, -6, -1, 4, 11, 20, 5, 6, 4, 5, -5, 3, 4, 15, 2, -7, -3, 4, -3, 7, 19, 36, -4, -7, 8, 20, -10, 10, 31, 44, -10, -9, 1, 11, -7, 10, 34, 57, -9, 2, 8, 19, -18, 6, 45, 77, 12, 10, 13, 32, 46, 45, 67, 90, 26, 31, 31, 56, 47, 79, 69, 192, 23, 27, 42, 63, 54, 61, 93, 107, 48, 40, 74, 82, 37, 112, 94, 219, -299, 188, 505, 501, 113, 619, 4, -19, 0, 380, 1, 532, 0, -4, 12, -9, 0, -1, 0, 0, 551, 1, 0, 0, 0, 625, 421, 1, 15, 0, 0, 1];
const EG_KNIGHT_MOB_BONUS: [i32; 9] = [-125, -82, -56, -47, -41, -31, -34, -39, -51];
const EG_BISHOP_MOB_BONUS: [i32; 14] = [-98, -67, -54, -43, -33, -26, -20, -17, -16, -17, -18, -26, -16, -36];
const EG_ROOK_MOB_BONUS: [i32; 15] = [-25, -15, -17, -19, -10, -9, -8, -9, -1, 2, 11, 16, 19, 19, 1];
const EG_QUEEN_MOB_BONUS: [i32; 28] = [-152, -34, -41, -39, -38, -24, -23, -14, -14, -7, -6, 2, 6, 12, 11, 13, 10, 3, 4, -13, -26, -67, -68, -115, -76, -84, -178, -230];
const KNIGHT_MOB_BONUS: [i32; 9] = [-31, -10, -2, 6, 13, 17, 23, 31, 47];
const BISHOP_MOB_BONUS: [i32; 14] = [1, 9, 19, 24, 31, 37, 41, 44, 49, 57, 63, 84, 75, 131];
const ROOK_MOB_BONUS: [i32; 15] = [-29, -19, -17, -12, -12, -6, 0, 7, 7, 13, 11, 12, 11, 21, 56];
const QUEEN_MOB_BONUS: [i32; 28] = [-13, -23, -14, -13, -7, -7, -3, -3, 1, 2, 5, 3, 3, 2, 2, 5, 8, 17, 19, 38, 55, 110, 106, 184, 130, 152, 313, 438];
const EG_PAWN_PST: [i32; 64] = [1, 0, 0, 5, 0, 0, -4, 0, 36, 63, 61, 23, 43, 39, 94, 79, 27, 31, 11, 8, -3, 3, 12, 17, 16, 10, 1, -18, -20, -16, 5, -9, 6, 7, -15, -22, -24, -17, -6, -16, -3, -3, -10, -10, -13, -7, -14, -18, 2, 7, -7, 3, -1, -10, -14, -17, -9, 0, 1, 0, 0, 0, 0, 0];
const PAWN_PST: [i32; 64] = [1, -2, 0, 1, -5, 0, 0, 0, 152, 124, 123, 110, 63, 38, -33, -46, 2, 10, 44, 30, 30, 40, 54, 3, -24, -15, -18, -6, 9, 2, -9, -10, -29, -23, -15, -14, -3, -3, -11, -27, -29, -24, -26, -18, -12, -24, 1, -26, -23, -19, -17, -27, -12, 9, 21, -22, 4, -6, -1, 0, 0, 0, 0, 0];
const EG_KNIGHT_PST: [i32; 64] = [-14, -43, -35, -61, -44, -46, -57, -69, -20, -11, -19, -16, -21, -55, -43, -35, -13, -25, -1, -11, -29, -36, -46, -40, -21, -9, -2, -8, -4, -10, -10, -41, -17, -16, -3, -2, -2, -1, -20, -18, -36, -21, -25, -13, -17, -26, -32, -27, -56, -30, -37, -34, -27, -29, -20, -23, -90, -36, -28, -36, -28, -40, -40, -64];
const KNIGHT_PST: [i32; 64] = [-117, -4, 2, 58, 42, 22, -48, -136, -59, -46, -11, -2, 21, 36, 11, -69, -48, -1, -17, 12, 47, 81, 22, 2, -7, -21, -8, 29, 4, 25, -11, 28, -15, -18, -3, -7, 2, -7, 2, -8, -34, -26, -11, -6, 5, -6, -7, -24, -23, -23, -15, 8, 1, -4, -16, -8, -18, -12, -31, -3, -12, 2, -6, -26];
const EG_BISHOP_PST: [i32; 64] = [-9, -7, -11, -17, -11, -27, -16, -22, 24, -6, -9, -12, -4, -26, -13, -33, -9, -10, -15, -20, -22, -35, -23, -27, -14, -18, -9, -23, -15, -23, -17, -22, -23, -15, -18, -19, -23, -15, -18, -38, -21, -23, -14, -26, -13, -25, -28, -25, -39, -33, -45, -25, -28, -35, -31, -59, -39, -41, -27, -26, -33, -29, -45, -54];
const BISHOP_PST: [i32; 64] = [-25, -21, -32, -16, -31, 12, -10, -19, -95, -40, -30, -26, -30, 11, -24, -6, -22, 2, -15, 5, 17, 79, 26, 25, -19, 5, -7, 37, 15, 15, 9, -6, -9, -8, 14, 28, 19, 5, -8, 17, -3, 14, 8, 22, 22, 18, 14, 9, 16, 13, 27, 10, 23, 30, 32, 25, -10, 27, 11, -3, 10, 11, 15, 12];
const EG_ROOK_PST: [i32; 64] = [-22, -1, 10, 11, 15, 23, 8, -5, 15, 21, 8, 9, 19, 2, 11, 4, 20, 9, 13, 11, 5, 4, 1, -1, 19, 18, 16, 8, 9, 9, 4, 3, 15, 15, 18, 14, 12, 17, 3, 9, -4, -1, 2, 1, -6, -2, -13, -12, -20, -10, -6, -14, -14, -17, -14, -35, -10, -15, -6, -14, -12, -9, -16, -6];
const ROOK_PST: [i32; 64] = [44, 21, 9, 3, 2, -1, 41, 47, -19, -21, 26, 26, -7, 50, 26, 16, -23, 13, 3, 5, 20, 34, 52, 14, -34, -19, -14, 4, -13, -2, 11, -16, -42, -38, -31, -29, -34, -41, -14, -54, -35, -31, -28, -32, -22, -21, 1, -31, -30, -29, -26, -16, -18, -2, -8, -17, -6, -3, -2, 4, 3, 11, 7, -25];
const EG_QUEEN_PST: [i32; 64] = [-5, -4, 8, 10, -6, 16, 7, 9, 66, 74, 45, 48, 61, 11, 71, 30, 17, 38, 65, 33, 50, 9, 28, 5, 23, 32, 48, 41, 47, 41, 63, 30, 10, 43, 32, 33, 43, 33, 23, 11, 1, 6, 5, 2, -1, 7, 9, -9, -5, 0, -4, -28, -17, -62, -52, -52, -56, -16, -36, -11, -46, -48, -70, -79];
const QUEEN_PST: [i32; 64] = [15, 27, 14, 22, 8, -1, 19, 37, -79, -82, -18, -19, -57, 2, -58, -10, -31, -27, -44, -3, 2, 28, 3, -5, -31, -19, -30, -10, -9, -2, -20, -3, -13, -36, -14, -6, -20, -10, 5, -1, -15, -4, -3, -5, 2, 10, 9, 5, -7, -7, 5, 13, 16, 35, 12, 13, 20, -4, 16, 23, 21, 10, 19, 10];
const EG_KING_PST: [i32; 64] = [-232, -127, -99, -78, -55, -64, -82, -141, -64, -40, -53, -32, -25, -16, -14, -36, -41, -5, -11, -10, -7, -3, 1, -26, -50, -13, -4, 7, 5, -1, 4, -26, -35, -7, 5, 15, 21, 18, 10, 1, -24, -6, 5, 15, 22, 14, 7, -1, -23, 6, 9, 14, 15, 13, -1, -14, -12, -20, -2, 10, 0, 8, -31, -38];
const KING_PST: [i32; 64] = [368, 329, 272, 206, 124, 219, 261, 80, 105, 114, 199, 105, 110, 104, 139, 80, 96, 69, 112, 116, 130, 119, 98, 95, 106, 96, 105, 88, 115, 137, 83, 73, 66, 53, 61, 54, 45, 39, 15, -37, -7, 10, 18, 14, 5, 9, -4, -34, 2, -23, -6, -22, -12, -13, 8, -3, -82, 0, -2, -58, -48, -50, 10, -8];
    
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
    pub fn get_futility_margin_multiplier(&self) -> i32 {
        FUTILITY_MARGIN_MULTIPLIER
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
    pub fn get_timeext_score_change_threshold(&self) -> i32 {
        TIMEEXT_SCORE_CHANGE_THRESHOLD
    }


    #[inline]
    pub fn get_timeext_score_fluctuation_threshold(&self) -> i32 {
        TIMEEXT_SCORE_FLUCTUATION_THRESHOLD
    }


    #[inline]
    pub fn get_timeext_score_fluctuation_reductions(&self) -> i32 {
        TIMEEXT_SCORE_FLUCTUATION_REDUCTIONS
    }


    #[inline]
    pub fn get_razor_margin(&self) -> i32 {
        RAZOR_MARGIN
    }


    #[inline]
    pub fn get_queen_king_threat(&self) -> i32 {
        QUEEN_KING_THREAT
    }


    #[inline]
    pub fn get_rook_king_threat(&self) -> i32 {
        ROOK_KING_THREAT
    }


    #[inline]
    pub fn get_bishop_king_threat(&self) -> i32 {
        BISHOP_KING_THREAT
    }


    #[inline]
    pub fn get_knight_king_threat(&self) -> i32 {
        KNIGHT_KING_THREAT
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
    pub fn get_rook_on_half_open_file_bonus(&self) -> i32 {
        ROOK_ON_HALF_OPEN_FILE_BONUS
    }


    #[inline]
    pub fn get_rook_on_open_file_bonus(&self) -> i32 {
        ROOK_ON_OPEN_FILE_BONUS
    }


    #[inline]
    pub fn get_eg_rook_on_half_open_file_bonus(&self) -> i32 {
        EG_ROOK_ON_HALF_OPEN_FILE_BONUS
    }


    #[inline]
    pub fn get_eg_rook_on_open_file_bonus(&self) -> i32 {
        EG_ROOK_ON_OPEN_FILE_BONUS
    }


    #[inline]
    pub fn get_bishop_pin_bonus(&self) -> i32 {
        BISHOP_PIN_BONUS
    }


    #[inline]
    pub fn get_eg_bishop_pin_bonus(&self) -> i32 {
        EG_BISHOP_PIN_BONUS
    }


    #[inline]
    pub fn get_rook_pin_bonus(&self) -> i32 {
        ROOK_PIN_BONUS
    }


    #[inline]
    pub fn get_eg_rook_pin_bonus(&self) -> i32 {
        EG_ROOK_PIN_BONUS
    }


    #[inline]
    pub fn get_uncovered_piece_penalty(&self) -> i32 {
        UNCOVERED_PIECE_PENALTY
    }

    #[inline]
    pub fn get_eg_passed_pawn_bonus(&self, index: usize) -> i32 {
        unsafe { *EG_PASSED_PAWN_BONUS.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_passed_pawn_bonus(&self, index: usize) -> i32 {
        unsafe { *PASSED_PAWN_BONUS.get_unchecked(index) }
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
    pub fn get_king_danger_piece_penalty(&self, index: usize) -> i32 {
        unsafe { *KING_DANGER_PIECE_PENALTY.get_unchecked(index) }
    }
    
    #[inline]
    pub fn get_king_threat_adjustment(&self, index: usize) -> i32 {
        unsafe { *KING_THREAT_ADJUSTMENT.get_unchecked(index) }
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

