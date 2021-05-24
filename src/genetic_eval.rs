
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

// Auto-generated file (see src/bin/gen_eval.rs)

use crate::random::Random;

pub struct GeneticEvaluator {}

impl GeneticEvaluator {

    pub fn new() -> Self {
        GeneticEvaluator{}
    }

    pub fn compile(&mut self) {}

    #[inline]
    // Eval terms evolved using genetic programming
    pub fn eval(&self, own_pawns: u64, opp_pawns: u64, own_knights: u64, opp_knights: u64, own_bishops: u64, opp_bishops: u64, own_rooks: u64, opp_rooks: u64, own_queens: u64, opp_queens: u64, own_king_bb: u64, opp_king_bb: u64, own_pawn_attacks: u64, opp_pawn_attacks: u64, own_knight_attacks: u64, opp_knight_attacks: u64, own_bishop_attacks: u64, opp_bishop_attacks: u64, own_rook_attacks: u64, opp_rook_attacks: u64, own_queen_attacks: u64, opp_queen_attacks: u64, opp_king_half: u64, own_king_half: u64) -> i32 {
        let mut score: i32 = 0;
        // ------------------------------------------------
        let mut tmp_score = 26744u64.wrapping_mul(own_pawns).wrapping_mul(own_pawns.wrapping_mul(own_pawns).wrapping_mul(own_queen_attacks.wrapping_mul(own_pawns.wrapping_mul(26744u64)).wrapping_mul(26739u64)).wrapping_mul(26787u64).wrapping_mul(26739u64).wrapping_mul(26744u64.wrapping_mul(own_queen_attacks.wrapping_mul(26744u64).wrapping_mul(own_pawns.wrapping_mul((own_queen_attacks ^ own_queen_attacks.wrapping_mul(own_pawns).wrapping_mul(26791u64)))).wrapping_mul(own_queen_attacks).wrapping_mul(own_queen_attacks.wrapping_mul(26744u64))).wrapping_mul(own_pawns))).wrapping_mul(own_pawns).wrapping_mul(own_queen_attacks.wrapping_mul(own_pawns)).wrapping_mul(own_pawns.wrapping_mul(own_pawns)).wrapping_mul(4503599627397287u64).wrapping_mul(own_pawns).wrapping_mul(own_pawns.wrapping_mul(own_pawns).wrapping_mul(own_queen_attacks.wrapping_mul(own_pawns.wrapping_mul(own_queen_attacks).wrapping_mul(own_pawns)).wrapping_mul(own_pawns).wrapping_mul(own_queen_attacks).wrapping_mul(own_pawns.wrapping_mul(own_queen_attacks).wrapping_mul(own_pawns))).wrapping_mul(26739u64.wrapping_mul(own_queen_attacks))).wrapping_mul(26744u64).count_ones() as i32;
        score += tmp_score * -22;
        // ------------------------------------------------
        let mut tmp_score = own_rooks.count_ones() as i32;
        score += tmp_score * -21;
        // ------------------------------------------------
        let mut tmp_score = (8791u64 ^ (if (if (own_king_half.wrapping_mul(own_king_half) & (opp_queen_attacks.wrapping_mul(opp_king_half) & (if (if ((if own_bishops != 0 { 1u64 } else { 0u64 }).wrapping_mul((if opp_queen_attacks != 0 { 1u64 } else { 0u64 })) & own_king_half).wrapping_mul((if own_bishops != 0 { 1u64 } else { 0u64 }).wrapping_mul((if (if opp_king_half != 0 { 1u64 } else { 0u64 }).wrapping_mul((if ((1u64 | own_king_half) & own_bishops) != 0 { 1u64 } else { 0u64 })) != 0 { 1u64 } else { 0u64 }))).wrapping_mul(own_king_half) != 0 { 1u64 } else { 0u64 }) != 0 { 1u64 } else { 0u64 }).wrapping_mul((if opp_queen_attacks != 0 { 1u64 } else { 0u64 }))).wrapping_mul((if opp_queen_attacks != 0 { 1u64 } else { 0u64 }))).wrapping_mul((if (opp_queens & (((if (if own_bishops != 0 { 1u64 } else { 0u64 }) != 0 { 1u64 } else { 0u64 }) | 18446744073709551614u64) & own_king_half).wrapping_mul(opp_king_half).wrapping_mul(own_king_half)) != 0 { 1u64 } else { 0u64 }).wrapping_mul((if (if own_bishops != 0 { 1u64 } else { 0u64 }) != 0 { 1u64 } else { 0u64 }))) != 0 { 1u64 } else { 0u64 }) != 0 { 1u64 } else { 0u64 })).count_ones() as i32;
        score += tmp_score * -19 * (tmp_score + 1) / 2 * 1;
        // ------------------------------------------------
        let mut tmp_score = (opp_queens & own_knight_attacks).count_ones() as i32;
        score += tmp_score * -11;
        // ------------------------------------------------
        let mut tmp_score = own_rooks.count_ones() as i32;
        score += tmp_score * -8;
        // ------------------------------------------------
        let mut tmp_score = (13783u64 & (((own_king_half & (((843134u64 ^ own_pawns) & ((if own_pawns != 0 { 1u64 } else { 0u64 }) & 72057594037941718u64)) & (own_king_half & ((1u64 & (if (((13783u64 | 5605u64) & ((13783u64 | 13783u64) | ((if 13783u64 != 0 { 1u64 } else { 0u64 }) & (if (((own_pawns | 13783u64) & ((13783u64 | (13783u64 & (13783u64 & 13783u64))) | (5605u64 & own_king_half))) & (((if 13783u64 != 0 { 1u64 } else { 0u64 }) & 13783u64) & 13783u64)) != 0 { 1u64 } else { 0u64 })))) & (13783u64 & ((if 13783u64 != 0 { 1u64 } else { 0u64 }) & (own_pawns | 5605u64)))) != 0 { 1u64 } else { 0u64 })) & own_king_half)))) & 13783u64) & 1u64)).count_ones() as i32;
        score += tmp_score * -6;
        // ------------------------------------------------
        let mut tmp_score = (if (((((((((if (own_rook_attacks | (((((((if ((own_rook_attacks ^ own_pawn_attacks) | (((own_rook_attacks.wrapping_mul((2091739439495836932u64 ^ own_rook_attacks)) | (own_pawn_attacks | own_pawn_attacks).wrapping_mul(own_rook_attacks)) | (own_pawn_attacks.wrapping_mul(own_pawn_attacks) ^ own_pawn_attacks)) & own_rook_attacks)) != 0 { 1u64 } else { 0u64 }) | own_rook_attacks.wrapping_mul(own_pawn_attacks)) | (own_rook_attacks ^ 2091739439495836948u64)) & own_pawn_attacks) | 327u64) | (own_pawn_attacks ^ own_pawn_attacks)) & own_pawn_attacks)) != 0 { 1u64 } else { 0u64 }) | own_rook_attacks) | own_pawn_attacks) & own_rook_attacks.wrapping_mul((2091739439495836932u64 ^ opp_rook_attacks))) | 2091739439495836948u64) | own_pawn_attacks) & opp_bishops.wrapping_mul(own_pawn_attacks)) | opp_rook_attacks) | own_pawn_attacks.wrapping_mul(own_queen_attacks)) != 0 { 1u64 } else { 0u64 }).count_ones() as i32;
        score += tmp_score * -6 * (tmp_score + 1) / 2 * 1;
        // ------------------------------------------------
        let mut tmp_score = opp_knight_attacks.count_ones() as i32;
        score += tmp_score * -3;
        // ------------------------------------------------
        let mut tmp_score = opp_bishop_attacks.count_ones() as i32;
        tmp_score += ((((opp_bishops ^ 98285u64) & own_rook_attacks) & own_knights) & own_rook_attacks).count_ones() as i32;
        tmp_score += ((opp_pawns | opp_pawn_attacks) ^ (((((((((opp_bishops | ((2538u64 | (7u64 | own_rooks)) | opp_pawn_attacks)) ^ opp_pawn_attacks) | ((((2538u64 & (opp_rook_attacks | ((((opp_pawn_attacks ^ opp_king_bb) | ((opp_pawn_attacks ^ 2320u64) | opp_queens)) | ((opp_bishops & 2538u64) & (opp_pawns ^ (2538u64 & (((opp_rook_attacks | opp_pawn_attacks) | (2537u64 | own_rooks)) | opp_pawn_attacks))))) ^ opp_rook_attacks))) ^ 2538u64) | 1809684u64) | opp_bishops)) | 9228u64) | opp_pawns) ^ (opp_king_bb | opp_rook_attacks)) ^ 2538u64) | (opp_rook_attacks | 2537u64)) ^ opp_pawn_attacks)).count_ones() as i32;
        tmp_score += own_king_bb.count_ones() as i32;
        tmp_score += opp_queen_attacks.count_ones() as i32;
        score += tmp_score * -2;
        // ------------------------------------------------
        let mut tmp_score = own_rooks.count_ones() as i32;
        tmp_score += ((own_bishops ^ 359034u64) ^ (own_knight_attacks ^ ((own_bishops ^ (own_bishops | (359034u64 ^ (((own_bishops | 359034u64) ^ (own_bishops ^ 359034u64)) | own_bishops)))) ^ (own_king_bb ^ own_queen_attacks)))).count_ones() as i32;
        score += tmp_score * 1;
        // ------------------------------------------------
        let mut tmp_score = (((own_pawns | ((own_pawns | (own_pawns | own_pawn_attacks.wrapping_mul((own_pawns | (own_pawns | own_pawns.wrapping_mul((own_pawns | ((own_pawns | (own_pawns | opp_knights.wrapping_mul(opp_king_bb.wrapping_mul((own_pawns | (opp_king_bb.wrapping_mul((own_pawns | own_pawns.wrapping_mul((own_pawns | (own_pawns | own_pawns.wrapping_mul(own_pawns)))))) | own_pawns).wrapping_mul(own_pawns)))))) | own_pawns))).wrapping_mul(((own_pawns | own_pawns.wrapping_mul(own_pawns)) | own_pawns))))))) | own_pawns)) | own_pawns) | own_pawns).count_ones() as i32;
        tmp_score += ((opp_knight_attacks & 278996u64) ^ (((8573u64 ^ ((278992u64 | (opp_queens & opp_knight_attacks)) | ((278992u64 ^ (own_rook_attacks | (((((opp_knight_attacks & (opp_queens ^ (opp_knight_attacks & 8573u64))) | (opp_knight_attacks & (((opp_queens ^ 278988u64) | opp_queens) ^ (((opp_queens | (opp_knight_attacks ^ opp_knights)) | opp_knight_attacks) | (((opp_queens | (278992u64 | own_bishop_attacks)) ^ (opp_knight_attacks & 8573u64)) | opp_queens))))) & (((278996u64 ^ opp_knights) | opp_queens) ^ (opp_queens | opp_knight_attacks))) | (opp_queens ^ (opp_knight_attacks & 278992u64))) ^ opp_queens))) | (opp_queens | 278992u64)))) & (278992u64 | opp_queens)) & opp_queens)).count_ones() as i32;
        tmp_score += own_knights.count_ones() as i32;
        tmp_score += (((own_queen_attacks & ((((((opp_queen_attacks ^ (opp_queen_attacks | (opp_queen_attacks ^ (874u64 ^ opp_queen_attacks)))) & opp_queen_attacks) ^ own_queen_attacks) ^ own_queen_attacks) & own_queen_attacks) | (((18374686479671623624u64 & ((opp_bishop_attacks | (opp_queen_attacks & opp_queen_attacks)) ^ (opp_queen_attacks ^ own_queen_attacks))) ^ ((opp_bishop_attacks | opp_queen_attacks) | ((opp_rooks | opp_queen_attacks) ^ (((18374686479671623624u64 & opp_queen_attacks) ^ opp_queen_attacks) & opp_queen_attacks)))) & own_queen_attacks))) | (opp_bishop_attacks & own_queen_attacks)) ^ (opp_queen_attacks & opp_queen_attacks)).count_ones() as i32;
        score += tmp_score * 2;
        // ------------------------------------------------
        let mut tmp_score = (own_rook_attacks ^ (own_rook_attacks ^ ((((own_rooks ^ (own_rook_attacks ^ own_pawns)) ^ own_rooks) ^ (own_rooks ^ own_rook_attacks)) ^ (((own_rooks ^ own_rook_attacks) ^ (own_rooks ^ ((82069u64 ^ ((((own_rooks ^ ((((((82069u64 ^ (((own_rooks ^ own_rook_attacks) ^ own_rook_attacks) ^ 82069u64)) ^ own_pawns) ^ own_rooks) ^ 82069u64) ^ 82069u64) ^ own_rook_attacks)) ^ (82069u64 ^ ((82069u64 ^ own_rooks) ^ own_rook_attacks))) ^ own_rooks) ^ 82069u64)) ^ 82069u64))) ^ 82069u64)))).count_ones() as i32;
        tmp_score += (((((opp_king_bb.wrapping_mul(((6413831855076651u64 | opp_king_bb) ^ 35184379450022u64)) & opp_king_bb) | 30u64).wrapping_mul(30u64.wrapping_mul(((((opp_king_bb & (opp_king_bb | 30u64)) | 30u64).wrapping_mul((30u64 | (6413831855076651u64 | opp_king_bb)).wrapping_mul(opp_king_bb)) | 30u64).wrapping_mul(((6413831855076651u64 | opp_king_bb) & 35184379450022u64)) ^ opp_king_bb))) & 35184379450022u64).wrapping_mul((opp_king_bb & 35184379450022u64).wrapping_mul((opp_king_bb.wrapping_mul(((6413831855076651u64 | opp_king_bb) ^ 30u64)) & opp_king_bb))) & opp_king_bb.wrapping_mul(((6413831855076651u64 | opp_king_bb) ^ (opp_king_bb & opp_king_bb)))) ^ (30u64 & opp_king_bb)).wrapping_mul(opp_king_bb).count_ones() as i32;
        score += tmp_score * 3;
        // ------------------------------------------------
        let mut tmp_score = ((opp_pawns & (((if ((((opp_queens | opp_pawns) & (opp_queens.wrapping_mul(opp_queens) | (opp_queens | ((if (425837u64 & ((if ((opp_king_bb | ((opp_pawns | opp_queens) & opp_queens.wrapping_mul((opp_pawn_attacks & opp_bishops)))).wrapping_mul((if ((opp_pawn_attacks & opp_bishops) & opp_queens) != 0 { 1u64 } else { 0u64 })).wrapping_mul((opp_pawns | opp_queens)) | (27373u64 ^ (opp_pawns & 27356u64))) != 0 { 1u64 } else { 0u64 }) & opp_queens)) != 0 { 1u64 } else { 0u64 }) & opp_pawn_attacks.wrapping_mul((opp_pawn_attacks & 27373u64)))).wrapping_mul(opp_queens)).wrapping_mul(opp_queens)) | (opp_pawn_attacks & 425837u64)).wrapping_mul((if opp_queens != 0 { 1u64 } else { 0u64 })).wrapping_mul(27373u64) | ((opp_queens ^ opp_knights) ^ opp_queens)) != 0 { 1u64 } else { 0u64 }) & opp_queens) | opp_king_bb).wrapping_mul(opp_bishops)) | opp_pawns).wrapping_mul(opp_queens).count_ones() as i32;
        score += tmp_score * 10;
        // ------------------------------------------------
        let mut tmp_score = (own_pawns.wrapping_mul(opp_knights) | (opp_knights.wrapping_mul((opp_knights.wrapping_mul(((opp_knights & opp_king_bb) & (opp_knights.wrapping_mul(opp_knights).wrapping_mul(((opp_knights | opp_knights.wrapping_mul((opp_knights.wrapping_mul((opp_knights.wrapping_mul(17179869185u64) & opp_king_bb.wrapping_mul(own_pawns))) & 2u64))) | ((own_pawns.wrapping_mul(17179869185u64) | own_pawns.wrapping_mul((opp_knights.wrapping_mul((((own_pawns.wrapping_mul(own_pawns) | own_pawns.wrapping_mul((opp_knights.wrapping_mul((opp_knights.wrapping_mul(((2u64 & opp_king_bb.wrapping_mul(own_pawns)) | opp_knights)) & own_bishop_attacks)) & own_pawns))) | (2u64 & opp_knights)) & own_bishop_attacks)) & own_pawns))) | (2u64 & opp_knights)))) | opp_knights))) & (17179869185u64 & own_pawns))).wrapping_mul((2u64 | opp_knights)) | opp_knights)).count_ones() as i32;
        score += tmp_score * 11;
        // ------------------------------------------------
        let mut tmp_score = (((opp_rooks & ((opp_rooks.wrapping_mul(opp_rooks).wrapping_mul(own_rooks) | (own_rooks.wrapping_mul(own_rooks) & !((opp_pawns & own_rooks) ^ ((opp_pawns & own_rooks).wrapping_mul(own_rooks) & (((own_rooks & (opp_pawns & own_rooks)) & !opp_rooks) & opp_queens).wrapping_mul((own_rooks.wrapping_mul(opp_rooks) & !own_rooks.wrapping_mul(own_rooks)))))).wrapping_mul((1962897u64 & opp_rooks)).wrapping_mul((own_rooks ^ opp_pawns))) ^ opp_pawns)) ^ own_rooks.wrapping_mul((own_rooks ^ own_rooks.wrapping_mul(opp_bishop_attacks)))) & ((own_rooks & (opp_pawns & own_rooks)) & !own_rooks)).count_ones() as i32;
        tmp_score += own_pawn_attacks.count_ones() as i32;
        score += tmp_score * 14;
        score
    }


    pub fn add_program(&mut self, _: GeneticProgram) {}

    pub fn clear(&mut self) {}

    pub fn init_generation(&self, _: &mut Random, _: u32) {}

    pub fn create_new_generation(&self, _: &mut Random) {}

    pub fn print_rust_code(&self) {}
}

#[derive(Copy, Clone)]
#[allow(unused_mut)]
pub struct GeneticProgram {}

impl GeneticProgram {
    pub fn new_from_str(_: &str, _: [u64; 8], _: i32, _: i32) -> GeneticProgram {
        GeneticProgram{}
    }
}

