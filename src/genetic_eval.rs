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

use crate::random::Random;

pub struct GeneticEvaluator {}

impl GeneticEvaluator {

    pub fn new() -> Self {
        GeneticEvaluator{}
    }

    pub fn compile(&mut self) {}

    #[inline]
    pub fn eval(&mut self, own_pawns: u64, opp_pawns: u64, own_knights: u64, opp_knights: u64, own_bishops: u64, opp_bishops: u64,
                own_rooks: u64, opp_rooks: u64, own_queens: u64, opp_queens: u64, own_king_bb: u64, opp_king_bb: u64,
                own_pawn_attacks: u64, opp_pawn_attacks: u64, own_knight_attacks: u64, opp_knight_attacks: u64, own_bishop_attacks: u64, opp_bishop_attacks: u64,
                own_rook_attacks: u64, opp_rook_attacks: u64, own_queen_attacks: u64, opp_queen_attacks: u64,
                opp_king_half: u64, own_king_half: u64) -> i64 {

        0
    }

    pub fn add_program(&mut self, _: GeneticProgram) {}

    pub fn clear(&mut self) {}

    pub fn init_generation(&self, _: &mut Random, _: u32) {}

    pub fn create_new_generation(&self, _: &mut Random) {}

    pub fn print_rust_code(&self) {}
}

#[derive(Copy, Clone)]
pub struct GeneticProgram {}

impl GeneticProgram {
    pub fn new_from_str(_: &str, _: [u64; 8], _: i32, _: i32) -> GeneticProgram {
        GeneticProgram{}
    }
}
