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

pub struct EvalOptions {
    passed_pawn_bonus: [i32; 4],
    passed_pawn_king_defense_bonus: [i32; 8],
    passed_pawn_king_attacked_penalty: [i32; 8],
}

impl EvalOptions {
    pub fn new() -> Self {
        EvalOptions{
            passed_pawn_bonus: PASSED_PAWN_BONUS,
            passed_pawn_king_defense_bonus: PASSED_PAWN_KING_DEFENSE_BONUS,
            passed_pawn_king_attacked_penalty: PASSED_PAWN_KING_ATTACKED_PENALTY,
        }
    }

    pub fn set_passed_pawns(&mut self, index: i32, value: i32) {
        self.passed_pawn_bonus[index as usize] = value;
    }

    pub fn set_passed_pawn_king_defense_bonus(&mut self, index: i32, value: i32) {
        self.passed_pawn_king_defense_bonus[index as usize] = value;
    }

    pub fn set_passed_pawn_king_attacked_penalty(&mut self, index: i32, value: i32) {
        self.passed_pawn_king_attacked_penalty[index as usize] = value;
    }

    pub fn get_passed_pawn_bonus(&self, pos: u32) -> i32 {
        self.passed_pawn_bonus[pos as usize]
    }

    pub fn get_passed_pawn_king_defense_bonus(&self, distance: i32) -> i32 {
        self.passed_pawn_king_defense_bonus[distance as usize]
    }

    pub fn get_passed_pawn_king_attacked_penalty(&self, distance: i32) -> i32 {
        self.passed_pawn_king_attacked_penalty[distance as usize]
    }
}

const PASSED_PAWN_BONUS: [i32; 4] = [121, 91, 45, 10];
const PASSED_PAWN_KING_DEFENSE_BONUS: [i32; 8] = [0, 37, 27, 7, -3, -5, 1, -1];
const PASSED_PAWN_KING_ATTACKED_PENALTY: [i32; 8] = [0, 53, 25, -6, -19, -23, -19, -10];
