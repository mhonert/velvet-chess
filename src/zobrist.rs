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


// Random numbers to be used for calculating Zobrist hashes for board positions (see https://www.chessprogramming.org/Zobrist_Hashing):

// Note: the incremental calculation of the Zobrist hashes takes place in the Board class (see board.rs), so
// the unit tests for the hash calculation can be found in the board.rs file as well

use crate::random::{Random};

pub struct Zobrist {
    pieces: [u64; 13 * 64],
    pub player: u64,
    pub en_passant: [u64; 16],
    pub castling: [u64; 16]
}

impl Zobrist {
    pub fn new() -> Self {
        let mut random = Random::new();

        let mut pieces: [u64; 13 * 64] = [0; 13 * 64];
        for i in 0..13 * 64 {
            pieces[i] = random.rand64();
        }

        let player = random.rand64();

        let mut en_passant: [u64; 16] = [0; 16];
        for i in 0..16 {
            en_passant[i] = random.rand64();
        }

        let mut castling: [u64; 16] = [0; 16];

        // Optimization: setting the last element to 0 allows to remove some branching (xor with 0 does not change the hash)
        for i in 0..(16 - 1) {
            castling[i] = random.rand64();
        }


        Zobrist{ pieces, player, en_passant, castling }
    }
    
    pub fn piece_numbers(&self, piece: i8, pos: usize) -> u64 {
        self.pieces[((piece + 6) as usize) * 64 + pos] 
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zobrist_keys_initialized() {
        let zobrist = Zobrist::new();
        assert_ne!(zobrist.pieces[0], zobrist.pieces[1]);
        assert_ne!(zobrist.player, 0);
    }
}
