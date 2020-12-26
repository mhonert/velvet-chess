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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the * GNU General Public License for more details. *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

use crate::move_gen::{Move, NO_MOVE};

pub const MAX_HASH_SIZE_MB: i32 = 4096;

// Transposition table entry
// Bits 63 - 23: 41 highest bits of the hash
const HASHCHECK_MASK: u64 = 0b1111111111111111111111111111111111111111100000000000000000000000;

// Bits 22 - 15: Depth
pub const MAX_DEPTH: usize = 255;
const DEPTH_BITSHIFT: i32 = 15;
const DEPTH_MASK: u64 = 0b11111111;

// Bits 14 - 13: Score Type
pub const EXACT: u8 = 0;
pub const UPPER_BOUND: u8 = 1;
pub const LOWER_BOUND: u8 = 2;

const SCORE_TYPE_BITSHIFT: u32 = 13;
const SCORE_TYPE_MASK: u64 = 0b11;

// Bits 12 - 0: Age
const AGE_MASK: u64 = 0b1111111111111;

pub const DEFAULT_SIZE_MB: u64 = 32;
const PER_ENTRY_BYTE_SIZE: u64 = 8 + 4;

pub struct TranspositionTable {
    index_mask: u64,
    entries: Vec<u64>,
    moves: Vec<Move>,
    age: i32,
}

impl TranspositionTable {
    pub fn new(size_mb: u64) -> Self {
        let mut tt = TranspositionTable {
            index_mask: 0,
            entries: Vec::new(),
            moves: Vec::new(),
            age: 0,
        };

        tt.resize(size_mb, true);

        tt
    }

    pub fn resize(&mut self, size_mb: u64, initialize: bool) {
        // Calculate table size as close to the desired sizeInMB as possible, but never above it
        let size_bytes = size_mb * 1_048_576;
        let entry_count = size_bytes / PER_ENTRY_BYTE_SIZE;
        let index_bit_count = 31 - (entry_count as u32 | 1).leading_zeros();

        let size = (1u64 << index_bit_count) as usize;
        if initialize || size != self.entries.len() {
            self.index_mask = (size as u64) - 1;

            self.entries.resize(size, 0);
            self.moves.resize(size, NO_MOVE);

            self.entries.shrink_to_fit();
            self.moves.shrink_to_fit();
        }
    }

    pub fn increase_age(&mut self) {
        self.age = (self.age + 1) & AGE_MASK as i32;
    }

    pub fn write_entry(&mut self, hash: u64, depth: i32, scored_move: Move, typ: u8) {
        let index = self.calc_index(hash);

        let entry = unsafe { self.entries.get_unchecked_mut(index) };

        if *entry != 0
            && (*entry & AGE_MASK) as i32 == self.age
            && depth < ((*entry >> DEPTH_BITSHIFT) & DEPTH_MASK) as i32
        {
            return;
        }

        let mut new_entry: u64 = hash & HASHCHECK_MASK;
        new_entry |= (depth << DEPTH_BITSHIFT) as u64;
        new_entry |= (typ as u64) << SCORE_TYPE_BITSHIFT;
        new_entry |= self.age as u64;

        *entry = new_entry;
        unsafe { *self.moves.get_unchecked_mut(index) = scored_move };
    }

    pub fn get_entry(&self, hash: u64) -> u64 {
        let index = self.calc_index(hash);

        let entry = unsafe { *self.entries.get_unchecked(index) };
        let age_diff = self.age - (entry & AGE_MASK) as i32;

        if entry == 0
            || age_diff < 0
            || age_diff > 1
            || (entry & HASHCHECK_MASK) != (hash & HASHCHECK_MASK)
        {
            return 0;
        }
        (unsafe { *self.moves.get_unchecked(index) }.to_u32() as u64) << 32 | (entry & !HASHCHECK_MASK)
    }

    pub fn get_age_diff(&self, entry: u64) -> i32 {
        self.age - (entry & AGE_MASK) as i32
    }

    fn calc_index(&self, hash: u64) -> usize {
        (hash & self.index_mask) as usize
    }

    pub fn clear(&mut self) {
        for i in 0..self.entries.len() {
            self.entries[i] = 0;
            self.moves[i] = NO_MOVE;
        }
        self.age = 0;
    }
}

pub fn get_scored_move(entry: u64) -> Move {
    Move::from_u32((entry >> 32) as u32)
}

pub fn get_depth(entry: u64) -> i32 {
    ((entry >> DEPTH_BITSHIFT) & DEPTH_MASK) as i32
}

pub fn get_score_type(entry: u64) -> u8 {
    ((entry >> SCORE_TYPE_BITSHIFT) & SCORE_TYPE_MASK) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_entry() {
        let mut tt = TranspositionTable::new(1);
        let hash = u64::max_value();
        let depth = MAX_DEPTH as i32;
        let score = -10;

        let m = Move::new(5, 32, 33).with_score(score);
        let typ = EXACT;

        tt.write_entry(hash, depth, m, typ);

        let entry = tt.get_entry(hash);

        assert_eq!(m, get_scored_move(entry));
        assert_eq!(depth, get_depth(entry));
        assert_eq!(typ, get_score_type(entry));
    }

    #[test]
    fn encodes_negative_score_correctly() {
        let mut tt = TranspositionTable::new(1);
        let score = -16383;
        let hash = u64::max_value();
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, EXACT);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_scored_move(entry).score());
    }

    #[test]
    fn encodes_positive_score_correctly() {
        let mut tt = TranspositionTable::new(1);
        let score = 16383;
        let hash = u64::max_value();
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, EXACT);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_scored_move(entry).score());
    }
}
