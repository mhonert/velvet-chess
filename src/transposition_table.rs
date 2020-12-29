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

use crate::moves::{Move};
use std::intrinsics::transmute;

pub const MAX_HASH_SIZE_MB: i32 = 4096;

// Transposition table entry
// Bits 63 - 38: 26 highest bits of the hash
const HASHCHECK_MASK: u64 = 0b1111111111111111111111111100000000000000000000000000000000000000;

// Bits 37 - 32: Depth
pub const MAX_DEPTH: usize = 63;
const DEPTH_BITSHIFT: u32 = 32;
const DEPTH_MASK: u64 = 0b111111;

// Bits 31 - 30: Score Type
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum ScoreType {
    Exact = 0,
    UpperBound = 1,
    LowerBound = 2
}

impl ScoreType {
    #[inline]
    pub fn from(code: u8) -> Self {
        unsafe {
            transmute(code)
        }
    }
}

const SCORE_TYPE_BITSHIFT: u32 = 30;
const SCORE_TYPE_MASK: u64 = 0b11;

pub const DEFAULT_SIZE_MB: u64 = 32;
const PER_ENTRY_BYTE_SIZE: u64 = 8;

pub struct TranspositionTable {
    index_mask: u64,
    entries: Vec<u64>
}

impl TranspositionTable {
    pub fn new(size_mb: u64) -> Self {
        let mut tt = TranspositionTable {
            index_mask: 0,
            entries: Vec::new(),
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
            self.entries.shrink_to_fit();
        }
    }

    pub fn write_entry(&mut self, hash: u64, depth: i32, scored_move: Move, typ: ScoreType) {
        let mut new_entry: u64 = hash & HASHCHECK_MASK;
        new_entry |= (depth as u64) << DEPTH_BITSHIFT;
        new_entry |= (typ as u64) << SCORE_TYPE_BITSHIFT;
        new_entry |= scored_move.to_bit29() as u64;

        let index = self.calc_index(hash);
        unsafe { *self.entries.get_unchecked_mut(index) = new_entry };
    }

    pub fn get_entry(&mut self, hash: u64) -> u64 {
        let index = self.calc_index(hash);

        let entry = unsafe { *self.entries.get_unchecked(index) };
        if entry == 0 || (entry & HASHCHECK_MASK) != (hash & HASHCHECK_MASK) {
            return 0;
        }

        entry
    }

    fn calc_index(&self, hash: u64) -> usize {
        (hash & self.index_mask) as usize
    }

    pub fn clear(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = 0;
        }
    }
}

pub fn get_untyped_move(entry: u64) -> Move {
    Move::from_bit29((entry & 0b00011111111111111111111111111111) as u32)
}

pub fn get_depth(entry: u64) -> i32 {
    ((entry >> DEPTH_BITSHIFT) & DEPTH_MASK) as i32
}

pub fn get_score_type(entry: u64) -> ScoreType {
    ScoreType::from(((entry >> SCORE_TYPE_BITSHIFT) & SCORE_TYPE_MASK) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score_util::{MIN_SCORE, MAX_SCORE};
    use crate::moves::{MoveType, NO_MOVE};

    #[test]
    fn writes_entry() {
        let mut tt = TranspositionTable::new(1);
        let hash = u64::max_value();
        let depth = MAX_DEPTH as i32;
        let score = -10;

        let m = Move::new(MoveType::Quiet, 5, 32, 33).with_score(score);
        let typ = ScoreType::Exact;

        tt.write_entry(hash, depth, m, typ);

        let entry = tt.get_entry(hash);

        assert_eq!(m.to_bit29(), get_untyped_move(entry).to_bit29());
        assert_eq!(depth, get_depth(entry));
        assert_eq!(typ as u8, get_score_type(entry) as u8);
    }

    #[test]
    fn encodes_negative_score_correctly() {
        let mut tt = TranspositionTable::new(1);
        let score = MIN_SCORE;
        let hash = u64::max_value();
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, ScoreType::Exact);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_untyped_move(entry).score());
    }

    #[test]
    fn encodes_positive_score_correctly() {
        let mut tt = TranspositionTable::new(1);
        let score = MAX_SCORE;
        let hash = u64::max_value();
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, ScoreType::Exact);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_untyped_move(entry).score());
    }
}
