/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use crate::align::A64;
use crate::moves::Move;
use crate::scores::{MATED_SCORE, MATE_SCORE};
use std::intrinsics::transmute;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const MAX_HASH_SIZE_MB: i32 = 256 * 1024;

// Transposition table entry
// Bits 63 - 39: 25 highest bits of the hash
const HASHCHECK_MASK: u64 = 0b1111111111111111111111111000000000000000000000000000000000000000;

// Bits 38 - 32: Depth
pub const MAX_DEPTH: usize = 127;
const DEPTH_BITSHIFT: u32 = 32;
const DEPTH_MASK: u64 = 0b1111111;

// Bits 31 - 30: Score Type
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum ScoreType {
    Exact = 0,
    UpperBound = 1,
    LowerBound = 2,
}

impl ScoreType {
    #[inline]
    pub fn from(code: u8) -> Self {
        unsafe { transmute(code) }
    }
}

const SCORE_TYPE_BITSHIFT: u32 = 30;
const SCORE_TYPE_MASK: u64 = 0b11;

const SLOTS_PER_SEGMENT: usize = 4;

pub const DEFAULT_SIZE_MB: u64 = 32;
const SEGMENT_BYTE_SIZE: u64 = (64 / 8) * SLOTS_PER_SEGMENT as u64;

pub struct TranspositionTable {
    index_mask: u64,
    segments: A64<Vec<[AtomicU64; SLOTS_PER_SEGMENT]>>,
}

impl TranspositionTable {
    pub fn new(size_mb: u64) -> Arc<Self> {
        let mut tt = Arc::new(TranspositionTable { index_mask: 0, segments: A64(Vec::with_capacity(0)) });

        Arc::get_mut(&mut tt).unwrap().resize(size_mb);
        TranspositionTable::clear(&tt, 0, 1);

        tt
    }

    pub fn resize(&mut self, size_mb: u64) {
        // Calculate table size as close to the desired size_mb as possible, but never above it
        let size_bytes = size_mb * 1_048_576;
        let segment_count = size_bytes / SEGMENT_BYTE_SIZE;
        let index_bit_count = 63 - (segment_count as u64 | 1).leading_zeros();

        let size = (1u64 << index_bit_count) as usize;
        self.index_mask = (size as u64) - 1;

        if size > self.segments.0.len() {
            self.segments.0.reserve_exact(size - self.segments.0.len());
            unsafe { self.segments.0.set_len(size) };
        } else {
            self.segments.0.truncate(size);
            self.segments.0.shrink_to_fit();
        }
    }

    // Important: mate scores must be stored relative to the current node, not relative to the root node
    pub fn write_entry(&self, hash: u64, new_depth: i32, scored_move: Move, typ: ScoreType) {
        let index = self.calc_index(hash);
        let segment = unsafe { self.segments.0.get_unchecked(index) };

        for slot in segment.iter() {
            let existing_entry = slot.load(Ordering::Relaxed) ^ hash;
            if existing_entry & HASHCHECK_MASK == 0 {
                slot.store(0, Ordering::Relaxed);
            }
        }

        let mut new_entry = hash;
        new_entry ^= (new_depth as u64) << DEPTH_BITSHIFT;
        new_entry ^= (typ as u64) << SCORE_TYPE_BITSHIFT;
        new_entry ^= scored_move.to_bit29() as u64;

        segment[new_depth as usize & (SLOTS_PER_SEGMENT - 1)].store(new_entry, Ordering::Relaxed);
    }

    pub fn get_entry(&self, hash: u64) -> u64 {
        let index = self.calc_index(hash);
        let slots = unsafe { self.segments.0.get_unchecked(index) };

        for slot in slots.iter() {
            let entry = slot.load(Ordering::Relaxed) ^ hash;
            if (entry & HASHCHECK_MASK) == 0 {
                return entry;
            }
        }

        0
    }

    fn calc_index(&self, hash: u64) -> usize {
        (hash & self.index_mask) as usize
    }

    pub fn clear(&self, thread_no: usize, total_threads: usize) {
        let chunk_size = (self.segments.0.len() + total_threads - 1) / total_threads;

        for segment in self.segments.0.chunks(chunk_size).skip(thread_no).take(1).last().unwrap().iter() {
            for entry in segment.iter() {
                entry.store(0, Ordering::Relaxed);
            }
        }
    }

    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_feature = "sse")]
        {
            let index = self.calc_index(hash);
            unsafe {
                core::arch::x86_64::_mm_prefetch::<0>(self.segments.0.as_ptr().add(index) as *const i8);
            }
        }

        #[cfg(not(target_feature = "sse"))]
        {
            // No op
        }
    }

    // hash_full returns an approximation of the fill level in per mill
    pub fn hash_full(&self) -> usize {
        self.segments
            .0
            .iter()
            .take(1024 / SLOTS_PER_SEGMENT)
            .flat_map(|entries| entries.iter())
            .filter(|entry| entry.load(Ordering::Relaxed) != 0)
            .count()
            * 1000
            / 1024
    }
}

pub fn get_untyped_move(entry: u64) -> Move {
    Move::from_bit29((entry & 0b00011111111111111111111111111111) as u32)
}

#[inline]
// Convert current-node-relative mate scores to root-relative mate scores
pub fn to_root_relative_score(ply: i32, score: i32) -> i32 {
    if score <= MATED_SCORE + MAX_DEPTH as i32 {
        score + ply
    } else if score >= MATE_SCORE - MAX_DEPTH as i32 {
        score - ply
    } else {
        score
    }
}

#[inline]
// Convert root-relative mate scores to current-node-relative mate scores
pub fn from_root_relative_score(ply: i32, score: i32) -> i32 {
    if score <= MATED_SCORE + MAX_DEPTH as i32 {
        score - ply
    } else if score >= MATE_SCORE - MAX_DEPTH as i32 {
        score + ply
    } else {
        score
    }
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
    use crate::moves::{MoveType, NO_MOVE};
    use crate::scores::{MAX_SCORE, MIN_SCORE};

    #[test]
    fn writes_entry() {
        let tt = TranspositionTable::new(1);
        let hash = u64::MAX;
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
        let tt = TranspositionTable::new(1);
        let score = MIN_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, ScoreType::Exact);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_untyped_move(entry).score());
    }

    #[test]
    fn encodes_positive_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MAX_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 1, m, ScoreType::Exact);

        let entry = tt.get_entry(hash);
        assert_eq!(score, get_untyped_move(entry).score());
    }
}
