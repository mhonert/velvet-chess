/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
use crate::moves::{TTPackedMove};
use crate::scores::{is_mate_score, is_mated_score, sanitize_mate_score, sanitize_mated_score};
use std::intrinsics::transmute;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const MAX_HASH_SIZE_MB: i32 = 512 * 1024;

// Transposition table entry
// Bits 63 - 40: 24 highest bits of the hash
const HASHCHECK_MASK: u64 = 0b1111111111111111111111110000000000000000000000000000000000000000;

// Bits 39 - 12: Move + Score
const MOVE_BITSHIFT: u32 = 12;
const MOVE_MASK: u64 = 0b1111111111111111111111111111;


// Bits 11 - 5: Depth
pub const MAX_DEPTH: usize = 127;
const DEPTH_BITSHIFT: u32 = 5;
const DEPTH_MASK: u64 = 0b1111111;

// Bits 4 - 3: Score Type
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

const SCORE_TYPE_BITSHIFT: u32 = 3;
const SCORE_TYPE_MASK: u64 = 0b11;

// Bits 2 - 0: Generation
pub const MAX_GENERATION: u16 = 7;
const GENERATION_MASK: u64 = 0b111;

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
        let index_bit_count = 63 - (segment_count | 1).leading_zeros();

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
    pub fn write_entry(&self, hash: u64, generation: u16, new_depth: i32, scored_move: TTPackedMove, typ: ScoreType) {
        let index = self.calc_index(hash);
        let segment = unsafe { self.segments.0.get_unchecked(index) };
        let hash_check = hash & HASHCHECK_MASK;

        let mut new_entry = hash_check;
        new_entry |= generation as u64;
        new_entry |= (new_depth as u64) << DEPTH_BITSHIFT;
        new_entry |= (typ as u64) << SCORE_TYPE_BITSHIFT;
        new_entry |= (scored_move.to_bits28() as u64) << MOVE_BITSHIFT;

        let mut target_slot = MaybeUninit::uninit();
        let mut lowest_sort_score = i16::MAX;
        for slot in segment.iter() {
            let entry = slot.load(Ordering::Relaxed);

            let age = get_age(entry, generation);
            let depth = get_depth(entry);
            let sort_score = if entry == 0 {
                i16::MIN
            } else if entry & HASHCHECK_MASK == hash_check {
                if matches!(typ, ScoreType::Exact) || new_depth >= get_depth(entry) - 3 {
                    i16::MIN
                } else {
                    depth as i16 - age as i16 * (MAX_DEPTH as i16 + 1)
                }

            } else {
                depth as i16 - age as i16 * (MAX_DEPTH as i16 + 1)
            };

            if sort_score < lowest_sort_score {
                lowest_sort_score = sort_score;
                target_slot = MaybeUninit::new(slot);
            }
        }

        unsafe { target_slot.assume_init() }.store(new_entry, Ordering::Relaxed);
    }

    pub fn get_entry(&self, hash: u64) -> (u64, Option<&AtomicU64>) {
        let index = self.calc_index(hash);
        let slots = unsafe { self.segments.0.get_unchecked(index) };
        let hash_check = hash & HASHCHECK_MASK;

        for value_slot in slots.iter() {
            let value_entry = value_slot.load(Ordering::Relaxed);
            if value_entry & HASHCHECK_MASK == hash_check {
                return (value_entry, Some(value_slot));
            }
        }

        (0, None)
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
            use core::arch::x86_64::{_MM_HINT_NTA, _mm_prefetch};
            let index = self.calc_index(hash);
            unsafe {
                _mm_prefetch::<{ _MM_HINT_NTA }>(self.segments.0.as_ptr().add(index) as *const i8);
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

pub fn update_generation(mut entry: u64, slot: &AtomicU64, generation: u16) {
    entry &= !(MAX_GENERATION as u64);
    entry |= generation as u64;
    slot.store(entry, Ordering::Relaxed);
}

pub fn get_hash_move(entry: u64) -> TTPackedMove {
    TTPackedMove::new(((entry >> MOVE_BITSHIFT) & MOVE_MASK) as u32)
}

#[inline]
// Convert current-node-relative mate scores to root-relative mate scores
pub fn to_root_relative_score(ply: i32, score: i16) -> i16 {
    if is_mate_score(score) {
        sanitize_mate_score(score - ply as i16)
    } else if is_mated_score(score) {
        sanitize_mated_score(score + ply as i16)
    } else {
        score
    }
}

#[inline]
// Convert root-relative mate scores to current-node-relative mate scores
pub fn from_root_relative_score(ply: i32, score: i16) -> i16 {
    if is_mate_score(score) {
        sanitize_mate_score(score + ply as i16)
    } else if is_mated_score(score) {
        sanitize_mated_score(score - ply as i16)
    } else {
        score
    }
}

pub fn get_depth(entry: u64) -> i32 {
    ((entry >> DEPTH_BITSHIFT) & DEPTH_MASK) as i32
}

pub fn is_lower_bound(entry: u64) -> bool {
    ((entry >> SCORE_TYPE_BITSHIFT) & SCORE_TYPE_MASK) == 0
}

pub fn get_age(entry: u64, curr_generation: u16) -> u16 {
    let entry_generation = (entry & GENERATION_MASK) as u16;
    calc_age(curr_generation, entry_generation)
}

fn calc_age(current: u16, previous: u16) -> u16 {
    current.wrapping_sub(previous) & GENERATION_MASK as u16
}

pub fn get_score_type(entry: u64) -> ScoreType {
    ScoreType::from(((entry >> SCORE_TYPE_BITSHIFT) & SCORE_TYPE_MASK) as u8)
}

#[cfg(test)]
mod tests {
    use crate::bitboard::BitBoards;
    use crate::colors::WHITE;
    use super::*;
    use crate::moves::{Move, MoveType, NO_MOVE};
    use crate::pieces::Q;
    use crate::scores::{MAX_SCORE, MIN_SCORE};

    #[test]
    fn writes_entry() {
        let tt = TranspositionTable::new(1);
        let hash = u64::MAX;
        let depth = MAX_DEPTH as i32;
        let score = -10;

        let m = Move::new(MoveType::QueenQuiet, 32, 33).with_score(score);
        let typ = ScoreType::Exact;

        let mut bitboards = BitBoards::default();
        bitboards.flip(WHITE, Q, 32);

        let tpm = m.to_tt_packed_move(WHITE, &bitboards);
        tt.write_entry(hash, 0, depth,  tpm, typ);

        let (entry, _) = tt.get_entry(hash);

        assert_eq!(tpm.to_bits28(), get_hash_move(entry).to_bits28());
        assert_eq!(depth, get_depth(entry));
        assert_eq!(typ as u8, get_score_type(entry) as u8);
    }

    #[test]
    fn encodes_negative_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MIN_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        let tpm = TTPackedMove::new(m.to_u32());
        tt.write_entry(hash, 0, 1, tpm, ScoreType::Exact);

        let (entry, _) = tt.get_entry(hash);
        assert_eq!(tpm.to_bits28(), get_hash_move(entry).to_bits28());
    }

    #[test]
    fn encodes_positive_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MAX_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        let tpm = TTPackedMove::new(m.to_u32());
        tt.write_entry(hash, 0, 1, tpm, ScoreType::Exact);

        let (entry, _) = tt.get_entry(hash);
        assert_eq!(tpm.to_bits28(), get_hash_move(entry).to_bits28());
    }


    #[test]
    fn calculates_age_from_generation_diff() {
        assert_eq!(7, MAX_GENERATION);
        assert_eq!(GENERATION_MASK, MAX_GENERATION as u64);

        assert_eq!(calc_age(0, 0), 0);
        assert_eq!(calc_age(1, 1), 0);
        assert_eq!(calc_age(2, 2), 0);
        assert_eq!(calc_age(3, 3), 0);
        assert_eq!(calc_age(4, 4), 0);
        assert_eq!(calc_age(5, 5), 0);
        assert_eq!(calc_age(6, 6), 0);
        assert_eq!(calc_age(7, 7), 0);

        assert_eq!(calc_age(1, 0), 1);
        assert_eq!(calc_age(2, 0), 2);
        assert_eq!(calc_age(3, 0), 3);
        assert_eq!(calc_age(4, 0), 4);
        assert_eq!(calc_age(5, 0), 5);
        assert_eq!(calc_age(6, 0), 6);
        assert_eq!(calc_age(7, 0), 7);

        assert_eq!(calc_age(2, 1), 1);
        assert_eq!(calc_age(3, 1), 2);
        assert_eq!(calc_age(0, 1), 7);

        assert_eq!(calc_age(3, 2), 1);
        assert_eq!(calc_age(0, 2), 6);
        assert_eq!(calc_age(1, 2), 7);

        assert_eq!(calc_age(0, 3), 5);
        assert_eq!(calc_age(1, 3), 6);
        assert_eq!(calc_age(2, 3), 7);
    }
}
