/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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
use crate::moves::{Move};
use crate::scores::{is_mate_score, is_mated_score, sanitize_mate_score, sanitize_mated_score, clock_scaled_eval, sanitize_eval_score};
use std::mem::transmute;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crate::slices::SliceElementAccess;

pub const MAX_HASH_SIZE_MB: i32 = 8192 * 1024;

// Transposition table entry
// Bits 63 - 44: 20 highest bits of the hash
const HASHCHECK_MASK: u64 = 0b1111111111111111111100000000000000000000000000000000000000000000;

// Bits 43 - 12: Move + Score
const MOVE_BITSHIFT: u32 = 12;
const MOVE_MASK: u64 = 0b11111111111111111111111111111111;

// Bits 11 - 10: Clock bits
const CLOCK_BITSHIFT: u32 = 10;
const CLOCK_MASK: u64 = 0b11;

// Bits 9 - 8: Score Type
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum ScoreType {
    LowerBound = 0,
    Exact = 1,
    UpperBound = 2,
}

impl ScoreType {
    #[inline]
    pub fn from(code: u8) -> Self {
        unsafe { transmute(code) }
    }
}

const SCORE_TYPE_BITSHIFT: u32 = 8;
const SCORE_TYPE_MASK: u64 = 0b11;

// Bits 7 - 0: Depth
pub const MAX_DEPTH: usize = 255;
const DEPTH_MASK: u64 = 0b11111111;

const EVAL_SCORE_MASK: u64 = 0b111111111111111;
const EVAL_HASHCHECK_MASK: u64 = !EVAL_SCORE_MASK;


const SLOTS_PER_SEGMENT: usize = 8;

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
    pub fn write_entry(&self, hash: u64, ply: usize, new_depth: i32, m: Move, score: i16, typ: ScoreType, halfmove_clock: u8) {
        let index = self.calc_index(hash);
        let segment = self.segments.0.el(index);
        let hash_check = hash & HASHCHECK_MASK;

        let mut new_entry = hash_check;
        new_entry |= new_depth as u64;
        new_entry |= (typ as u64) << SCORE_TYPE_BITSHIFT;
        new_entry |= (m.with_score(from_root_relative_score(ply, score)).to_u32() as u64) << MOVE_BITSHIFT;

        let (slot_id, clock_bits) = calc_slot_id(hash, halfmove_clock);
        new_entry |= (clock_bits as u64) << CLOCK_BITSHIFT;

        let slot = segment.el(slot_id as usize);
        let entry = slot.load(Ordering::Relaxed);
        if entry & HASHCHECK_MASK == hash_check {
            if matches!(typ, ScoreType::Exact) || new_depth >= get_depth(entry) - 4 {
                slot.store(new_entry, Ordering::Relaxed);
            }
            return;
        }

        slot.store(new_entry, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn get_entry(&self, hash: u64, halfmove_clock: u8) -> Option<(u64, bool)> {
        let index = self.calc_index(hash);
        let slots = self.segments.0.el(index);
        let hash_check = hash & HASHCHECK_MASK;

        let (slot_id, clock_bits) = calc_slot_id(hash, halfmove_clock);
        let slot = slots.el(slot_id as usize);
        let entry = slot.load(Ordering::Relaxed);
        if entry & HASHCHECK_MASK == hash_check {
            return Some((entry, get_clock_bits(entry) == clock_bits));
        }

        slots.iter().skip(1).map(|s| s.load(Ordering::Relaxed)).find(|e| e & HASHCHECK_MASK == hash_check).map(|e| (e, false))
    }

    pub fn get_or_calc_eval<E: FnOnce() -> i16>(&self, hash: u64, halfmove_clock: u8, calc_eval: E, corr_eval: i16) -> i16 {
        let index = self.calc_index(hash);
        let slots = self.segments.0.el(index);
        let hash_check = hash & EVAL_HASHCHECK_MASK;
        let slot = slots.first().unwrap();
        let entry = slot.load(Ordering::Relaxed);
        if entry & EVAL_HASHCHECK_MASK == hash_check {
            return sanitize_eval_score(clock_scaled_eval(halfmove_clock, decode_score(entry).wrapping_add(corr_eval)) as i32) as i16;
        }

        let score = calc_eval();
        let entry = hash_check | encode_score(score);
        slot.store(entry, Ordering::Relaxed);

        sanitize_eval_score(clock_scaled_eval(halfmove_clock, score.wrapping_add(corr_eval)) as i32) as i16
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
    #[allow(unused_variables)]
    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_feature = "sse")]
        {
            use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let index = self.calc_index(hash);
            unsafe {
                _mm_prefetch::<{ _MM_HINT_T0 }>(self.segments.0.as_ptr().add(index) as *const i8);
            }
        }
        
        #[cfg(target_feature = "fp-armv8")]
        {
            use core::arch::aarch64::__prefetch;
            let index = self.calc_index(hash);
            unsafe {
                __prefetch(self.segments.0.as_ptr().add(index) as *const i8);
            }
        }

        #[cfg(all(not(target_feature = "sse"), not(target_feature = "fp-armv8")))]
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

fn get_clock_bits(entry: u64) -> u8 {
    ((entry >> CLOCK_BITSHIFT) & CLOCK_MASK) as u8
}

fn decode_score(entry: u64) -> i16 {
    if entry & 0b100000000000000 != 0 {
        -((entry & 0b011111111111111) as i16)
    } else {
        (entry & 0b011111111111111) as i16
    }
}

fn encode_score(score: i16) -> u64 {
    if score < 0 {
         0b100000000000000 | (-score as u64)
    } else {
        score as u64
    }
}

pub fn get_tt_move(entry: u64, ply: usize) -> Move {
    let m = Move::from_u32(((entry >> MOVE_BITSHIFT) & MOVE_MASK) as u32);
    let score = to_root_relative_score(ply, m.score());
    m.with_score(score)
}

#[inline]
// Convert current-node-relative mate scores to root-relative mate scores
fn to_root_relative_score(ply: usize, score: i16) -> i16 {
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
fn from_root_relative_score(ply: usize, score: i16) -> i16 {
    if is_mate_score(score) {
        sanitize_mate_score(score + ply as i16)
    } else if is_mated_score(score) {
        sanitize_mated_score(score - ply as i16)
    } else {
        score
    }
}

pub fn get_depth(entry: u64) -> i32 {
    (entry & DEPTH_MASK) as i32
}

pub fn get_score_type(entry: u64) -> ScoreType {
    ScoreType::from(((entry >> SCORE_TYPE_BITSHIFT) & SCORE_TYPE_MASK) as u8)
}

#[inline(always)]
fn calc_slot_id(hash: u64, clock: u8) -> (u8, u8) {
    let bits = (clock >> 2) & 0b11;

    let slot_base = (clock >> 2) >> 2;
    if hash & (0b1 << 40) == 0 {
        (slot_base + 1, bits)
    } else {
        (7 - slot_base, bits)
    }
}

#[cfg(test)]
mod tests {
    use crate::bitboard::BitBoards;
    use crate::colors::WHITE;
    use super::*;
    use crate::moves::{Move, MoveType, NO_MOVE};
    use crate::pieces::Q;
    use crate::scores::{MATE_SCORE, MATED_SCORE, MAX_EVAL, MIN_EVAL};

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

        tt.write_entry(hash, 0, depth,  m, score, typ, 0);

        let (entry, _) = tt.get_entry(hash, 0).expect("entry must exist");

        assert_eq!(m.to_u32(), get_tt_move(entry, 0).to_u32());
        assert_eq!(depth, get_depth(entry));
        assert_eq!(typ as u8, get_score_type(entry) as u8);
    }

    #[test]
    fn encodes_negative_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MIN_EVAL;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 0, 1, m, score, ScoreType::Exact, 0);

        let (entry, _) = tt.get_entry(hash, 0).expect("entry must exist");
        assert_eq!(m.to_u32(), get_tt_move(entry, 0).to_u32());
    }

    #[test]
    fn encodes_positive_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MAX_EVAL;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash,  0, 1, m, score, ScoreType::Exact, 0);

        let (entry, _) = tt.get_entry(hash, 0).expect("entry must exist");
        assert_eq!(m.to_u32(), get_tt_move(entry, 0).to_u32());
    }

    #[test]
    fn encodes_mated_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MATED_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash, 0, 1, m, score, ScoreType::Exact, 0);

        let (entry, _) = tt.get_entry(hash, 0).expect("entry must exist");
        assert_eq!(m.to_u32(), get_tt_move(entry, 0).to_u32());
    }

    #[test]
    fn encodes_mate_score_correctly() {
        let tt = TranspositionTable::new(1);
        let score = MATE_SCORE;
        let hash = u64::MAX;
        let m = NO_MOVE.with_score(score);
        tt.write_entry(hash,  0, 1, m, score, ScoreType::Exact, 0);

        let (entry, _) = tt.get_entry(hash, 0).expect("entry must exist");
        assert_eq!(m.to_u32(), get_tt_move(entry, 0).to_u32());
    }
}
