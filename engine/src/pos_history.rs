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
use crate::bitboard::BitBoard;
use crate::board::cycledetection;
use crate::zobrist::player_zobrist_key;

#[derive(Clone)]
pub struct PositionHistory {
    positions: Vec<Entry>,
    root: usize,
}

#[derive(Clone)]
struct Entry {
    hash: u64,
    is_after_root: bool,
    is_repetition: bool,
}

impl Default for PositionHistory {
    fn default() -> Self {
        PositionHistory { positions: Vec::with_capacity(16), root: 0 }
    }
}

impl PositionHistory {
    pub fn push(&mut self, hash: u64) {
        self.positions.push(Entry{hash, is_after_root: true, is_repetition: false});
    }

    pub fn pop(&mut self) {
        self.positions.pop();
    }

    pub fn is_repetition_draw(&self, hash: u64, halfmove_clock: u8) -> bool {
        for entry in self.positions.iter().rev().skip(1).step_by(2).take(halfmove_clock as usize / 2) {
            if entry.hash == hash && (entry.is_after_root || entry.is_repetition) {
                return true;
            }
        }

        false
    }

    pub fn has_upcoming_repetition(&self, occupancy: BitBoard, hash: u64, halfmove_clock: u8) -> bool {
        if halfmove_clock < 3 {
            return false;
        }
            
        let last_opp = self.positions.last().unwrap();
        let mut other = hash ^ last_opp.hash ^ player_zobrist_key();

        for (own, opp) in self.positions.iter().rev().skip(1).step_by(2)
            .zip(self.positions.iter().rev().skip(2).step_by(2)).take(halfmove_clock as usize / 2) {
            
            other ^= own.hash ^ opp.hash ^ player_zobrist_key();
            if other != 0 {
                continue;
            }

            if (opp.is_after_root || opp.is_repetition) && cycledetection::has_cycle_move(hash ^ opp.hash, occupancy) {
                return true;
            }
        }
        
        false
    }

    pub fn clear(&mut self) {
        self.positions.clear();
        self.root = 0;
    }

    pub fn mark_root(&mut self, halfmove_clock: u8) {
        self.root = self.positions.len();
        let mut existing = StackSet::new();
        for entry in self.positions.iter_mut().rev().take(halfmove_clock as usize) {
            entry.is_after_root = false;
            if !existing.insert(entry.hash) {
                entry.is_repetition = true;
            }
        }
    }
}

struct StackSet([u64; 256]);

impl StackSet {
    pub fn new() -> Self {
        StackSet([0; 256])
    }

    fn insert(&mut self, hash: u64) -> bool {
        let mut idx = (hash as usize) & 255;
        loop {
            let existing_value = self.0[idx];
            if existing_value == 0 {
                self.0[idx] = hash;
                return true;
            } else if existing_value == hash {
                return false;
            }
            idx = (idx + 1) & 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_single_repetition() {
        let mut history = PositionHistory::default();
        history.push(1000);
        history.push(1001);
        history.push(1002);
        history.push(1003);
        history.push(1004);

        assert!(!history.is_repetition_draw(1, 1));

        history.push(1);
        assert!(!history.is_repetition_draw(2, 2));

        history.push(2);
        assert!(history.is_repetition_draw(1, 3));
    }
}
