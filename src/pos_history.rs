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

use std::cmp::min;

pub struct PositionHistory {
    positions: [u64; 1024],
    index: usize,
}

impl PositionHistory {
    pub fn new() -> Self {
        PositionHistory {
            positions: [0; 1024],
            index: 0,
        }
    }

    pub fn push(&mut self, hash: u64) {
        unsafe { *self.positions.get_unchecked_mut(self.index) = hash };
        self.index += 1;
    }

    pub fn pop(&mut self) {
        self.index -= 1;
    }

    pub fn is_single_repetition(&self, halfmove_clock: u8) -> bool {
        let mut pos = self.index - 1;
        let hash = unsafe { *self.positions.get_unchecked(pos) };

        let earliest_index = self.index + 2 - (min(self.index, halfmove_clock as usize));
        while pos >= earliest_index {
            pos -= 2;
            if unsafe { *self.positions.get_unchecked(pos) } == hash {
                return true;
            }
        }

        false
    }

    pub fn is_checked_single_repetition(&self, halfmove_clock: u8) -> bool {
         self.index > 0 && self.is_single_repetition(halfmove_clock)
    }

    pub fn clear(&mut self) {
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_single_repetition() {

        let mut history = PositionHistory::new();
        history.push(1000);
        history.push(1001);
        history.push(1002);
        history.push(1003);
        history.push(1004);

        history.push(1);
        assert!(!history.is_single_repetition(1));

        history.push(2);
        assert!(!history.is_single_repetition(2));

        history.push(1);
        assert!(history.is_single_repetition(3));
    }
}
