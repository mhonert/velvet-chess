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

pub struct PositionHistory {
    positions: [u64; 1024],
    index: usize
}

impl PositionHistory {
    pub fn new() -> Self {
        PositionHistory{positions: [0; 1024], index: 0}
    }

    pub fn push(&mut self, hash: u64) {
        self.positions[self.index] = hash;
        self.index += 1;
    }

    pub fn pop(&mut self) {
        self.index -= 1;
    }

    pub fn is_threefold_repetition(&self) -> bool {
        if self.index <= 2 {
            return false;
        }

        let hash = self.positions[self.index - 1];

        let mut count: i32 = 0;
        for i in 0..self.index - 1 {
            if self.positions[i] == hash {
                count += 1;
                if count == 2 {
                    return true;
                }
            }
        }

        false
    }

    pub fn is_single_repetition(&self) -> bool {
        if self.index <= 1 {
            return false;
        }

        let hash = self.positions[self.index - 1];
        for i in 0..self.index - 1 {
            if self.positions[i] == hash {
                return true;
            }
        }

        false
    }

    pub fn clear(&mut self) {
        self.index = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_threefold_repetition() {
        let mut history = PositionHistory::new();
        history.push(1);
        history.push(1);
        assert!(!history.is_threefold_repetition());

        history.push(1);
        assert!(history.is_threefold_repetition());
    }

    #[test]
    fn detects_single_repetition() {
        let mut history = PositionHistory::new();
        history.push(1);
        assert!(!history.is_single_repetition());

        history.push(2);
        assert!(!history.is_single_repetition());

        history.push(1);
        assert!(history.is_threefold_repetition());
    }
}
