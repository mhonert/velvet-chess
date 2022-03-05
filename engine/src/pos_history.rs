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

#[derive(Clone)]
pub struct PositionHistory(Vec<u64>);

impl Default for PositionHistory {
    fn default() -> Self {
        PositionHistory(Vec::with_capacity(16))
    }
}

impl PositionHistory {
    pub fn push(&mut self, hash: u64) {
        self.0.push(hash);
    }

    pub fn pop(&mut self) {
        self.0.pop();
    }

    pub fn is_repetition_draw(&self, hash: u64, halfmove_clock: u8) -> bool {
        self.0.iter().rev().skip(1).step_by(2).take(halfmove_clock as usize / 2).any(|pos| *pos == hash)
    }

    pub fn clear(&mut self) {
        self.0.clear();
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
