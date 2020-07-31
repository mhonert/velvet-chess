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

const MULTIPLIER: u64 = 6364136223846793005;
const INCREMENT: u64 = 1442695040888963407;

pub struct Random {
    state: u64
}

impl Random {
    pub fn new() -> Self {
        Random{state: 0x4d595df4d0f33173}
    }

    fn rand32(&mut self) -> u32 {
        let mut x = self.state;
        let count = (x >> 59) as u32;
        self.state = x.wrapping_mul(MULTIPLIER).wrapping_add(INCREMENT);
        x ^= x >> 18;

        ((x >> 27) as u32).rotate_right(count)
    }

    pub fn rand64(&mut self) -> u64 {
        ((self.rand32() as u64) << 32) | (self.rand32() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_evenly_distributed_random_numbers() {
        let mut rnd = Random::new();
        let mut number_counts: [i32; 6] = [0, 0, 0, 0, 0, 0];
        let iterations = 1_000_000;

        for i in 0..iterations {
            let number = (rnd.rand64() % 6) as i32;
            number_counts[number as usize] += 1;
        }

        let deviation_tolerance = (iterations as f64 * 0.001) as i32; // accept a low deviation from the "ideal" random distribution

        let ideal_distribution = iterations / 6;
        for i in 0..6 {
            let number_count = number_counts[i];
            let deviation_from_ideal = (ideal_distribution - number_count).abs();
            assert!(deviation_from_ideal < deviation_tolerance);
        }
    }
}
