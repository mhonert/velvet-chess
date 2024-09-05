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
use crate::selfplay::Outcome;

#[derive(Copy, Clone, Default, Debug)]
pub struct PentanomialCount {
    pub ll: usize, // Loss-Loss
    pub ld: usize, // Loss-Draw or Draw-Loss
    pub d2: usize, // Draw-Draw or Win-Loss/Loss-Win
    pub wd: usize, // Win-Draw or Draw-Win
    pub ww: usize, // Win-Win
}

impl PentanomialCount {
    pub fn add(&mut self, game_pair: (Outcome, Outcome)) {
        match game_pair {
            (Outcome::Win, Outcome::Win) => self.ww += 1,
            (Outcome::Win, Outcome::Loss) => self.d2 += 1,
            (Outcome::Win, Outcome::Draw) => self.wd += 1,
            (Outcome::Loss, Outcome::Win) => self.d2 += 1,
            (Outcome::Loss, Outcome::Loss) => self.ll += 1,
            (Outcome::Loss, Outcome::Draw) => self.ld += 1,
            (Outcome::Draw, Outcome::Win) => self.wd += 1,
            (Outcome::Draw, Outcome::Loss) => self.ld += 1,
            (Outcome::Draw, Outcome::Draw) => self.d2 += 1,
        }
    }
    
    pub fn clear(&mut self) {
        self.ll = 0;
        self.ld = 0;
        self.d2 = 0;
        self.wd = 0;
        self.ww = 0;
    }

    pub fn add_all(&mut self, counts: PentanomialCount) {
        self.ll += counts.ll;
        self.ld += counts.ld;
        self.d2 += counts.d2;
        self.wd += counts.wd;
        self.ww += counts.ww;
    }
    
    pub fn score(&self) -> f64 {
        self.ld as f64 * 0.25 + self.d2 as f64 * 0.5 + self.wd as f64 * 0.75 + self.ww as f64
    }

    pub fn total(&self) -> usize {
        self.ll + self.ld + self.d2 + self.wd + self.ww
    }

    pub fn gradient(&self) -> f64 {
        let total = self.total() as f64;
        (self.score() / total - 0.5) * 2.0
    }
}


#[derive(Default)]
pub struct PentanomialModel {
    pub ll: f64, // Loss-Loss
    pub ld: f64, // Loss-Draw or Draw-Loss
    pub d2: f64, // Draw-Draw or Win-Loss/Loss-Win
    pub wd: f64, // Win-Draw or Draw-Win
    pub ww: f64, // Win-Win
}

impl PentanomialModel {
    pub fn score(&self) -> f64 {
        self.ld * 0.25 + self.d2 * 0.5 + self.wd * 0.75 + self.ww
    }

    pub fn deviation(&self, score: f64) -> PentanomialModel {
        PentanomialModel {
            ll: self.ll * (0.0 - score).powi(2),
            ld: self.ld * (0.25 - score).powi(2),
            d2: self.d2 * (0.5 - score).powi(2),
            wd: self.wd * (0.75 - score).powi(2),
            ww: self.ww * (1.0 - score).powi(2),
        }
    }

    pub fn total(&self) -> f64 {
        self.ll + self.ld + self.d2 + self.wd + self.ww
    }
}

impl From<PentanomialCount> for PentanomialModel {
    fn from(counts: PentanomialCount) -> Self {
        let total = counts.total() as f64;
        PentanomialModel {
            ll: counts.ll as f64 / total,
            ld: counts.ld as f64 / total,
            d2: counts.d2 as f64 / total,
            wd: counts.wd as f64 / total,
            ww: counts.ww as f64 / total,
        }
    }
}