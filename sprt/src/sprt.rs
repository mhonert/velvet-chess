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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use selfplay::pentanomial::{PentanomialCount, PentanomialModel};
use crate::{SPRT_ALPHA, SPRT_BETA};

#[derive(Default)]
pub struct SprtState {
    ll: AtomicUsize,
    ld: AtomicUsize,
    d2: AtomicUsize,
    wd: AtomicUsize,
    ww: AtomicUsize,
    avg_depths: AtomicUsize,
    depth_counts: AtomicUsize,
    time_losses: AtomicUsize,
    stopped: AtomicBool,
    elo0: f64,
    elo1: f64,
    upper_bound: f64,
    lower_bound: f64,
}

const STD_NORM_DEV_95: f64 = 1.959963984540054;

impl SprtState {
    pub fn new(elo0: f64, elo1: f64) -> SprtState {
        SprtState {
            ll: AtomicUsize::new(0),
            ld: AtomicUsize::new(0),
            d2: AtomicUsize::new(0),
            wd: AtomicUsize::new(0),
            ww: AtomicUsize::new(0),
            avg_depths: AtomicUsize::new(0),
            depth_counts: AtomicUsize::new(0),
            time_losses: AtomicUsize::new(0),
            stopped: AtomicBool::new(false),
            elo0,
            elo1,
            upper_bound: ((1.0 - SPRT_BETA) / SPRT_ALPHA).ln(),
            lower_bound: (SPRT_BETA / (1.0 - SPRT_ALPHA)).ln(),
        }
    }

    pub fn update(&self, counts: &PentanomialCount, add_avg_depth: usize, time_losses: usize) -> bool {
        if self.stopped() {
            return true;
        }

        let total_time_losses = self.time_losses.fetch_add(time_losses, Ordering::Relaxed) + time_losses;

        let total_ll = self.ll.fetch_add(counts.ll, Ordering::Relaxed) + counts.ll;
        let total_ld = self.ld.fetch_add(counts.ld, Ordering::Relaxed) + counts.ld;
        let total_d2 = self.d2.fetch_add(counts.d2, Ordering::Relaxed) + counts.d2;
        let total_wd = self.wd.fetch_add(counts.wd, Ordering::Relaxed) + counts.wd;
        let total_ww = self.ww.fetch_add(counts.ww, Ordering::Relaxed) + counts.ww;

        let total = total_ll + total_ld + total_d2 + total_wd + total_ww;

        let p = PentanomialModel {
            ll: total_ll as f64 / total as f64,
            ld: total_ld as f64 / total as f64,
            d2: total_d2 as f64 / total as f64,
            wd: total_wd as f64 / total as f64,
            ww: total_ww as f64 / total as f64,
        };

        let score = p.score();
        let deviation = p.deviation(score);
        let variance = deviation.total();
        let pair_variance = variance / total as f64;

        let upper_bound = score + STD_NORM_DEV_95 * pair_variance.sqrt();
        let lower_bound = score - STD_NORM_DEV_95 * pair_variance.sqrt();

        let draw_ratio = p.d2;

        let elo_diff = score_to_norm_elo(score, variance);
        let elo_error = (score_to_norm_elo(upper_bound, variance) - score_to_norm_elo(lower_bound, variance)) / 2.0;
        
        let llr = self.llr(total as f64, variance, &p);

        let avg_depths = self.avg_depths.fetch_add(add_avg_depth, Ordering::Relaxed) + add_avg_depth;
        let depth_counts = self.depth_counts.fetch_add(1, Ordering::Relaxed) + 1;
        let avg_depth = avg_depths / depth_counts;

        println!("Norm. Elo: {:>6.2} (+/- {:>5.2}) / Draw ratio: {:>5.2}% / Avg. depth {:2} / Time losses: {:>5.3}% / Game pairs: {} / LLR: {:>5.2}",
                 elo_diff, elo_error, draw_ratio * 100.0, avg_depth, ((total_time_losses * 100000) as f64 / total as f64) / 1000.0, total, llr);

        let mut stopped = false;
        if llr >= self.upper_bound {
            println!("----------------------------------------------");
            println!("> SPRT: Stopped (upper bound) -> H0 accepted ->");
            stopped = true;
        } else if llr <= self.lower_bound {
            println!("----------------------------------------------");
            println!("> SPRT: Stopped (lower bound) -> H1 accepted");
            stopped = true;
        }

        if stopped {
            self.stopped.store(stopped, Ordering::Relaxed);
        }
        stopped
    }

    fn llr(&self, total: f64, variance: f64, p: &PentanomialModel) -> f64 {
        if variance == 0.0 {
            return 0.0;
        }
        let score0 = norm_elo_to_score(self.elo0, variance);
        let score1 = norm_elo_to_score(self.elo1, variance);

        let deviation0 = p.deviation(score0);
        let variance0 = deviation0.total();

        let deviation1 = p.deviation(score1);
        let variance1 = deviation1.total();

        if variance0 == 0.0 || variance1 == 0.0 {
            return 0.0;
        }

        total * 0.5 * (variance0 / variance1).ln()
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }
}

fn score_to_norm_elo(score: f64, variance: f64) -> f64 {
    (score - 0.5) / (variance * 2.0).sqrt() * (400.0 / 10.0f64.ln())
}

fn norm_elo_to_score(nelo: f64, variance: f64) -> f64 {
    nelo * variance.sqrt() / (400.0 / 10.0f64.ln()) + 0.5
}
