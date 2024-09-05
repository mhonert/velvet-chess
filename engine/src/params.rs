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


#[macro_use]
mod macros;

#[allow(unused)]
use std::str::FromStr;

tunable_params!(
    fp_base_margin = 17
    fp_margin_multiplier = 22
    
    razor_margin_multiplier = 200
    
    rfp_base_margin_improving = 19
    rfp_margin_multiplier_improving = 22
    rfp_base_margin_not_improving = 23
    rfp_margin_multiplier_not_improving = 28
    
    nmp_base = 768
    nmp_divider = 672
    
    se_double_ext_margin = 4
    se_double_ext_limit = 12
    
    prob_cut_margin = 150
    prob_cut_depth = 4
    
    lmr_base = 256
    lmr_divider = 1024

    lmp_max_depth = 4
    lmp_improving_base = 3
    lmp_improving_multiplier = 65
    lmp_not_improving_base = 2
    lmp_not_improving_multiplier = 35

    nmp_enabled = 1
    razoring_enabled = 1
    rfp_enabled = 1
    prob_cut_enabled = 1
    fp_enabled = 1
    se_enabled = 1
);

derived_array_params!(
    lmr: [MAX_LMR_MOVES] = calc_late_move_reductions
);

static STRENGTH_NODE_LIMITS: [u16; 72] = [
    2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 18, 21, 25, 31, 40, 59, 102, 149, 196, 239, 276, 318, 354, 394,
    438, 486, 539, 597, 661, 731, 826, 933, 1053, 1188, 1340, 1492, 1661, 1825, 2005, 2202, 2418,
    2678, 2965, 3282, 3632, 4019, 4447, 4920, 5443, 6021, 6660, 7290, 7979, 8733, 9558, 10460, 11447,
    12527, 13708, 15000, 16413, 17959, 19435, 21032, 22760, 24629, 26651, 28839, 31527, 34465, 37676,
];


#[cfg(not(feature = "tune"))]
pub fn print_options() {}

#[cfg(feature = "tune")]
pub fn print_options() {
    print_single_options();
}

pub fn calc_node_limit_from_elo(elo: i32) -> u64 {
    let elo_normalized = (elo.clamp(1225, 3000) - 1225) / 25;
    STRENGTH_NODE_LIMITS[elo_normalized as usize] as u64
}

const MAX_LMR_MOVES: usize = 64;

pub fn lmr_idx(moves: i16) -> usize {
    (moves as usize).min(MAX_LMR_MOVES - 1)
}

fn calc_late_move_reductions(params: &SingleParams) -> [i16; MAX_LMR_MOVES] {
    let mut lmr = [0i16; MAX_LMR_MOVES];
    for moves in 1..MAX_LMR_MOVES {
        lmr[lmr_idx(moves as i16)] = (from_fp(params.lmr_base()) + (moves as f64) / from_fp(params.lmr_divider())).log2() as i16;
    }

    lmr
}

impl SingleParams {
    #[inline]
    pub fn lmp(&self, improving: bool, depth: i32) -> i32 {
        if improving {
            (depth * depth + self.lmp_improving_base() as i32) * self.lmp_improving_multiplier() as i32 / 64
        } else {
            (depth * depth + self.lmp_not_improving_base() as i32) * self.lmp_not_improving_multiplier() as i32 / 64
        }
    }
}


// Convert a fixed point value to a floating point value.
fn from_fp(fp: i16) -> f64 {
    (fp as f64) / 256.0
}
