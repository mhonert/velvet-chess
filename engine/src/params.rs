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
    nmp_base = 3
    nmp_divider = 3
    se_double_ext_margin = 4
    se_double_ext_limit = 12
    prob_cut_margin = 150
    prob_cut_depth = 4
    
    nmp_enabled = 1
    razoring_enabled = 1
    rfp_enabled = 1
    prob_cut_enabled = 1
    fp_enabled = 1
    se_enabled = 1
);

tunable_array_params!(
    lmp_improving = [0, 4, 7]
    lmp_not_improving = [0, 2, 3]
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
    print_array_options();
}

pub fn calc_node_limit_from_elo(elo: i32) -> u64 {
    let elo_normalized = (elo.clamp(1225, 3000) - 1225) / 25;
    STRENGTH_NODE_LIMITS[elo_normalized as usize] as u64
}
