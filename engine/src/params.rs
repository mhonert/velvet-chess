/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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
    fp_base_margin = 23
    fp_margin_multiplier = 42
    razor_margin_multiplier = 200
    rfp_base_margin_improving = 19
    rfp_base_margin_not_improving = 23
    rfp_margin_improving_multiplier = 57
    rfp_margin_not_improving_multiplier = 70

    qs_see_threshold = 111
);

tunable_array_params!(
    see_piece_values = [0, 98, 349, 350, 523, 1016, 8000]
    lmp_improving = [0, 4, 7, 12, 19, 28, 37, 50, 64]
    lmp_not_improving = [0, 2, 3, 6, 9, 14, 20, 27, 35]
);

#[cfg(not(feature = "tune"))]
pub fn set(name: String, _value: i32) {
    println!("Unknown option: {}", name)
}

#[cfg(not(feature = "tune"))]
pub fn print_options() {}

#[cfg(feature = "tune")]
pub fn set(name: String, value: i32) {
    if !set_param(name.clone(), value) && !set_array_param(name.clone(), value) {
        println!("Unknown option: {}", name);
        return;
    }
    println!("debug set {} = {}", name, value);
}

#[cfg(feature = "tune")]
pub fn print_options() {
    print_single_options();
    print_array_options();
}
