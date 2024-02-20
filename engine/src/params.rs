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
);

tunable_array_params!(
    see_piece_values = [0, 98, 349, 350, 523, 1016, 8000]
    lmp_improving = [0, 4, 7]
    lmp_not_improving = [0, 2, 3]
);

#[cfg(not(feature = "tune"))]
pub fn set(name: String, _value: i16) {
    println!("Unknown option: {}", name)
}

#[cfg(not(feature = "tune"))]
pub fn print_options() {}

#[cfg(feature = "tune")]
pub fn set(name: String, value: i16) {
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
