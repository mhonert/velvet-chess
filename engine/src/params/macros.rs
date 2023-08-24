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

#[cfg(feature = "tune")]
macro_rules! tunable_params {
    ($($name:ident = $value:literal)+) => {
        #[allow(non_upper_case_globals)]
        mod sdecl {
        $(
            pub static mut $name: i16 = $value;
        )+
        }

        $(
        pub fn $name() -> i16 { unsafe { sdecl::$name } }
        )+

        fn set_param(name: String, value: i16) -> bool {
            match name.as_str() {
                $(
                stringify!($name) => {
                    unsafe { sdecl::$name = value };
                    return true;
                },
                )+
                _ => {}
            }
            false
        }

        fn print_single_options() {
            $(
                println!("option name {} type spin default {} min {} max {}", stringify!($name), $name(), i16::MIN, i16::MAX);
            )+
        }
    }
}

#[cfg(not(feature = "tune"))]
macro_rules! tunable_params {
    ($($name:ident = $value:literal)+) => {
        #[allow(non_upper_case_globals)]
        mod sdecl {
        $(
            pub const $name: i16 = $value;
        )+
        }

        $(
        pub const fn $name() -> i16 { sdecl::$name }
        )+
    }
}

macro_rules! count_elems {
    () => { 0usize };
    ($_head:literal $($tail:literal)*) => { 1usize + count_elems!($($tail)*)};
}

#[cfg(feature = "tune")]
macro_rules! tunable_array_params {
    ($($name:ident = [$($value:literal),+])+) => {
        #[allow(non_upper_case_globals)]
        mod adecl {
        $(
            pub static mut $name: [i16; count_elems!($($value)+)] = [$($value),+];
        )+
        }

        $(
        pub fn $name(i: usize) -> i16 { unsafe { *adecl::$name.get_unchecked(i) } }
        )+

        fn set_array_param(name: String, value: i16) -> bool {
            if let Some((base_name, idx_str)) = name.rsplit_once('_') {
                if let Ok(idx) = usize::from_str(idx_str) {
                    match base_name {
                        $(
                        stringify!($name) => {
                            unsafe { adecl::$name[idx] = value };
                            return true;
                        },
                        )+
                        _ => {}
                    }
                }
            }
            false
        }

        fn print_array_options() {
            $(
            for i in 0..count_elems!($($value)+) {
                println!("option name {}_{} type spin default {} min {} max {}", stringify!($name), i, $name(i), i16::MIN, i16::MAX);
            }
            )+
        }
    }
}

#[cfg(not(feature = "tune"))]
macro_rules! tunable_array_params {
    ($($name:ident = [$($value:literal),+])+) => {
        #[allow(non_upper_case_globals)]
        mod adecl {
        $(
            pub static $name: [i16; count_elems!($($value)+)] = [$($value),+];
        )+
        }

        $(
        #[inline(always)]
        pub fn $name(i: usize) -> i16 { unsafe { *adecl::$name.get_unchecked(i) } }
        )+
    }
}
