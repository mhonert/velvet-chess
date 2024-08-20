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

#[cfg(feature = "tune")]
macro_rules! tunable_params {
    ($($name:ident = $value:literal)+) => {
        #[derive(Copy, Clone)]
        pub struct SingleParams {
        $(
            $name: i16,
        )+
        }

        impl Default for SingleParams {
            fn default() -> Self {
                SingleParams {
                $(
                    $name: $value,
                )+
                }
            }
        }

        impl SingleParams {
            $(
            #[inline(always)]
            pub fn $name(&self) -> i16 { self.$name }
            )+

            pub fn set_param(&mut self, name: &str, value: i16) -> bool {
                match name {
                    $(
                    stringify!($name) => {
                        self.$name = value;
                        return true;
                    },
                    )+
                    _ => {}
                }
                false
            }
        }

        fn print_single_options() {
            $(
                println!("option name {} type spin default {} min {} max {}", stringify!($name), $value, i16::MIN, i16::MAX);
            )+
        }
    }
}

#[cfg(not(feature = "tune"))]
macro_rules! tunable_params {
    ($($name:ident = $value:literal)+) => {
        #[derive(Copy, Clone, Default)]
        pub struct SingleParams;

        impl SingleParams {
            $(
            #[inline(always)]
            pub fn $name(&self) -> i16 { $value }
            )+

            pub fn set_param(&self, _name: &str, _value: i16) -> bool {
                false
            }
        }
    }
}

macro_rules! count_elems {
    () => { 0usize };
    ($_head:literal $($tail:literal)*) => { 1usize + count_elems!($($tail)*)};
}

#[cfg(feature = "tune")]
macro_rules! tunable_array_params {
    ($($name:ident = [$($value:literal),+])+) => {
        #[derive(Copy, Clone)]
        pub struct ArrayParams {
        $(
            $name: [i16; count_elems!($($value)+)],
        )+
        }
        
        impl Default for ArrayParams {
            fn default() -> Self {
                ArrayParams {
                $(
                    $name: [$($value),+],
                )+
                }
            }
        }
        
        impl ArrayParams {
            $(
            #[inline(always)]
            pub fn $name(&self, i: usize) -> i16 { self.$name[i] }
            )+
        
            pub fn set_array_param(&mut self, name: &str, value: i16) -> bool {
                if let Some((base_name, idx_str)) = name.rsplit_once('_') {
                    if let Ok(idx) = usize::from_str(idx_str) {
                        match base_name {
                            $(
                            stringify!($name) => {
                                self.$name[idx] = value;
                                return true;
                            },
                            )+
                            _ => {}
                        }
                    }
                }
                false
            }
        }

        fn print_array_options() {
            $(
            for i in 0..count_elems!($($value)+) {
                println!("option name {}_{} type spin default {} min {} max {}", stringify!($name), i, 0, i16::MIN, i16::MAX);
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

        #[derive(Copy, Clone, Default)]
        pub struct ArrayParams;
        
        impl ArrayParams {
            $(
            #[inline(always)]
            pub fn $name(&self, i: usize) -> i16 { unsafe { *adecl::$name.get_unchecked(i) } }
            )+

            pub fn set_array_param(&self, _name: &str, _value: i16) -> bool {
                false
            }
        }
    }
}
