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

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Default for TbRootMoves {
    fn default() -> Self {
        TbRootMoves{
            size: 0,
            moves: [TbRootMove::default(); TB_MAX_MOVES as usize],
        }
    }
}

impl Default for TbRootMove {
    fn default() -> Self {
        TbRootMove{
            move_: 0,
            pv: [0; TB_MAX_PLY as usize],
            pvSize: 0,
            tbScore: 0,
            tbRank: 0,
        }
    }
}

