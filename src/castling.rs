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

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum Castling {
    WhiteKingSide = 1 << 0,
    BlackKingSide = 1 << 1,
    WhiteQueenSide = 1 << 2,
    BlackQueenSide = 1 << 3,

    WhiteHasCastled = 1 << 4,
    BlackHasCastled = 1 << 5,
}

