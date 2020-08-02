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
pub enum WhiteBoardPos {
    KingSideRook = 63,
    QueenSideRook = 56,

    PawnLineStart = 48,
    PawnLineEnd = 55,

    EnPassantLineStart = 16,
    EnPassantLineEnd = 23,

    KingStart = 60,
}

#[repr(u8)]
pub enum BlackBoardPos {
    QueenSideRook = 0,
    KingSideRook = 7,

    PawnLineStart = 8,
    PawnLineEnd = 15,

    EnPassantLineStart = 40,
    EnPassantLineEnd = 47,

    KingStart = 4,
}

