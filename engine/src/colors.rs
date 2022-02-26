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

pub type Color = i8;

pub const WHITE: Color = 1;
pub const BLACK: Color = -1;

pub trait ToIndex {
    fn idx(self) -> usize;
}

pub trait Flip {
    fn flip(self) -> Self;
}

impl ToIndex for Color {
    // Returns 0 for black and 1 for white
    #[inline(always)]
    fn idx(self) -> usize {
        ((self >> 1) + 1) as usize
    }
}

impl Flip for Color {
    #[inline(always)]
    fn flip(self) -> Self {
        -self
    }
}

#[cfg(test)]
mod tests {
    use crate::colors::{ToIndex, BLACK, WHITE, Flip};

    #[test]
    fn index_for_black() {
        assert_eq!(0, BLACK.idx())
    }

    #[test]
    fn index_for_white() {
        assert_eq!(1, WHITE.idx())
    }

    #[test]
    fn flip_black() {
        assert_eq!(WHITE.idx(), BLACK.flip().idx())
    }

    #[test]
    fn flip_white() {
        assert_eq!(BLACK.idx(), WHITE.flip().idx())
    }
}