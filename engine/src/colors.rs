/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

#[derive(Copy, Clone, Debug)]
pub struct Color(u8);

pub const WHITE: Color = Color(0);
pub const BLACK: Color = Color(1);

impl Color {
    /// Returns the color of the given piece
    pub fn from_piece(piece: i8) -> Self {
        if piece > 0 {
            WHITE
        } else {
            BLACK
        }
    }

    /// Determines the active color from the halfmove count
    pub fn from_halfmove_count(halfmove_count: u16) -> Self {
        Color((halfmove_count & 1) as u8)
    }

    /// Returns 0 for white and 1 for black
    pub fn idx(self) -> usize {
        self.0 as usize & 1
    }

    /// Flips the current color (WHITE -> BLACK and vice versa)
    pub fn flip(self) -> Self {
        Color(self.0 ^ 1)
    }

    pub fn is_white(self) -> bool {
        self.0 == 0
    }

    pub fn is_black(self) -> bool {
        self.0 != 0
    }

    /// Returns a piece for the current color using the given piece ID (1-6)
    pub fn piece(self, piece_id: i8) -> i8 {
        debug_assert!((1..=6).contains(&piece_id));
        if self.0 == 0 {
            piece_id
        } else {
            -piece_id
        }
    }

    pub fn is_own_piece(self, piece: i8) -> bool {
        debug_assert_ne!(piece, 0);
        piece.is_positive() == (self.0 == 0)
    }

    pub fn is_opp_piece(self, piece: i8) -> bool {
        debug_assert_ne!(piece, 0);
        piece.is_negative() == (self.0 == 0)
    }

    /// Converts the given score (white perspective) to a score from the own perspective
    pub fn score(self, score_white_pov: i32) -> i32 {
        if self.0 == 0 {
            score_white_pov
        } else {
            -score_white_pov
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::colors::{Color, BLACK, WHITE};

    #[test]
    fn test_from_piece() {
        for i in 1..=6 {
            assert!(Color::from_piece(i).is_white());
            assert!(Color::from_piece(-i).is_black());
        }
    }

    #[test]
    fn test_index() {
        assert_eq!(0, WHITE.idx());
        assert_eq!(1, BLACK.idx());
    }

    #[test]
    fn test_flip() {
        assert_eq!(WHITE.idx(), BLACK.flip().idx());
        assert_eq!(BLACK.idx(), WHITE.flip().idx());
    }

    #[test]
    fn test_is_white() {
        assert!(WHITE.is_white());
        assert!(!BLACK.is_white());
    }

    #[test]
    fn test_is_black() {
        assert!(BLACK.is_black());
        assert!(!WHITE.is_black());
    }

    #[test]
    fn test_piece() {
        assert_eq!(2, WHITE.piece(2));
        assert_eq!(-2, BLACK.piece(2));
    }

    #[test]
    fn test_is_own_piece() {
        assert!(WHITE.is_own_piece(2));
        assert!(BLACK.is_own_piece(-2));

        assert!(!WHITE.is_own_piece(-2));
        assert!(!BLACK.is_own_piece(2));
    }

    #[test]
    fn test_is_opp_piece() {
        assert!(WHITE.is_opp_piece(-4));
        assert!(BLACK.is_opp_piece(4));

        assert!(!WHITE.is_opp_piece(4));
        assert!(!BLACK.is_opp_piece(-4));
    }

    #[test]
    fn test_score() {
        assert_eq!(123, WHITE.score(123));
        assert_eq!(-123, WHITE.score(-123));

        assert_eq!(-123, BLACK.score(123));
        assert_eq!(123, BLACK.score(-123));
    }
}
