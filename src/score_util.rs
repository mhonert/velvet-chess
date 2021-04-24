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

pub const MIN_SCORE: i32 = -8191;
pub const MAX_SCORE: i32 = 8191;

pub const MATED_SCORE: i32 = -8000;
pub const MATE_SCORE: i32 = 8000;

pub const fn pack_scores(score: i16, eg_score: i16) -> u32 {
    (score as u32) << 16 | ((eg_score as u32) & 0xFFFF)
}

#[inline]
pub fn unpack_score(packed_score: u32) -> i16 {
    (packed_score >> 16) as i16
}

#[inline]
pub fn unpack_eg_score(packed_score: u32) -> i16 {
    (packed_score & 0xFFFF) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_any_scores() {
        let a = 4711;
        let b = 1024;
        let packed = pack_scores(a, b);

        assert_eq!(unpack_score(packed), a);
        assert_eq!(unpack_eg_score(packed), b);
    }

    #[test]
    fn pack_min_val_scores() {
        let a = i16::MIN;
        let b = i16::MIN;
        let packed = pack_scores(a, b);

        assert_eq!(unpack_score(packed), a);
        assert_eq!(unpack_eg_score(packed), b);
    }

    #[test]
    fn pack_max_val_scores() {
        let a = i16::MAX;
        let b = i16::MAX;
        let packed = pack_scores(a, b);

        assert_eq!(unpack_score(packed), a);
        assert_eq!(unpack_eg_score(packed), b);
    }

    #[test]
    fn pack_zero_val_scores() {
        let a = 0;
        let b = 0;
        let packed = pack_scores(a, b);

        assert_eq!(unpack_score(packed), a);
        assert_eq!(unpack_eg_score(packed), b);
    }

}
