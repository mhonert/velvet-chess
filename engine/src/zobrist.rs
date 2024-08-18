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

use crate::random::rand64;

const SEED: u64 = 0x4d595df4d0f33173;

const PLAYER_ZOBRIST_KEY: u64 = 1;
const EP: u128 = 0xeedf589a68d8b723acdd859e243fc895;
static CASTLING_ZOBRIST_KEYS: [u64; 16] = gen_keys::<16>(0);
static PIECE_ZOBRIST_KEYS: [u64; 13 * 64] = gen_keys::<832>(16);

#[inline]
pub fn player_zobrist_key() -> u64 {
    PLAYER_ZOBRIST_KEY
}

#[inline]
pub fn enpassant_zobrist_key(en_passant_state: u8) -> u64 {
    (EP >> en_passant_state) as u64
}

#[inline]
pub fn castling_zobrist_key(castling_state: u8) -> u64 {
    unsafe { *CASTLING_ZOBRIST_KEYS.get_unchecked(castling_state as usize) }
}

#[inline]
pub fn piece_zobrist_key(piece: i8, pos: usize) -> u64 {
    unsafe { *PIECE_ZOBRIST_KEYS.get_unchecked(((piece + 6) as usize) * 64 + pos) }
}

const fn gen_keys<const N: usize>(mut skip: usize) -> [u64; N] {
    let mut state = SEED;
    while skip > 0 {
        let (new_state, _) = rand64(state);
        state = new_state;
        skip -= 1;
    }
    let mut keys = [0u64; N];

    let mut i = 0;
    while i < N {
        let (new_state, key) = rand64(state);
        keys[i] = key;
        state = new_state;
        i += 1;
    }

    keys
}

#[cfg(test)]
mod tests {
    use crate::board::{BlackBoardPos, WhiteBoardPos};
    use super::*;

    #[test]
    fn check_key_quality() {
        let mut all_keys = vec![PLAYER_ZOBRIST_KEY];
        for ep in (WhiteBoardPos::EnPassantLineStart as u8)..=(WhiteBoardPos::EnPassantLineEnd as u8) {
            all_keys.push(enpassant_zobrist_key(ep));
        }
        for ep in (BlackBoardPos::EnPassantLineStart as u8)..=(BlackBoardPos::EnPassantLineEnd as u8) {
            all_keys.push(enpassant_zobrist_key(ep));
        }

        CASTLING_ZOBRIST_KEYS.iter().for_each(|&k| all_keys.push(k));
        PIECE_ZOBRIST_KEYS.iter().for_each(|&k| all_keys.push(k));
        let mut duplicates = all_keys.len();
        all_keys.sort_unstable();
        all_keys.dedup();
        duplicates -= all_keys.len();

        assert_eq!(0, duplicates);
        assert_eq!(0, all_keys.iter().filter(|&k| *k == 0 || *k == u64::MAX).count());
    }
}
