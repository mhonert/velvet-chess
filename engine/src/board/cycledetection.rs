/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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
use std::mem::swap;
use crate::bitboard::{get_king_attacks, get_knight_attacks, BitBoard};
use crate::colors::{BLACK, WHITE};
use crate::magics::{get_bishop_attacks, get_queen_attacks, get_ray, get_rook_attacks};
use crate::zobrist::{piece_zobrist_key, player_zobrist_key};

static mut CUCKOO_KEYS: [u64; 8192] = [0; 8192];
static mut CUCKOO_MOVES: [u16; 8192] = [0; 8192];

pub fn has_cycle_move(key: u64, occupancy: BitBoard) -> bool {
    let mut i = cuckoo_hash1(key);
    if key != get_cuckoo_key(i) {
        i = cuckoo_hash2(key);
        if key != get_cuckoo_key(i) {
            return false;
        }
    }

    let m = get_cuckoo_move(i);
    if (occupancy & get_ray(m)).is_occupied() {
        return false
    }

    true
}

fn get_cuckoo_key(i: usize) -> u64 {
    unsafe {
        let ptr = &raw const CUCKOO_KEYS;
        *(*ptr).get_unchecked(i)
    }
}

fn get_cuckoo_move(i: usize) -> u16 {
    unsafe {
        let ptr = &raw const CUCKOO_MOVES;
        *(*ptr).get_unchecked(i)
    }
}

fn cuckoo_hash1(hash: u64) -> usize {
    (hash & 0x1FFF) as usize
}

fn cuckoo_hash2(hash: u64) -> usize {
    ((hash >> 16) & 0x1FFF) as usize
}

pub fn init() {
    let mut count = 0;
    for player in [WHITE, BLACK] {
        for piece in 2..=6 {
            for start in 0..64 {
                for end in (start + 1)..64 {
                    let targets = attacks(piece, start);
                    if !targets.is_set(end) {
                        continue;
                    }
                    let mut m = (start << 6 | end) as u16;
                    let mut hash = piece_zobrist_key(player.piece(piece), start) ^ piece_zobrist_key(player.piece(piece), end) ^ player_zobrist_key();

                    let mut i = cuckoo_hash1(hash);
                    while m != 0 {
                        swap(unsafe { &mut CUCKOO_KEYS[i] }, &mut hash);
                        swap(unsafe { &mut CUCKOO_MOVES[i] }, &mut m);
                        i = if i == cuckoo_hash1(hash) { cuckoo_hash2(hash) } else { cuckoo_hash1(hash) };
                    }
                    count += 1;
                }
            }
        }
    }
    assert_eq!(count, 3668);
}

fn attacks(piece: i8, pos: usize) -> BitBoard {
    match piece {
        2 => get_knight_attacks(pos),
        3 => get_bishop_attacks(!0, pos),
        4 => get_rook_attacks(!0, pos),
        5 => get_queen_attacks(!0, pos),
        6 => get_king_attacks(pos),
        _ => unreachable!()
    }
}
