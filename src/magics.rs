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

use crate::bitboard::{get_rook_attacks, get_bishop_attacks};
use crate::random::Random;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn find_magics() {
    let rook_magics = find_magics_fast(get_rook_attacks, 12, 0xEB168680668B590E, 0xFF801FFFFFFFFFFF);
    println!("\nRook magic numbers:\n{:?}", rook_magics);

    let bishop_magics = find_magics_fast(get_bishop_attacks, 9, 0x7A5AB079FD61A9F2, u64::max_value());
    println!("\nBishop magic numbers:\n{:?}\n", bishop_magics);
}

fn find_magics_fast(gen_attacks: fn(u64, i32) -> u64, shift: u32, rnd_seed: u64, magic_mask07: u64) -> Vec<u64> {
    let mut r = Random::new_with_seed(rnd_seed);

    let mut magics: Vec<u64> = Vec::with_capacity(64);

    let mut move_target_table: Vec<u64> = vec!(u64::max_value(); 1 << shift);

    for pos in 0..64 {
        let move_mask = gen_attacks(0, pos);
        let block_mask = mask_without_outline(move_mask, pos as u32);

        let blocker_count = block_mask.count_ones();

        let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
        create_blocker_permutations(&mut permutations, 0, block_mask);

        // Special mask for squares 0 and 7 to speed up the search for valid magic numbers for rooks
        let magic_mask = if pos == 0 || pos == 7 { magic_mask07 } else { u64::max_value() };

        loop {
            let mut magic_num = 0;
            let mut rnd = 0;
            for _ in 0..(blocker_count + (blocker_count / 5)) {
                if rnd == 0 {
                    rnd = r.rand32();
                }
                magic_num |= 1 << (rnd & 63);
                rnd >>= 6;
            }
            magic_num &= magic_mask;

            let (is_valid, _) = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
            if is_valid {
                magics.push(magic_num);
                break;
            }
        }
    }

    magics
}


pub fn find_sparse_rook_magics(pos: i32) {
    find_sparse_magics(pos, get_rook_attacks, 12, 0xFF801FFFFFFFFFFF, false);
}

pub fn find_sparse_bishop_magics(pos: i32) {
    find_sparse_magics(pos, get_bishop_attacks, 9, u64::max_value(), true);
}

fn find_sparse_magics(pos: i32, gen_attacks: fn(u64, i32) -> u64, shift: u32, magic_mask07: u64, sparse_random: bool) {
    println!("Searching magic numbers for position {} ...", pos);
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    let mut r = Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64);

    // Special mask for squares 0 and 7 to speed up the search for valid magic numbers (only relevant for rooks)
    let magic_mask = if pos == 0 || pos == 7 { magic_mask07 } else { u64::max_value() };

    let move_mask = gen_attacks(0, pos);
    let block_mask = mask_without_outline(move_mask, pos as u32);

    let blocker_count = block_mask.count_ones();

    let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
    create_blocker_permutations(&mut permutations, 0, block_mask);

    let mut move_target_table: Vec<u64> = vec!(u64::max_value(); 1 << shift);

    let mut best_fill_count = u32::max_value();

    let mut unique_move_targets: Vec<u64> = permutations.iter().map(|&p| gen_attacks(p, pos)).collect();
    unique_move_targets.sort_unstable();
    unique_move_targets.dedup();
    let optimal_fill_count = unique_move_targets.len() as u32;

    loop {
        let mut magic_num = r.rand64();
        if sparse_random {
            magic_num &= r.rand64();
        }
        magic_num &= magic_mask;

        let (is_valid, fill_count) = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
        if !is_valid {
            continue;
        }

        // Found a magic number candidate
        if fill_count < best_fill_count {
            best_fill_count = fill_count;
            println!("{:2} - Best: ({:016x}): {:4} / {:4} / {:4}", pos, magic_num, 1 << blocker_count, best_fill_count, optimal_fill_count);
        }

        // -> Toggle some bits to check, if the fill count can be reduced
        let mut local_best_fill_count = fill_count;
        loop {
            let mut best_bit = -1;
            for i in 0..64 {
                let toggle_mask = 1 << i;
                magic_num ^= toggle_mask;

                let (is_valid, fill_count) = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
                if is_valid && (fill_count < local_best_fill_count || (fill_count == local_best_fill_count && magic_num & toggle_mask == 0)) {
                    best_bit = i;
                    local_best_fill_count = fill_count;
                }

                magic_num ^= toggle_mask;
            }

            if best_bit >= 0 {
                magic_num ^= 1 << best_bit;

                if local_best_fill_count < best_fill_count {
                    best_fill_count = local_best_fill_count;
                    println!("{:2} - Best: ({:016x}): {:4} / {:4} / {:4}", pos, magic_num, 1 << blocker_count, best_fill_count, optimal_fill_count);
                }
            } else {
                break;
            }
        }
    }
}

fn validate_magic_num(permutations: &Vec<u64>, gen_attacks: fn(u64, i32) -> u64, move_target_table: &mut Vec<u64>, pos: i32, magic_num: u64, shift: u32) -> (bool, u32) {
    move_target_table.fill(u64::max_value());
    let mut fill_count = 0;
    for &p in permutations.iter() {
        let move_targets = gen_attacks(p, pos);

        let index = (p.wrapping_mul(magic_num)) >> (64 - shift);
        if move_target_table[index as usize] != u64::max_value() {
            // Already occupied?
            if move_target_table[index as usize] != move_targets {
                // Invalid magic number
                return (false, 0);
            }
        } else {
            move_target_table[index as usize] = move_targets;
            fill_count += 1;
        }
    }

    (true, fill_count)
}

pub fn create_blocker_permutations(permutations: &mut Vec<u64>, prev_blockers: u64, blockers: u64) {
    permutations.push(blockers | prev_blockers);

    let mut rem_blockers = blockers;
    while rem_blockers != 0 {
        let pos = rem_blockers.trailing_zeros();
        rem_blockers ^= 1 << pos as u64;
        create_blocker_permutations(permutations, (prev_blockers | blockers) & ((1 << pos as u64) - 1), rem_blockers);
    }
}

fn mask_without_outline(mut mask: u64, pos: u32) -> u64 {
    if pos & 7 > 0 {
        mask &= !0b00000001_00000001_00000001_00000001_00000001_00000001_00000001_00000001;
    }

    if pos & 7 < 7 {
        mask &= !0b10000000_10000000_10000000_10000000_10000000_10000000_10000000_10000000;
    }

    if pos / 8 > 0 {
        mask &= !0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_11111111;
    }

    if pos / 8 < 7 {
        mask &= !0b11111111_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
    }

    mask
}