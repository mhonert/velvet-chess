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

const ROOK_MAGIC_CANDIDATES: [u64; 64] = [0x00800122543fffff, 0x0040004010000820, 0x3fdfffdff3fbffe8, 0x7fe00c0e080c0020, 0x001819c017f3c0cd, 0x5580008003060004, 0x000810c021800844, 0x02000171d9bbfffa, 0x0000082104104200, 0x0000100400080010, 0x0000420008404004, 0x0000202004000200, 0x0000202001000200, 0x0000202000800100, 0x0000400080004001, 0x0000200020820041, 0x0040002000100020, 0x0004001000080010, 0x0004000802010008, 0x0002002004002002, 0x0002002001002002, 0x0001002020008001, 0x0000004040008001, 0x000000200040c020, 0x0000200040001040, 0x0000080010040010, 0x0000080040044010, 0x0000040020020020, 0x0000020020200100, 0x0000008020010020, 0x0000400040008001, 0x0000820020004020, 0x0040001000200020, 0x0000080400100010, 0x0008404080420008, 0x0000200200200400, 0x0000200200200100, 0x0000200080200100, 0x0000008000404001, 0x0000802000200040, 0x0040201004000800, 0x0010080102000400, 0x0000044008004010, 0x0004002002002004, 0x0001002002002001, 0x0001002000802001, 0x0000400080004001, 0x0000944804004801, 0x0040001000200020, 0x0000100400080010, 0x0004010008020008, 0x0000200200040020, 0x0000020020010020, 0x0000010020008020, 0x0000200040008020, 0x0000802000400020, 0x00fff77fdbbeffee, 0x007fffef80fff7bf, 0x0000400804108022, 0x0000100820044002, 0x0000200110040802, 0x0003fc8002040801, 0x0003ffffbf7803fe, 0x0001fffee83f7d9e];

const BISHOP_MAGIC_CANDIDATES: [u64; 64] = [ 0x0002008020084040, 0x0000208020082000, 0x0000808010020000, 0x0000806004000000, 0x0000440200000000, 0x000021c100800000, 0x0000208044004000, 0x0000020080410080, 0x0000010041000820, 0x0000004100202002, 0x0000010010200200, 0x0000008060040000, 0x0000004402000000, 0x00000021c1008000, 0x0000002082008040, 0x0000002010420020, 0x0004000200801010, 0x0002000020404008, 0x0000400400102040, 0x0000200200801000, 0x0000240080840000, 0x0000080080840080, 0x0000040010808040, 0x0000040010108010, 0x0001008000408100, 0x0000420000202040, 0x0000804000820080, 0xa202002208008020, 0x05810070c3014000, 0x0000204000804020, 0x0000200800208010, 0x0000081000084008, 0x0000820080010020, 0x0000208040020010, 0x0001008080004100, 0x24014008a19a0200, 0x11010304002a0020, 0x0002040008004020, 0x0000108008001040, 0x0000200410002008, 0x0000404200400080, 0x0000208040200040, 0x0000808080800040, 0x0000008840200040, 0x0000020041000010, 0x0000810200800008, 0x0002008010100004, 0x0000208020080002, 0x5100402084004202, 0x0000084042002000, 0x0000001010806000, 0x0000000008403000, 0x0000000100202000, 0x0000008101000800, 0x0000802040404000, 0x0000210040080800, 0x0000201008010080, 0x0000001080804020, 0x0000000008208060, 0x0000000000084030, 0x0000000001002020, 0x0000000100210010, 0x0000040080204004, 0x0000810041000420];

// static mut ROOK_ATTACKS: [u64; 64 * 4096] = [u64::max_value(); 64 * 4096];
// static mut BISHOP_ATTACKS: [u64; 64 * 512] = [u64::max_value(); 64 * 512];

static mut ATTACKS: [u64; 121328 + 4096] = [u64::max_value(); 121328 + 4096];

pub fn optimize_attack_tables() {
    let mut attacks: Vec<u64> = vec!(u64::max_value(); 4096 * 64);

    println!("Rooks:");
    let table_length = unsafe { optimize_rook_attacks(&mut ROOK_ATTACKS, &mut attacks, 4096) };
    println!("\nBishops:");
    unsafe { optimize_bishop_attacks(&mut BISHOP_ATTACKS, &mut attacks, 512, table_length) };
}

pub fn optimize_rook_attacks(src_attacks: &mut[u64], attacks: &mut Vec<u64>, size: usize) -> i32 {

    let mut offsets: Vec<usize> = vec!(0; 64);
    let mut finished: HashSet<i32> = HashSet::new();
    let mut best_pos = 0;
    let mut best_offset = 0;
    let mut current_index = 0;

    let mut backup = vec!(0; size);

    for iteration in 0..64 {
        let mut lowest_index = i32::max_value();
        for src_pos in 0..64 {

            if finished.contains(&src_pos) {
                continue;
            }

            let src_offset = src_pos as usize * size;

            let mut max_index: i32 = 0;
            for offset_adj in (-(src_offset as i32))..(current_index - (src_offset as i32) + size as i32) {
                let mut last_pos = 0;
                let mut is_valid = true;
                let start_index = (src_offset as i32 + offset_adj);
                let end_index = (src_offset as i32 + offset_adj + (size as i32 - 1));
                if start_index < 0 || end_index >= (attacks.len() as i32) {
                    continue;
                }

                for i in 0..size {
                    let attack = src_attacks[src_offset + i];
                    let index = (src_offset as i32 + offset_adj + i as i32) as usize;
                    if index >= attacks.len() {
                        is_valid = false;
                        break;
                    }

                    backup[i] = attacks[index];

                    if attack == u64::max_value() {
                        continue;
                    }

                    if attacks[index] != u64::max_value() && attacks[index] != attack {
                        is_valid = false;
                        break;
                    }

                    attacks[index] = attack;
                    max_index = max(index as i32, max_index);
                    last_pos = i;
                }

                // Revert
                for pos in 0..=last_pos {
                    let index = (src_offset as i32 + offset_adj + pos as i32) as usize;
                    attacks[index] = backup[pos];
                }

                if is_valid {
                    // Could transfer all attack values without collision
                    if max_index < lowest_index {
                        lowest_index = max_index;
                        best_pos = src_pos;
                        best_offset = offset_adj;
                    }
                }
            }
        }

        // Apply best
        println!("{}: Best {} / {}: {}", iteration, best_pos, best_offset, lowest_index);
        let src_offset = best_pos as usize * size;
        for pos in 0..size {
            let index = (src_offset as i32 + best_offset + pos as i32) as usize;
            let attack = unsafe { src_attacks[src_offset + pos] };
            if attack != u64::max_value() {
                attacks[index] = attack;
            }
        }
        offsets[best_pos as usize] = (best_offset + best_pos * size as i32) as usize;
        finished.insert(best_pos);
        current_index = lowest_index + 1;
    }

    println!("Finished");
    println!("{:?}", offsets);

    current_index
}

pub fn optimize_bishop_attacks(src_attacks: &mut[u64], attacks: &mut Vec<u64>, size: usize, mut curr_table_length: i32) {

    let mut offsets: Vec<usize> = vec!(0; 64);
    let mut finished: HashSet<i32> = HashSet::new();
    let mut best_pos = 0;
    let mut best_offset = 0;
    let mut best_index = 0;
    let mut current_index = 0;

    let mut backup = vec!(0; size);

    for iteration in 0..64 {
        let mut lowest_gap_count = i32::max_value();
        for src_pos in 0..64 {

            if finished.contains(&src_pos) {
                continue;
            }

            let src_offset = src_pos as usize * size;

            let mut max_index: i32 = 0;
            for offset_adj in (-(src_offset as i32))..(curr_table_length - (src_offset as i32) + size as i32) {
                let mut last_pos = 0;
                let mut is_valid = true;
                let start_index = (src_offset as i32 + offset_adj);
                let end_index = (src_offset as i32 + offset_adj + (size as i32 - 1));
                if start_index < 0 || end_index >= (attacks.len() as i32) {
                    continue;
                }

                for i in 0..size {
                    let attack = src_attacks[src_offset + i];
                    let index = (src_offset as i32 + offset_adj + i as i32) as usize;
                    if index >= attacks.len() {
                        is_valid = false;
                        break;
                    }

                    backup[i] = attacks[index];

                    if attack == u64::max_value() {
                        continue;
                    }

                    if attacks[index] != u64::max_value() && attacks[index] != attack {
                        is_valid = false;
                        break;
                    }

                    attacks[index] = attack;
                    max_index = max(index as i32, max_index);
                    last_pos = i;
                }

                if is_valid { // Could transfer all attack values without collision
                    let mut gap_count = 0;
                    for pos in 0..=last_pos {
                        let index = (src_offset as i32 + offset_adj + pos as i32) as usize;
                        if attacks[index] == u64::max_value() {
                            gap_count += 1;
                        }
                    }

                    if gap_count < lowest_gap_count {
                        lowest_gap_count = gap_count;
                        best_index = max_index;
                        curr_table_length = max(best_index, curr_table_length);
                        best_pos = src_pos;
                        best_offset = offset_adj;
                    }
                }

                // Revert
                for pos in 0..=last_pos {
                    let index = (src_offset as i32 + offset_adj + pos as i32) as usize;
                    attacks[index] = backup[pos];
                }
            }
        }

        // Apply best
        println!("{}: Best {} / {} / {}: {}", iteration, best_pos, best_offset, lowest_gap_count, best_index);
        let src_offset = best_pos as usize * size;
        for pos in 0..size {
            let index = (src_offset as i32 + best_offset + pos as i32) as usize;
            let attack = unsafe { src_attacks[src_offset + pos] };
            if attack != u64::max_value() {
                attacks[index] = attack;
            }
        }
        offsets[best_pos as usize] = (best_offset + best_pos * size as i32) as usize;
        finished.insert(best_pos);
        current_index = best_index + 1;
        curr_table_length = max(current_index, curr_table_length);
    }

    println!("Finished");
    println!("{:?}", offsets);

}

pub fn find_magics() {
    let rook_magics = find_magics_fast(gen_rook_attacks, 12, 0xEB168680668B590E, 0xFF801FFFFFFFFFFF);
    println!("\nRook magic numbers:\n{:?}", rook_magics);

    let bishop_magics = find_magics_fast(gen_bishop_attacks, 9, 0x7A5AB079FD61A9F2, u64::max_value());
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

            let is_valid = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
            if is_valid {
                magics.push(magic_num);
                break;
            }
        }
    }

    magics
}


pub fn find_sparse_rook_magics(pos: i32) {
    find_sparse_magics(pos, gen_rook_attacks, 12, 0xFF801FFFFFFFFFFF, false);
}

pub fn find_sparse_bishop_magics(pos: i32) {
    find_sparse_magics(pos, gen_bishop_attacks, 9, u64::max_value(), true);
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

    let mut best_gap_count = -1;

    loop {
        let mut magic_num = r.rand64();
        if sparse_random {
            magic_num &= r.rand64();
        }
        magic_num &= magic_mask;

        let is_valid = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
        if !is_valid {
            continue;
        }

        // Found a magic number candidate
        let gap_count = magic_num_score(&move_target_table, shift);
        if gap_count > best_gap_count {
            best_gap_count = gap_count;
            println!("{:2} - Best: ({:016x}): {:4}", pos, magic_num, best_gap_count);
        }

        // -> Toggle some bits to check, if the fill count can be reduced
        let mut local_best_gap_count = gap_count;
        loop {
            let mut best_bit = -1;
            for i in 0..64 {
                let toggle_mask = 1 << i;
                magic_num ^= toggle_mask;

                let is_valid= validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
                if is_valid {
                    let gap_count = magic_num_score(&move_target_table, shift);
                    if gap_count > local_best_gap_count || (gap_count == local_best_gap_count && magic_num & toggle_mask == 0) {
                        best_bit = i;
                        local_best_gap_count = gap_count;
                    }
                }

                magic_num ^= toggle_mask;
            }

            if best_bit >= 0 {
                magic_num ^= 1 << best_bit;

            } else {
                break;
            }
        }
        if local_best_gap_count > best_gap_count {
            best_gap_count = local_best_gap_count;
            println!("{:2} - Best: ({:016x}): {:4}", pos, magic_num, best_gap_count);
        }
    }
}

fn validate_magic_num(permutations: &[u64], gen_attacks: fn(u64, i32) -> u64, move_target_table: &mut Vec<u64>, pos: i32, magic_num: u64, shift: u32) -> bool {
    move_target_table.fill(u64::max_value());
    for &p in permutations.iter() {
        let move_targets = gen_attacks(p, pos);

        let index = (p.wrapping_mul(magic_num)) >> (64 - shift);
        if move_target_table[index as usize] != u64::max_value() {
            // Already occupied?
            if move_target_table[index as usize] != move_targets {
                // Invalid magic number
                return false;
            }
        } else {
            move_target_table[index as usize] = move_targets;
        }
    }

    true
}

fn magic_num_score(table: &Vec<u64>, shift: u32) -> i32 {
    let size = 1 << shift;
    let mut leading_gap: i32 = 0;
    for i in 0..size {
        if table[i] != u64::max_value() {
            break;
        }
        leading_gap += 1;
    }

    let mut trailing_gap: i32 = 0;
    for i in (0..size).rev() {
        if table[i] != u64::max_value() {
            break;
        }
        trailing_gap += 1;
    }

    let mut biggest_gap: i32 = 0;
    let mut current_gap: i32 = 0;
    for i in leading_gap..(size as i32 - trailing_gap) {
        if table[i as usize] == u64::max_value() {
            current_gap += 1;
        } else {
            biggest_gap = max(current_gap, biggest_gap);
            current_gap = 0;
        }
    }

    (leading_gap + trailing_gap) + biggest_gap / 2
}

// pub fn initialize_attacks() {
//     let mut offset = 0;
//
//     for pos in 0..64 {
//         let move_mask = gen_rook_attacks(0, pos);
//         let block_mask = mask_without_outline(move_mask, pos as u32);
//         unsafe { ROOK_BLOCK_MASKS[pos as usize] = block_mask };
//
//         let blocker_count = block_mask.count_ones();
//
//         let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
//         create_blocker_permutations(&mut permutations, 0, block_mask);
//
//
//         let magic_num = ROOK_MAGICS[pos as usize];
//
//         for &p in permutations.iter() {
//             let move_targets = gen_rook_attacks(p, pos);
//
//             let index = (p.wrapping_mul(magic_num)) >> (64 - 12);
//             unsafe { ROOK_ATTACKS[index as usize + offset] = move_targets };
//         }
//
//         offset += 4096;
//     }
//
//     offset = 0;
//
//     for pos in 0..64 {
//         let move_mask = gen_bishop_attacks(0, pos);
//         let block_mask = mask_without_outline(move_mask, pos as u32);
//         unsafe { BISHOP_BLOCK_MASKS[pos as usize] = block_mask };
//
//         let blocker_count = block_mask.count_ones();
//
//         let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
//         create_blocker_permutations(&mut permutations, 0, block_mask);
//
//
//         let magic_num = BISHOP_MAGICS[pos as usize];
//
//         for &p in permutations.iter() {
//             let move_targets = gen_bishop_attacks(p, pos);
//
//             let index = (p.wrapping_mul(magic_num)) >> (64 - 9);
//             unsafe { BISHOP_ATTACKS[index as usize + offset] = move_targets };
//         }
//
//         offset += 512;
//     }
// }

