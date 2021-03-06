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

use std::collections::HashSet;
use std::cmp::{max, min};
use velvet::bitboard::{gen_rook_attacks, gen_bishop_attacks, create_blocker_permutations, mask_without_outline};
use std::time::{SystemTime, UNIX_EPOCH};
use velvet::random::Random;
use std::{env};
use std::str::FromStr;
use std::process::exit;
use velvet::magics::{initialize_attacks, EMPTY_MAGIC};
use std::fs::File;
use std::io::{BufReader, BufRead, BufWriter, Write};

enum Command {
    FastMagics,
    SearchRookMagics(i32, fn(&[u64], u32) -> i32),
    SearchBishopMagics(i32, fn(&[u64], u32) -> i32),
    PackRooks(String, String),
    PackBishops(String, String, String),
}

fn main() {
    let args: Vec<String> = env::args().map(|s| s.to_lowercase()).skip(1).collect();
    if args.is_empty() {
        print_usage();
        return;
    }

    let cmd: Command =
        match args[0].as_str() {
            "fast" => Command::FastMagics,
            "packrooks" => {
                if args.len() < 3 {
                    print!("Missing file arguments");
                    print_usage();
                    exit(0);
                }
                Command::PackRooks(args[1].to_string(), args[2].to_string())
            },
            "packbishops" => {
                if args.len() < 4 {
                    print!("Missing file arguments");
                    print_usage();
                    exit(0);
                }
                Command::PackBishops(args[1].to_string(), args[2].to_string(), args[3].to_string())
            },
            "search" => {
                if args.len() < 4 {
                    print!("Missing arguments");
                    print_usage();
                    exit(0);
                }
                let pos = match i32::from_str(args[3].as_str()) {
                    Ok(pos) => pos,
                    Err(e) => {
                        println!("Could not parse position {}: {}", args[3], e);
                        print_usage();
                        exit(1);
                    }
                };

                if !(0..=63).contains(&pos) {
                    println!("Position must be in the range of 0..63");
                    print_usage();
                    exit(1);
                }

                let score_fn = if args[1] == "sparse" {
                    sparse_table_score
                } else if args[1] == "dense" {
                    dense_table_score
                } else {
                    println!("Search type must be either 'sparse' or 'dense'");
                    print_usage();
                    exit(1);
                };

                if args[2] == "rook" {
                    Command::SearchRookMagics(pos, score_fn)
                } else if args[2] == "bishop" {
                    Command::SearchBishopMagics(pos, score_fn)
                } else {
                    println!("Piece must be either rook or bishop");
                    print_usage();
                    exit(2);
                }
            },

            _ => {
                print_usage();
                exit(-1);
            }
        };

    match cmd {
        Command::FastMagics => {
            fast_magics();
        },

        Command::SearchRookMagics(pos, score_fn) => {
            find_rook_magics(pos, score_fn);
        },

        Command::SearchBishopMagics(pos, score_fn) => {
            find_bishop_magics(pos, score_fn);
        },

        Command::PackRooks(candidate_file, rook_file) => {
            pack_rook_attack_tables(candidate_file, rook_file);
        },

        Command::PackBishops(candidate_file, rook_file, output_file) => {
            pack_bishop_attack_tables(candidate_file, rook_file, output_file);
        }
    }
}

fn print_usage() {
    println!("Commands:");
    println!("  fast\n   - calculates and prints a set of magic numbers with fixed shifts for for rooks (12) and bishops (9)");
    println!("  search [sparse|dense] [rook|bishop] {{pos}}\n   - searches for magics for a specific board square (0-63)");
    println!("  packrooks <rook-candidates-input-file> <packed-rook-output-file>\n   - finds magic number combinations and offsets for a minimal attack table size");
    println!("  packbishops <bishop-candidates-input-file> <packed-rook-input-file> <packed-bishop-output-file>\n   - reads the packed rook magics and finds matching bishop magic number combinations for a minimal attack table size");
}

fn pack_rook_attack_tables(candidates_file: String, rook_out_file: String) {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    let mut rnd = Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64);

    println!("Reading rook magic candidates ...");
    let rook_magic_candidates = read_magic_candidates(candidates_file);

    let candidate_count = rook_magic_candidates.len();
    if candidate_count == 0 {
        panic!("file did not contain any magic number candidates");
    }

    let mut rook_attacks: Vec<[u64; 64 * 4096]> = Vec::with_capacity(candidate_count);
    for _ in 0..candidate_count {
        rook_attacks.push([u64::max_value(); 64 * 4096]);
    }

    initialize_rook_attacks(&rook_magic_candidates, &mut rook_attacks);

    let mut attacks: Vec<u64> = vec!(u64::max_value(); 4096 * 64);

    optimize_rook_attacks(rook_out_file, &mut rnd, &rook_magic_candidates, &rook_attacks, &mut attacks, 4096);
}

fn read_magic_candidates(file_name: String) -> Vec<[u64; 64]> {
    let mut candidates: Vec<[u64; 64]> = Vec::new();
    let file = File::open(file_name).expect("Unable to open magics candidates file");
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.expect("Could not read line");
        if line.trim().is_empty() || line.trim().starts_with("#") {
            // skip empty lines and comments
            continue;
        }

        let magics: Vec<u64> = parse_magics_str(line);
        if magics.len() != 64 {
            panic!("Expected 64 magic numbers per line in file");
        }

        let mut magics_arr: [u64; 64] = [0; 64];
        magics_arr.copy_from_slice(&magics[..]);

        candidates.push(magics_arr);
    }

    candidates
}

fn parse_magics_str(magics: String) -> Vec<u64> {
    magics.splitn(64, ',')
        .map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect(&format!("Could not parse magic number: {}", s)))
        .collect()
}

fn pack_bishop_attack_tables(bishop_file: String, rook_file: String, output_file: String) {
    let (mut magics_data, mut offset_data, mut len_data) = (String::new(), String::new(), String::new());
    let file = File::open(rook_file).expect("Unable to open packed rook magics file");
    let mut reader = BufReader::new(file);
    reader.read_line(&mut magics_data).expect("Could not read magics data from packed rook magics file");
    reader.read_line(&mut offset_data).expect("Could not read offset data from packed rook magics file");
    reader.read_line(&mut len_data).expect("Could not read len data from packed rook magics file");

    println!("Reading packed rook magics ...");
    let packed_rook_magics: Vec<u64> = parse_magics_str(magics_data);

    if packed_rook_magics.len() != 64 {
        panic!("Expected 64 magic numbers in file");
    }

    let packed_rook_offsets: Vec<u32>  = offset_data.splitn(64, ',').map(|s| u32::from_str(s.trim()).expect("Could not parse offset number")).collect();
    if packed_rook_offsets.len() != 64 {
        panic!("Invalid number of magic numbers in file");
    }

    let rook_table_length = i32::from_str(&len_data.trim()).expect("Could not parse len data");


    println!("Reading bishop magic number ...");
    let bishop_magics = read_magic_candidates(bishop_file);
    if bishop_magics.len() == 0 {
        panic!("There must be at least one bishop magic candidate in the input file");
    }
    let mut bishop_attacks: Vec<[u64; 64 * 512]> = Vec::with_capacity(bishop_magics.len());
    initialize_bishop_attacks(&bishop_magics, &mut bishop_attacks);

    let mut attacks: Vec<u64> = vec!(u64::max_value(); 4096 * 64);
    initialize_attacks(gen_rook_attacks, &packed_rook_magics, &packed_rook_offsets, &mut attacks, &mut [EMPTY_MAGIC; 64], 12);

    optimize_bishop_attacks(output_file, &bishop_magics, &bishop_attacks, &mut attacks, 512, rook_table_length);
}

fn optimize_rook_attacks(rook_file: String, rnd: &mut Random, magic_candidates: &[[u64; 64]], src_attacks: &[[u64; 4096 * 64]], attacks: &mut Vec<u64>, size: usize) {
    let candidate_count = magic_candidates.len();

    let mut current_index = 0;

    let mut best_offsets: Vec<usize>;
    let mut lowest_max_index = i32::max_value();
    let mut magic_nums: Vec<u64> = vec!(0; 64);

    let mut follow_up_scores: Vec<i32> = vec!(8; candidate_count * candidate_count * 64 * 64);

    loop {
        attacks.fill(u64::max_value());

        let mut max_index: i32 = 0;
        let mut found_pos = false;
        let mut best_offset = 0;
        let mut offsets: Vec<usize> = vec!(0; 64);
        let mut candidates: Vec<i32> = vec!(0; 64);
        let mut finished: HashSet<i32> = HashSet::new();
        let mut prev_pos: i32 = -1;
        let mut prev_candidate: i32 = -1;

        while finished.len() < 64 {
            let mut lowest_index = i32::max_value();
            let mut src_pos;
            let mut src_candidate;
            let mut follow_up_threshold = 8;
            let mut follow_up_index = -1;
            let mut best_so_far = 4096;
            loop {
                src_pos = (rnd.rand32() & 63) as i32;
                src_candidate = (rnd.rand32() % candidate_count as u32) as i32;

                if finished.contains(&src_pos) {
                    continue;
                }

                if prev_pos != -1 {
                    follow_up_index = prev_pos * 64 * candidate_count as i32 * prev_candidate + src_pos * candidate_count as i32 + src_candidate;
                    let follow_up_score = follow_up_scores[follow_up_index as usize];
                    if follow_up_score > follow_up_threshold || follow_up_score > best_so_far * 2 {
                        best_so_far = min(best_so_far, follow_up_score);
                        follow_up_threshold += 2;
                        continue;
                    }

                }

                break;
            }

            let src_offset = src_pos as usize * size;

            found_pos = false;
            for offset_adj in (-(src_offset as i32))..(current_index - (src_offset as i32) + size as i32) {
                let mut is_valid = true;
                let start_index = src_offset as i32 + offset_adj;
                let end_index = src_offset as i32 + offset_adj + (size as i32 - 1);
                if start_index < 0 || end_index >= (attacks.len() as i32) {
                    continue;
                }

                for i in 0..size {
                    let attack = src_attacks[src_candidate as usize][src_offset + i];
                    let index = (src_offset as i32 + offset_adj + i as i32) as usize;
                    if index >= attacks.len() {
                        is_valid = false;
                        break;
                    }

                    if attack == u64::max_value() {
                        continue;
                    }

                    if attacks[index] != u64::max_value() && attacks[index] != attack {
                        is_valid = false;
                        break;
                    }

                    attacks[index] = attack;
                    max_index = max(index as i32, max_index);
                }

                if is_valid {
                    // Could transfer all attack values without collision
                    found_pos = true;
                    lowest_index = max_index;
                    best_offset = offset_adj;
                    break;
                }
            }

            if !found_pos {
                break;
            }

            offsets[src_pos as usize] = (best_offset + src_pos * size as i32) as usize;
            candidates[src_pos as usize] = src_candidate;
            finished.insert(src_pos);
            if follow_up_index >= 0 {
                follow_up_scores[follow_up_index as usize] = (follow_up_scores[follow_up_index as usize] + lowest_index - current_index) / 2;
            }
            current_index = lowest_index + 1;
            prev_pos = src_pos;
            prev_candidate = src_candidate;
        }

        if found_pos && max_index < lowest_max_index {
            lowest_max_index = max_index;
            best_offsets = offsets;

            for i in 0..64 {
                magic_nums[i] = magic_candidates[candidates[i] as usize][i];
            }
            println!("Current Best {} - {:?}",
                     lowest_max_index,
                     (0..candidate_count).map(|n| candidates.iter().filter(|&&c| c == n as i32).count()).collect::<Vec<usize>>());

            let file = File::create(&rook_file).expect("Could not create output file");
            let mut writer = BufWriter::new(file);
            let magics_str = magic_nums.iter().map(|&n| format!("0x{:016x}", n)).collect::<Vec<String>>().join(", ");
            writeln!(writer, "{}", magics_str)
                .expect("Could not write magics to output file");
            writeln!(writer, "{}", best_offsets.iter().map(|&n| format!("{}", n)).collect::<Vec<String>>().join(", "))
                .expect("Could not write offsets to output file");
            write!(writer, "{}", lowest_max_index)
                .expect("Could not write len to output file");
        }

    }
}


fn optimize_bishop_attacks(bishop_file: String, bishop_candidates: &[[u64; 64]], src_attacks: &Vec<[u64; 64 * 512]>, attacks: &mut Vec<u64>, size: usize, mut curr_table_length: i32) {

    let mut offsets: Vec<usize> = vec!(0; 64);
    let mut candidates: Vec<i32> = vec!(0; 64);
    let mut finished: HashSet<i32> = HashSet::new();
    let mut best_pos = 0;
    let mut best_candidate = 0;
    let mut best_offset = 0;
    let mut best_index = 0;

    let mut backup = vec!(0; size);

    for iteration in 0..64 {
        let mut lowest_gap_count = i32::max_value();
        for src_pos in 0..64 {
            if finished.contains(&src_pos) {
                continue;
            }

            for src_candidate in 0..(src_attacks.len()) {
                let src_offset = src_pos as usize * size;

                let mut max_index: i32 = 0;
                for offset_adj in (-(src_offset as i32))..(curr_table_length - (src_offset as i32) + size as i32) {
                    let mut last_pos = 0;
                    let mut is_valid = true;
                    let start_index = src_offset as i32 + offset_adj;
                    let end_index = src_offset as i32 + offset_adj + (size as i32 - 1);
                    if start_index < 0 || end_index >= (attacks.len() as i32) {
                        continue;
                    }

                    for i in 0..size {
                        let attack = src_attacks[src_candidate][src_offset + i];
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
                            best_candidate = src_candidate;
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
        }

        // Apply best
        println!("{}: Best {} / {} / {} / {}: {}", iteration, best_pos, best_candidate, best_offset, lowest_gap_count, best_index);
        let src_offset = best_pos as usize * size;
        for pos in 0..size {
            let index = (src_offset as i32 + best_offset + pos as i32) as usize;
            let attack = src_attacks[best_candidate][src_offset + pos];
            if attack != u64::max_value() {
                attacks[index] = attack;
            }
        }
        offsets[best_pos as usize] = (best_offset + best_pos * size as i32) as usize;
        candidates[best_pos as usize] = best_candidate as i32;
        finished.insert(best_pos);
        curr_table_length = max(best_index + 1, curr_table_length);
    }

    println!("Finished");
    println!("{:?}", offsets);
    println!("{}", curr_table_length);

    let mut magic_nums: Vec<u64> = vec!(0; 64);
    for i in 0..64 {
        magic_nums[i] = bishop_candidates[candidates[i] as usize][i];
    }

    let file = File::create(&bishop_file).expect("Could not create output file");
    let mut writer = BufWriter::new(file);
    let magics_str = magic_nums.iter().map(|&n| format!("0x{:016x}", n)).collect::<Vec<String>>().join(", ");
    writeln!(writer, "{}", magics_str)
        .expect("Could not write magics to output file");
    writeln!(writer, "{}", offsets.iter().map(|&n| format!("{}", n)).collect::<Vec<String>>().join(", "))
        .expect("Could not write offsets to output file");
    write!(writer, "{}", curr_table_length)
        .expect("Could not write len to output file");
}

fn fast_magics() {
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

fn find_rook_magics(pos: i32, magic_num_score: fn(&[u64], u32) -> i32) {
    find_magics(pos, gen_rook_attacks, 12, 0xFF801FFFFFFFFFFF, false, magic_num_score);
}

fn find_bishop_magics(pos: i32, magic_num_score: fn(&[u64], u32) -> i32) {
    find_magics(pos, gen_bishop_attacks, 9, u64::max_value(), true, magic_num_score);
}

fn find_magics(pos: i32, gen_attacks: fn(u64, i32) -> u64, shift: u32, magic_mask07: u64, sparse_random: bool,
               magic_num_score: fn(&[u64], u32) -> i32) {
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

fn dense_table_score(table: &[u64], shift: u32) -> i32 {
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

    let mut gap_score: i32 = 0;
    let mut current_gap_size = 0;
    for i in 0..size {
        if table[i] != u64::max_value() {
            if current_gap_size >= 20 {
                gap_score += current_gap_size;
            }
            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }

    (leading_gap + trailing_gap) * 20 + gap_score
}

fn sparse_table_score(table: &[u64], shift: u32) -> i32 {
    let size = 1 << shift;
    let mut gaps: i32 = 0;

    let mut current_gap_size = 0;
    for i in 0..size {
        if table[i] != u64::max_value() {
            if current_gap_size >= 20 {
                gaps += 1;
            }
            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }

    gaps
}

fn initialize_rook_attacks(magic_candidates: &[[u64; 64]], rook_attacks: &mut Vec<[u64; 64 * 4096]>) {
    for i in 0..(magic_candidates.len()) {
        let mut offset = 0;

        for pos in 0..64 {
            let move_mask = gen_rook_attacks(0, pos);
            let block_mask = mask_without_outline(move_mask, pos as u32);

            let blocker_count = block_mask.count_ones();

            let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
            create_blocker_permutations(&mut permutations, 0, block_mask);

            let magic_num = magic_candidates[i as usize][pos as usize];

            for &p in permutations.iter() {
                let move_targets = gen_rook_attacks(p, pos);

                let index = (p.wrapping_mul(magic_num)) >> (64 - 12);
                rook_attacks[i as usize][index as usize + offset] = move_targets;
            }

            offset += 4096;
        }
    }
}

fn initialize_bishop_attacks(bishop_magics: &[[u64; 64]], bishop_attacks: &mut Vec<[u64; 64 * 512]>) {

    for i in 0..(bishop_magics.len()) {
        let mut offset = 0;
        let mut attacks = [u64::max_value(); 64 * 512];
        for pos in 0..64 {
            let move_mask = gen_bishop_attacks(0, pos);
            let block_mask = mask_without_outline(move_mask, pos as u32);

            let blocker_count = block_mask.count_ones();

            let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
            create_blocker_permutations(&mut permutations, 0, block_mask);


            let magic_num = bishop_magics[i][pos as usize];

            for &p in permutations.iter() {
                let move_targets = gen_bishop_attacks(p, pos);

                let index = (p.wrapping_mul(magic_num)) >> (64 - 9);
                attacks[index as usize + offset] = move_targets;
            }

            offset += 512;
        }
        bishop_attacks.push(attacks);
    }
}
