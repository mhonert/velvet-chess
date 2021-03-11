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
use std::fs::File;
use std::io::{BufReader, BufRead, BufWriter, Write};

enum Command {
    SearchRookMagics(i32, fn(&[u64], u32) -> i32),
    SearchBishopMagics(i32, fn(&[u64], u32) -> i32),
    PackAttacks(String, String, String),
}

fn main() {
    let args: Vec<String> = env::args().map(|s| s.to_lowercase()).skip(1).collect();
    if args.is_empty() {
        print_usage();
        return;
    }

    let cmd: Command =
        match args[0].as_str() {
            "packattacks" => {
                if args.len() < 4 {
                    print!("Missing file arguments");
                    print_usage();
                    exit(0);
                }
                Command::PackAttacks(args[1].to_string(), args[2].to_string(), args[3].to_string())
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
        Command::SearchRookMagics(pos, score_fn) => {
            find_rook_magics(pos, score_fn);
        },

        Command::SearchBishopMagics(pos, score_fn) => {
            find_bishop_magics(pos, score_fn);
        },

        Command::PackAttacks(rook_candidates, bishop_candidates, output_file) => {
            pack_attack_tables(rook_candidates, bishop_candidates, output_file);
        }
    }
}

fn print_usage() {
    println!("Commands:");
    println!("  search [sparse|dense] [rook|bishop] {{pos}}\n   - searches for magics for a specific board square (0-63)");
    println!("  pack <rook-candidates-input-file> <bishop-candidates-input-file <packed-magics-output-file>\n   - finds magic number combinations and offsets for a minimal attack table size");
}

fn pack_attack_tables(rook_candidates_file: String, bishop_candidates_file: String, out_file: String) {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    let mut rnd = Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64);

    println!("Reading rook magic candidates ...");
    let rook_magic_candidates = read_magic_candidates(rook_candidates_file);
    let rook_candidate_count = rook_magic_candidates.len();

    println!("Reading bishop magic candidates ...");
    let bishop_magic_candidates = read_magic_candidates(bishop_candidates_file);
    let bishop_candidate_count = bishop_magic_candidates.len();
    println!("{} bishop magic candidates", bishop_candidate_count);

    println!("Initializing rook attacks ...");
    let mut rook_attacks: Vec<Vec<u64>> = Vec::with_capacity(rook_candidate_count);
    initialize_rook_attacks(&rook_magic_candidates, &mut rook_attacks);

    println!("Initializing bishop attacks ...");
    let mut bishop_attacks: Vec<Vec<u64>> = Vec::with_capacity(bishop_candidate_count);
    initialize_bishop_attacks(&bishop_magic_candidates, &mut bishop_attacks);

    let mut attacks: Vec<u64> = vec!(u64::max_value(); 4096 * 64);

    println!("Optimizing attack table ...");
    optimize_attacks(out_file, &mut rnd, &rook_magic_candidates, &rook_attacks, &bishop_magic_candidates, &bishop_attacks, &mut attacks);
}

fn read_magic_candidates(file_name: String) -> Vec<[u64; 64]> {
    let mut candidates: Vec<[u64; 64]> = Vec::new();
    let file = File::open(file_name).expect("Unable to open magics candidates file");
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.expect("Could not read line");
        if line.trim().is_empty() || line.trim().starts_with('#') {
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

    if candidates.is_empty() {
        panic!("file did not contain any magic number candidates");
    }

    candidates
}

fn parse_magics_str(magics: String) -> Vec<u64> {
    magics.splitn(64, ',')
        .map(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).expect(&format!("Could not parse magic number: {}", s)))
        .collect()
}

const UNCHECKED: i32 = i32::min_value();

/// optimize_attacks compares different combinations of rook and bishop magic numbers with the
/// goal of finding a minimal, combined attack table.
/// The size reduction is achieved by overlapping the attack table entries without collisions.
/// The algorithm will start with random combinations of magic numbers and score pairs of magic numbers.
/// In following iterations the random selection will favor combinations with a lower score
fn optimize_attacks(output_file: String, rnd: &mut Random,
                    rook_magic_candidates: &[[u64; 64]], rook_attacks: &[Vec<u64>],
                    bishop_magic_candidates: &[[u64; 64]], bishop_attacks: &[Vec<u64>],
                    attacks: &mut Vec<u64>) {

    let rook_candidate_count = rook_magic_candidates.len();
    let bishop_candidate_count = bishop_magic_candidates.len();
    let candidate_count = rook_candidate_count + bishop_candidate_count;

    let idx = indexer(candidate_count as i32);

    let mut lowest_max_index = i32::max_value();

    let mut follow_up_scores: Vec<i32> = vec!(UNCHECKED; candidate_count * 128 * candidate_count * 128);
    let mut score_counts: Vec<i64> = vec!(0; candidate_count * 128 * candidate_count * 128);

    let mut current_index = 0;

    let mut selection: Vec<(usize, i32)> = Vec::with_capacity(64);

    let mut rook_offsets: Vec<usize> = vec!(0; 64);
    let mut rook_candidate_selection: Vec<i32> = vec!(0; 64);

    let mut bishop_offsets: Vec<usize> = vec!(0; 64);
    let mut bishop_candidate_selection: Vec<i32> = vec!(0; 64);

    let mut random_count: u64 = 0;
    let mut selected_count: u64 = 0;

    loop {
        attacks.iter_mut().take(current_index as usize).for_each(|e| *e = u64::max_value());
        selection.clear();

        current_index = 0;

        let mut max_index: i32 = 0;
        let mut found_pos = false;
        let mut best_offset = 0;
        let mut finished: HashSet<i32> = HashSet::new();
        let mut prev_pos: i32 = -1;
        let mut prev_candidate: i32 = -1;
        let mut prev_index: i32 = -1;

        while finished.len() < 128 {
            let mut lowest_index = i32::max_value();
            let mut src_pos;
            let mut src_candidate;


            // Find next position to check
            let mut min_score = 0;
            if prev_pos >= 0 {
                min_score = 16384;
                for sp in 0..128 {
                    let cc = if sp > 63 { rook_candidate_count } else { bishop_candidate_count };
                    for sc in 0..cc {
                        let i = idx(prev_pos, prev_candidate, sp, sc as i32);
                        let score = follow_up_scores[i];
                        if score != UNCHECKED {
                            min_score = min(score, min_score);
                        }
                    }
                }
            }

            let mut follow_up_threshold = min_score;

            loop {
                let r = rnd.rand32();
                src_pos = (r & 127) as i32;
                let cc = if src_pos > 63 { rook_candidate_count } else { bishop_candidate_count };
                src_candidate = ((r / 128) % cc as u32) as i32;

                if finished.contains(&src_pos) {
                    continue;
                }

                if !finished.is_empty() {

                    let i = idx(prev_pos, prev_candidate, src_pos, src_candidate);

                    let follow_up_score = follow_up_scores[i];
                    if follow_up_score == UNCHECKED {
                        random_count += 1;
                        break;
                    }

                    if follow_up_score > follow_up_threshold {
                        follow_up_threshold += 1;
                        continue;
                    }

                    selected_count += 1;
                    break;
                }

                break;
            }

            let size = if src_pos > 63 { 4096 } else { 512 };

            let src_offset = (src_pos & 63) as usize * size;
            let src_attacks = if src_pos > 63 { &rook_attacks[src_candidate as usize] } else { &bishop_attacks[src_candidate as usize] };

            let offsets = if src_pos > 63 { &mut rook_offsets } else { &mut bishop_offsets };
            let candidates = if src_pos > 63 { &mut rook_candidate_selection } else { &mut bishop_candidate_selection };

            // Check position
            found_pos = false;
            for offset_adj in (-(src_offset as i32))..(current_index - (src_offset as i32) + size as i32) {
                let mut is_valid = true;
                let start_index = src_offset as i32 + offset_adj;
                let end_index = src_offset as i32 + offset_adj + (size as i32 - 1);
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

            offsets[src_pos as usize & 63] = (best_offset + (src_pos & 63) * size as i32) as usize;
            candidates[src_pos as usize & 63] = src_candidate;
            finished.insert(src_pos);

            if prev_pos >= 0 {
                let i = idx(prev_pos, prev_candidate, src_pos, src_candidate);

                selection.push((i, lowest_index - prev_index));
            }

            prev_index = current_index;
            current_index = lowest_index + 1;
            prev_pos = src_pos;
            prev_candidate = src_candidate;
        }


        if found_pos {
            // Adjust follow up scores
            let total_length_score = (max_index - 100000) / 256;
            for s in selection.iter() {
                let score = (total_length_score + s.1) as i64;
                let all_scores = (follow_up_scores[s.0]) as i64;
                let score_count = score_counts[s.0];
                follow_up_scores[s.0] = ((all_scores * score_count + score) / (score_count + 1)) as i32;
                score_counts[s.0] += 1;
            }
        }

        if found_pos && max_index < lowest_max_index {
            lowest_max_index = max_index;

            let total_count = selected_count + random_count;

            println!("Current Best {} - Stats: {:4} / {:4}",
                     lowest_max_index,
                     (selected_count * 1000) / total_count, (random_count * 1000) / total_count);

            let file = File::create(&output_file).expect("Could not create output file");
            let mut writer = BufWriter::new(file);

            write_heading(&mut writer, "Rooks");
            write_magics(&mut writer, &rook_candidate_selection, &rook_magic_candidates);
            write_offsets(&mut writer, &rook_offsets);

            write_heading(&mut writer, "Bishops");
            write_magics(&mut writer, &bishop_candidate_selection, &bishop_magic_candidates);
            write_offsets(&mut writer, &bishop_offsets);

            write_heading(&mut writer, "Attack table size");
            write!(writer, "{}", lowest_max_index).expect("Could not write len to output file");

            selected_count = 0;
            random_count = 0;
        }

    }
}

fn write_magics(writer: &mut BufWriter<File>, candidate_selections: &[i32], magic_candidates: &[[u64; 64]]) {
    let magic_nums: Vec<u64> = candidate_selections
        .iter()
        .enumerate()
        .map(|(i, &c)| magic_candidates[c as usize][i])
        .collect();

    let magics_str = magic_nums
        .iter()
        .map(|&n| format!("0x{:016x}", n))
        .collect::<Vec<String>>()
        .join(", ");

    writeln!(writer, "{}", magics_str).expect("Could not write magics to output file");
}

fn write_offsets(writer: &mut BufWriter<File>, offsets: &[usize]) {
    writeln!(writer, "{}", offsets.iter().map(|&n| format!("{}", n)).collect::<Vec<String>>().join(", "))
        .expect("Could not write offsets to output file");
}

fn write_heading(writer: &mut BufWriter<File>, label: &str) {
    writeln!(writer, "--- {} {}", label, String::from("-").repeat(80)).expect("Could not write to file");
}

fn indexer(candidate_count: i32) -> impl Fn (i32, i32, i32, i32) -> usize {
    move |prev_pos: i32, prev_candidate: i32, pos: i32, candidate: i32| {
        (prev_pos
            + prev_candidate * 128
            + pos * 128 * candidate_count
            + candidate * 128 * candidate_count * 128) as usize
    }
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
    let mut start = 0;
    for i in 0..size {
        if table[i] != u64::max_value() {
            start = i;
            break;
        }
        leading_gap += 1;
    }

    let mut trailing_gap: i32 = 0;
    let mut end= size;
    for i in (0..size).rev() {
        if table[i] != u64::max_value() {
            end = i;
            break;
        }
        trailing_gap += 1;
    }

    let mut biggest_gap = 0;
    let mut current_gap_size = 0;
    for i in start..end {
        if table[i] != u64::max_value() {
            biggest_gap = max(biggest_gap, current_gap_size);
            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }

    (leading_gap + trailing_gap).pow(2) + biggest_gap
}

fn sparse_table_score(table: &[u64], shift: u32) -> i32 {
    let size = 1 << shift;
    let mut gap_score: i32 = 0;

    let mut current_gap_size = 0;
    for i in 0..size {
        if table[i] != u64::max_value() {
            if current_gap_size >= 16 {
                gap_score += current_gap_size * current_gap_size;
            }
            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }

    gap_score
}

fn initialize_rook_attacks(magic_candidates: &[[u64; 64]], rook_attacks: &mut Vec<Vec<u64>>) {
    for i in 0..(magic_candidates.len()) {
        let mut offset = 0;
        let mut attacks = vec!(u64::max_value(); 64 * 4096);

        for pos in 0..64 {
            let move_mask = gen_rook_attacks(0, pos);
            let block_mask = mask_without_outline(move_mask, pos as u32);

            let blocker_count = block_mask.count_ones();

            let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
            create_blocker_permutations(&mut permutations, 0, block_mask);

            let magic_num = magic_candidates[i][pos as usize];

            for &p in permutations.iter() {
                let move_targets = gen_rook_attacks(p, pos);

                let index = (p.wrapping_mul(magic_num)) >> (64 - 12);
                attacks[index as usize + offset] = move_targets;
            }

            offset += 4096;
        }
        rook_attacks.push(attacks);
    }
}

fn initialize_bishop_attacks(magic_candidates: &[[u64; 64]], bishop_attacks: &mut Vec<Vec<u64>>) {
    for i in 0..(magic_candidates.len()) {
        let mut offset = 0;
        let mut attacks = vec!(u64::max_value(); 64 * 512);
        for pos in 0..64 {
            let move_mask = gen_bishop_attacks(0, pos);
            let block_mask = mask_without_outline(move_mask, pos as u32);

            let blocker_count = block_mask.count_ones();

            let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
            create_blocker_permutations(&mut permutations, 0, block_mask);

            let magic_num = magic_candidates[i][pos as usize];

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
