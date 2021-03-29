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

use std::collections::{VecDeque};
use std::cmp::{max};
use velvet::bitboard::{gen_rook_attacks, gen_bishop_attacks, create_blocker_permutations, mask_without_outline};
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use velvet::random::Random;
use std::{env, thread, fmt};
use std::process::exit;
use std::fs::File;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::sync::mpsc;
use std::sync::mpsc::Sender;
use std::str::FromStr;
use std::fmt::Formatter;

enum Command {
    SearchRookMagics(usize, fn(&[u64], u32) -> i32, String),
    SearchBishopMagics(usize, fn(&[u64], u32) -> i32, String),
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
                if args.len() < 5 {
                    print!("Missing arguments");
                    print_usage();
                    exit(0);
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

                let set_count = usize::from_str(&args[3]).expect("Could not prase set_count parameter");

                let output_file = args[4].to_string();

                if args[2] == "rook" {
                    Command::SearchRookMagics(set_count, score_fn, output_file)
                } else if args[2] == "bishop" {
                    Command::SearchBishopMagics(set_count, score_fn, output_file)
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
        Command::SearchRookMagics(set_size, score_fn, output_file) => {
            find_magics(PieceType::Rook, set_size, score_fn, output_file);
        },

        Command::SearchBishopMagics(set_size, score_fn, output_file) => {
            find_magics(PieceType::Bishop, set_size, score_fn, output_file);
        },

        Command::PackAttacks(rook_candidates, bishop_candidates, output_file) => {
            pack_attack_tables(rook_candidates, bishop_candidates, output_file);
        }
    }
}

fn print_usage() {
    println!("Commands:");
    println!("  search [sparse|dense] [rook|bishop] <set-count> <output-file>\n   \
    - searches for sets of magic numbers (sparse = many small gaps in the resulting attack table, dense = few bigger gaps)");
    println!("  pack <rook-candidates-input-file> <bishop-candidates-input-file <packed-magics-output-file>\n   \
    - finds magic number combinations and offsets for a minimal attack table size");
}

fn pack_attack_tables(rook_candidates_file: String, bishop_candidates_file: String, out_file: String) {
    println!("Reading rook magic candidates ...");
    let rook_magic_candidates = read_magic_candidates(rook_candidates_file);

    println!("Reading bishop magic candidates ...");
    let bishop_magic_candidates = read_magic_candidates(bishop_candidates_file);

    println!("Initializing rook attacks ...");
    let rook_attacks = initialize_attacks(12, &rook_magic_candidates, gen_rook_attacks);

    println!("Initializing bishop attacks ...");
    let bishop_attacks = initialize_attacks(9, &bishop_magic_candidates, gen_bishop_attacks);

    println!("Optimizing attack table ...");
    optimize_attacks(out_file, &rook_magic_candidates, &rook_attacks, &bishop_magic_candidates, &bishop_attacks);
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
fn optimize_attacks(output_file: String,
                    rook_magic_candidates: &[[u64; 64]], rook_attacks: &[BitVec],
                    bishop_magic_candidates: &[[u64; 64]], bishop_attacks: &[BitVec]) {

    let mut attacks = BitVec::new((4096 + 512) * 64);

    let rook_candidate_count = rook_magic_candidates.len();
    let bishop_candidate_count = bishop_magic_candidates.len();

    let mut lowest_max_index = i32::max_value();

    let mut rook_offsets: Vec<usize> = vec!(0; 64);
    let mut rook_candidate_selection: Vec<i32> = vec!(0; 64);

    let mut bishop_offsets: Vec<usize> = vec!(0; 64);
    let mut bishop_candidate_selection: Vec<i32> = vec!(0; 64);

    let mut scores = ScoreHistory::new(rook_candidate_count, bishop_candidate_count);

    let mut iterations = 0;
    let start_time = Instant::now();

    loop {
        attacks.clear();
        scores.clear_iteration();

        let mut max_index: i32 = 0;

        while !scores.is_iteration_finished() {
            let (src_pos, src_candidate) = scores.choose_next_pos();

            let src_attacks = if src_pos > 63 { &rook_attacks[src_candidate as usize] } else { &bishop_attacks[src_candidate as usize] };

            let offsets = if src_pos > 63 { &mut rook_offsets } else { &mut bishop_offsets };
            let candidates = if src_pos > 63 { &mut rook_candidate_selection } else { &mut bishop_candidate_selection };

            // Check position
            let size = if src_pos > 63 { 4096 } else { 512 };
            let src_offset = (src_pos & 63) as usize * size;
            let (index, start_index, end_index) = attacks.insert(src_attacks, src_offset, size);
            let local_max_index = (index + size - 1) as i32;

            let best_offset = index as i32 - src_offset as i32;

            offsets[src_pos as usize & 63] = (best_offset + (src_pos & 63) * size as i32) as usize;
            candidates[src_pos as usize & 63] = src_candidate;

            let prev_index = max_index;
            max_index = max(max_index, local_max_index);
            scores.add(src_pos, src_candidate, max_index, prev_index, start_index, end_index);

            if src_pos < 64 && max_index > (4096 * 3) {
                attacks.set_lower_bound(max_index as usize - (4096 * 3));
            }
        }

        scores.finish_iteration(&attacks);
        iterations += 1;

        if max_index < lowest_max_index {
            lowest_max_index = max_index;

            let duration = start_time.elapsed().as_millis();
            let ips = if duration > 0 { iterations * 1000 / duration } else { 0 };
            println!("Current Best {} - Stats: {} / {} ips ({})", lowest_max_index, scores.stats(), ips, iterations);

            let file = File::create(&output_file).expect("Could not create output file");
            let mut writer = BufWriter::new(file);

            write_heading(&mut writer, "Rooks");
            write_magics(&mut writer, &rook_candidate_selection, &rook_magic_candidates);
            write_offsets(&mut writer, &rook_offsets);

            write_heading(&mut writer, "Bishops");
            write_magics(&mut writer, &bishop_candidate_selection, &bishop_magic_candidates);
            write_offsets(&mut writer, &bishop_offsets);

            write_heading(&mut writer, "Attack table size");
            write!(writer, "{}", lowest_max_index + 1).expect("Could not write len to output file");
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

#[derive(Copy, Clone)]
enum PieceType {
    Rook,
    Bishop
}

#[derive(Copy, Clone)]
struct MagicResult {
    score: i32, pos: i32, magic: u64
}

fn find_magics(piece: PieceType, set_size: usize, magic_num_score: fn(&[u64], u32) -> i32, output_file: String) {
    println!("Searching magic numbers ...");
    let (tx, rx) = mpsc::channel::<MagicResult>();

    spawn_search_threads(&tx, set_size, piece, magic_num_score);

    let mut results: Vec<VecDeque<MagicResult>> = Vec::with_capacity(64);
    for _ in 0..64 {
        results.push(VecDeque::with_capacity(set_size));
    }

    for result in rx {
        let pos = result.pos as usize;

        if results[pos].len() < set_size {
            while results[pos].len() < set_size {
                results[pos].push_front(result);
            }
        } else {
            for i in 0..set_size {
                if result.score >= results[pos][i].score {
                    results[pos].insert(i, result);
                    results[pos].pop_back();
                    break;
                }
            }
        }

        let mut magics: Vec<[u64; 64]> = vec!([0; 64]; set_size);
        let mut best_scores: Vec<i64> = vec!(0; set_size);

        let mut found_magics_for_all_pos = true;
        for pos in 0..64 {
            if results[pos].is_empty() {
                found_magics_for_all_pos = false;
                break;
            }

            for set in 0..set_size {
                magics[set][pos] = results[pos][set].magic;
                best_scores[set] += results[pos][set].score as i64;
            }
        }

        if !found_magics_for_all_pos {
            continue;
        }

        let file = File::create(&output_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        for set in 0..set_size {
            let magics_str = magics[set]
                .iter()
                .map(|&n| format!("0x{:016x}", n))
                .collect::<Vec<String>>()
                .join(", ");

            writeln!(writer, "{}", magics_str).expect("Could not write magics to output file");
            writeln!(writer).expect("Could not write to output file");
        }

        println!("Scores: {}", best_scores.iter().map(|&s| format!("{:⋆>6}", (s / 10000))).collect::<Vec<_>>().join(", "));
    }
}

fn spawn_search_threads(tx: &Sender<MagicResult>, set_size: usize, piece: PieceType, magic_num_score: fn(&[u64], u32) -> i32) {
    for pos in 0..64 {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        thread::spawn(move || {
            match piece {
                PieceType::Rook =>
                    find_magics_for_pos(&tx2, set_size, pos, gen_rook_attacks, 12, 0xFF801FFFFFFFFFFF, magic_num_score),

                PieceType::Bishop =>
                    find_magics_for_pos(&tx2, set_size, pos, gen_bishop_attacks, 9, u64::max_value(), magic_num_score)
            }
        });
    }
}

fn find_magics_for_pos(tx: &Sender<MagicResult>, set_size: usize, pos: i32,
                       gen_attacks: fn(u64, i32) -> u64, shift: u32, magic_mask07: u64,
                       magic_num_score: fn(&[u64], u32) -> i32) {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    let mut r = Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64);

    // Special mask for squares 0 and 7 to speed up the search for valid magic numbers (only relevant for rooks)
    let magic_mask = if pos == 0 || pos == 7 { magic_mask07 } else { u64::max_value() };

    let move_mask = gen_attacks(0, pos);
    let tmp_block_mask = mask_without_outline(move_mask, pos as u32);

    let blocker_count = tmp_block_mask.count_ones();

    let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
    create_blocker_permutations(&mut permutations, 0, tmp_block_mask);

    let mut move_target_table: Vec<u64> = vec!(u64::max_value(); 1 << shift);

    let mut best_scores = VecDeque::<i32>::with_capacity(set_size);
    for _ in 0..set_size {
        best_scores.push_back(-1);
    }

    loop {
        let mut magic_num = r.rand64() & r.rand64() & magic_mask;

        let is_valid = validate_magic_num(&permutations, gen_attacks, &mut move_target_table, pos, magic_num, shift);
        if !is_valid {
            continue;
        }

        // Found a magic number candidate
        let gap_count = magic_num_score(&move_target_table, shift);

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

        if local_best_gap_count > *best_scores.back().unwrap() {
            for i in 0..set_size {
                if local_best_gap_count >= best_scores[i] {
                    best_scores.insert(i, local_best_gap_count);
                    best_scores.pop_back();
                    break;
                }
            }

            tx.send(MagicResult{score: local_best_gap_count, pos, magic: magic_num}).expect("Could not send result");
        }
    }
}

fn validate_magic_num(permutations: &[u64], gen_attacks: fn(u64, i32) -> u64, move_target_table: &mut Vec<u64>, pos: i32, magic_num: u64, shift: u32) -> bool {
    move_target_table.fill(u64::max_value());

    for &p in permutations.iter() {
        let move_targets = gen_attacks(p, pos);

        let index = ((!p).wrapping_mul(magic_num)) >> (64 - shift);
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

    let mut middle_gap_score: i32 = 0;
    let mut current_gap_size: i32 = 0;
    let mut middle_gap_count = 0;

    for i in start..end {
        if table[i] != u64::max_value() {
            if current_gap_size >= 16 {
                middle_gap_score += current_gap_size.pow(2);
                middle_gap_count += 1;
            }

            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }
    if current_gap_size >= 16 {
        middle_gap_score += current_gap_size.pow(2);
        middle_gap_count += 1;
    }

    if middle_gap_count > 0 {
        leading_gap.pow(2) + trailing_gap.pow(2) + middle_gap_score / middle_gap_count
    } else {
        leading_gap.pow(2) + trailing_gap.pow(2)
    }
}

fn sparse_table_score(table: &[u64], shift: u32) -> i32 {
    let size = 1 << shift;
    let mut gap_score: i32 = 0;
    let mut current_gap_size = 0;

    for i in 0..size {
        if table[i] != u64::max_value() {
            if current_gap_size >= 8 {
                gap_score += 1;
            }
            current_gap_size = 0;
        } else {
            current_gap_size += 1;
        }
    }

    if current_gap_size >= 8 {
        gap_score += 1;
    }

    gap_score * 1000
}

struct BitVec(pub Vec<u64>, usize, usize);

impl BitVec {
    pub fn new(len: usize) -> Self {
        BitVec(vec!(0; (len + 64 - 1) / 64), 0, 0)
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        let bit = index & 63;
        let element = index / 64;
        (self.0[element] & (1u64 << bit)) != 0
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        let bit = index & 63;
        let element = index / 64;
        self.0[element] |= 1u64 << bit;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.fill(0);
        self.1 = 0;
        self.2 = 0;
    }

    fn set_lower_bound(&mut self, low: usize) {
        self.2 = low;
    }

    fn update_lower_bound(&mut self) {
        let mut i = self.1 / 64;
        while self.0[i] == u64::max_value() {
            i += 1;
        }
        if i > 0 {
            self.1 = max(self.1, i * 64);
        }
    }

    fn insert(&mut self, source: &BitVec, source_offset: usize, width: usize) -> (usize, usize, usize) {

        self.update_lower_bound();

        let source_dword = source_offset / 64;

        let mut offset = if width == 4096 { max(self.1, self.2) } else { self.1 };

        loop {

            let offset_dword = offset / 64;
            let offset_bitshift = offset & 63;

            let mut rem_width = width;
            let mut i = 0;
            let mut collision = false;
            while rem_width > 0 {
                rem_width -= 64;

                let source_item = source.0[source_dword + i];
                let target_item = self.0[offset_dword + i] as u64;
                let source_item_left = source_item << offset_bitshift;

                if source_item_left & target_item != 0 {
                    collision = true;
                    break;
                }

                i += 1;

                if offset_bitshift == 0 {
                    continue;
                }

                let target_next_item = self.0[offset_dword + i] as u64;
                let source_item_right = source_item >> (64 - offset_bitshift);

                if source_item_right & target_next_item != 0 {
                    collision = true;
                    break;
                }
            }

            if !collision {
                // Apply
                let mut min_index = offset + width;
                let mut max_index = offset;
                for i in 0..width {
                    if source.get(source_offset + i) {
                        self.set(offset + i);
                        if offset + i < min_index {
                            min_index = offset + i;
                        }
                        max_index = offset + i;
                    }
                }

                return (offset, min_index, max_index);
            }

            offset += 1;
        }
    }
}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "BitVec[{}]", self.0.iter()
            .map(|&n| format!("{:064b}", n))
            .collect::<Vec<String>>()
            .join(", "))
    }
}

fn initialize_attacks(shift: usize, magic_candidate_sets: &[[u64; 64]], gen_attacks: fn(u64, i32) -> u64) -> Vec<BitVec> {
    let size = 1 << shift;

    let mut attacks = Vec::<BitVec>::with_capacity(magic_candidate_sets.len());
    for &magic_candidates in magic_candidate_sets.iter() {
        let mut offset = 0;
        let mut attack_bits = BitVec::new(size * 64);

        for pos in 0..64 {
            let move_mask = gen_attacks(0, pos);
            let block_mask = mask_without_outline(move_mask, pos as u32);

            let blocker_count = block_mask.count_ones();

            let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
            create_blocker_permutations(&mut permutations, 0, block_mask);

            let magic_num = magic_candidates[pos as usize];

            for &p in permutations.iter() {
                let index = ((!p).wrapping_mul(magic_num)) >> (64 - shift);
                attack_bits.set(index as usize + offset);
            }

            offset += size;
        }
        attacks.push(attack_bits);
    }

    attacks
}

#[derive(Copy, Clone)]
struct Selection {
    pos: i32,
    candidate: i32,
    max_index: i32,
    prev_index: i32,
    insert_start: usize,
    insert_end: usize,
}

struct ScoreHistory {
    rnd: Random,
    finished: Vec<bool>,
    finished_count: i16,
    finished_rooks: i16,
    scores: Vec<i32>,
    counts: Vec<i64>,
    index_multiplier: usize,
    selection: Vec<Selection>,
    rook_candidate_count: usize,
    bishop_candidate_count: usize,
    max_candidate_count: usize,
    random_count: u64,
    selected_count: u64,
}

impl ScoreHistory {
    fn new(rook_candidate_count: usize, bishop_candidate_count: usize) -> ScoreHistory {
        let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d,
            Err(e) => panic!("Duration time error: {}", e)
        };

        let max_candidate_count = rook_candidate_count + bishop_candidate_count;

        let index_multiplier = (128 + 1) * (max_candidate_count + 1);

        ScoreHistory{
            rnd: Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64),
            finished: vec!(false; 128),
            finished_count: 0,
            finished_rooks: 0,
            scores: vec!(UNCHECKED; index_multiplier * index_multiplier),
            counts: vec!(0; index_multiplier * index_multiplier),
            index_multiplier,
            selection: Vec::with_capacity(65),
            rook_candidate_count,
            bishop_candidate_count,
            max_candidate_count,
            random_count: 0,
            selected_count: 0,
        }
    }

    fn clear_iteration(&mut self) {
        self.selection.clear();
        self.selection.push(Selection{pos: -1, candidate: -1, max_index: 0, prev_index: 0, insert_start: 0, insert_end: 0 });
        self.finished.fill(false);
        self.finished_count = 0;
        self.finished_rooks = 0;
    }

    fn is_iteration_finished(&self) -> bool {
        self.finished_count >= 128
    }

    fn index(&self, prev_pos: i32, prev_candidate: i32, pos: i32, candidate: i32) -> usize {
        ((prev_pos + 1) as usize * (self.max_candidate_count + 1) + (prev_candidate + 1) as usize) * self.index_multiplier
            + (pos as usize * (self.max_candidate_count + 1) + candidate as usize)
    }

    fn choose_next_pos(&mut self) -> (i32, i32) {
        let finished_bishops = self.finished_count - self.finished_rooks;
        let skip_bishops = if self.finished_rooks == 0 || finished_bishops == 64 { true } else if self.finished_rooks < 64 { self.rnd.rand32() & 31 > 2 } else { false };

        let prev = &self.selection[self.selection.len() - 1];
        let prev_pos = prev.pos;
        let prev_candidate = prev.candidate;

        let mut candidate_scores = Vec::<(i32, i32, i32)>::with_capacity(64 * self.max_candidate_count);

        if skip_bishops {
            for pos in 64..128 {
                if self.finished[pos as usize] {
                    continue;
                }

                for candidate in 0..self.rook_candidate_count {
                    let i = self.index(prev_pos, prev_candidate, pos, candidate as i32);
                    let score = self.scores[i];
                    candidate_scores.push((score, pos, candidate as i32));
                }
            }
        } else {
            for pos in 0..64 {
                if self.finished[pos as usize] {
                    continue;
                }

                for candidate in 0..self.bishop_candidate_count {
                    let i = self.index(prev_pos, prev_candidate, pos, candidate as i32);
                    let score = self.scores[i];
                    candidate_scores.push((score, pos, candidate as i32));
                }
            }
        }

        candidate_scores.sort_unstable_by_key(|e| e.0);
        if candidate_scores[0].0 == UNCHECKED {
            self.random_count += 1;
            return (candidate_scores[0].1, candidate_scores[0].2)
        }

        let min_score = candidate_scores.iter().map(|i| i.0 as i64).min().unwrap() as i32;
        let avg_score = (candidate_scores.iter().map(|i| i.0 as i64).sum::<i64>() / candidate_scores.len() as i64) as i32;

        let step_size = if min_score > 0 { max(1, (avg_score / min_score).abs()) } else { 1 };

        let mut threshold = min_score;

        loop {
            let i = self.rnd.rand32() as usize % candidate_scores.len();

            if candidate_scores[i].0 == UNCHECKED {
                self.random_count += 1;
                return (candidate_scores[i].1, candidate_scores[i].2)
            }

            if candidate_scores[i].0 > threshold {
                threshold += step_size;
                continue;
            }

            self.selected_count += 1;
            return (candidate_scores[i].1, candidate_scores[i].2)
        }

    }

    fn add(&mut self, pos: i32, candidate: i32, max_index: i32, prev_index: i32, insert_start: usize, insert_end: usize) {

        self.selection.push(Selection{pos, candidate, max_index, prev_index, insert_start, insert_end});

        self.finished[pos as usize] = true;
        self.finished_count += 1;

        if pos > 63 {
            self.finished_rooks += 1;
        }
    }

    fn finish_iteration(&mut self, attacks: &BitVec) {
        let max_index = self.selection[self.selection.len() - 1].max_index;
        let base_score = max_index - 80000;

        for (prev, curr) in self.selection.iter().zip(self.selection.iter().skip(1)) {
            let mut gap_score: i32 = 0;
            for i in curr.insert_start..curr.insert_end {
                if !attacks.get(i) {
                    gap_score += 1;
                }
            }
            for i in prev.insert_start..prev.insert_end {
                if !attacks.get(i) {
                    gap_score += 1;
                }
            }

            let curr_width = curr.insert_end - curr.insert_start;
            let prev_width = prev.insert_end - prev.insert_start;

            let score = base_score + gap_score + (curr.max_index - prev.prev_index).abs() - (curr_width + prev_width) as i32;

            let index = self.index(prev.pos, prev.candidate, curr.pos, curr.candidate);
            let all_scores = (self.scores[index]) as i64;
            let score_count = self.counts[index];
            let new_score = ((all_scores * score_count + score as i64) / (score_count + 1)) as i32;
            self.scores[index] = new_score;
            self.counts[index] += 1;
        }
    }

    fn stats(&self) -> String {
        let total_count = self.selected_count + self.random_count;
        if total_count == 0 {
            return "".to_string()
        }

        format!("{:4} / {:4}", (self.selected_count * 1000) / total_count, (self.random_count * 1000) / total_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_bitvec_insert() {
        let mut target = BitVec::new(2 * 64);

        for i in 0..60 {
            target.set(i);
        }

        target.set(62);
        target.set(64);
        target.set(66);
        target.set(68);

        let mut source = BitVec::new(2 * 64);

        source.set(3);
        source.set(5);
        source.set(7);

        let (index, _, _) = target.insert(&source, 0, 64);

        assert_eq!(index, 58);
    }
}
