/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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

use itertools::{Itertools};
use std::cmp::{min};
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Error, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use lz4_flex::frame::FrameEncoder;
use velvet::bitboard::{BitBoard};
use velvet::colors::Color;
use velvet::fen::{parse_fen, FenParseResult};
use velvet::nn::{piece_idx, KING_BUCKETS, SCORE_SCALE, king_bucket};
use velvet::nn::io::{read_f32, read_u16, read_u64, read_u8, write_f32, write_u16, write_u64, write_u8};

pub const SAMPLES_PER_SET: usize = 200_000;

pub const K: f64 = 1.603;

pub trait DataSamples {
    fn init(&mut self, idx: usize, result: f32, stm: u8);
    fn add_wpov(&mut self, idx: usize, pos: u16);
    fn add_bpov(&mut self, idx: usize, pos: u16);
    fn finalize(&mut self, idx: usize);
}

pub fn convert_sets(thread_count: usize, caption: &str, in_path: &str, out_path: &str, min_id: usize, use_game_result: bool, allow_transformations: bool) -> usize {
    let mut max_converted_id = 0;
    let mut max_set_id = 0;
    let mut min_unconverted_id = usize::MAX;

    for id in min_id..usize::MAX {
        if Path::new(&format!("{}/{}.lz4", out_path, id)).exists() {
            max_set_id = max_set_id.max(id);
            max_converted_id = id;
        } else if Path::new(&format!("{}/test_pos_{}.fen", in_path, id)).exists() {
            max_set_id = max_set_id.max(id);
            min_unconverted_id = min(id, min_unconverted_id);
        } else {
            break;
        }
    }

    if max_converted_id < max_set_id {
        println!("Converting {} added {} sets ...", (max_set_id - min_unconverted_id + 1), caption);
        max_set_id = convert_test_pos(thread_count, in_path.to_string(), out_path.to_string(), min_unconverted_id, max_set_id, use_game_result, allow_transformations)
            .expect("Could not convert test positions!");
    }

    max_set_id
}


struct OutputWriter {
    output_dir: String,
    next_set_id: Arc<AtomicUsize>,
    entries: Vec<Entry>,
    allow_transformations: bool,
}

struct Entry {
    bb_map: u16,
    score: f32,
    active_player: Color,
    white_king_pos: u8,
    black_king_pos: u8,
    bb: [u64; 13],
}

impl OutputWriter {
    fn new(output_dir: String, next_set_id: &Arc<AtomicUsize>, allow_transformations: bool) -> Self {
        OutputWriter{output_dir, next_set_id: next_set_id.clone(), entries: Vec::with_capacity(200000), allow_transformations}
    }

    pub fn write(&mut self, bb_map: u16, white_king_pos: u8, black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
        self.add_entry(bb_map, white_king_pos, black_king_pos, active_player, score, bb);

        if !self.allow_transformations {
            return;
        }

        let any_castling_rights = bb_map & (1 << 15) != 0;
        if any_castling_rights {
            return;
        }

        let white_has_pawns = bb_map & (1 << 0) != 0;
        let black_has_pawns = bb_map & (1 << 5) != 0;
        if white_has_pawns || black_has_pawns {
            return;
        }

        self.add_transformed_entry(rotate90_ccw_u16, bb_map, white_king_pos, black_king_pos, active_player, score, bb);
        self.add_transformed_entry(v_mirror_u16, bb_map, white_king_pos, black_king_pos, active_player, score, bb);
        self.add_transformed_entry(mirror_diagonal_u16, bb_map, white_king_pos, black_king_pos, active_player, score, bb);
    }

    fn add_entry(&mut self, bb_map: u16, white_king_pos: u8, black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
        self.entries.push(Entry{bb_map, score, active_player, white_king_pos, black_king_pos,bb});
        if self.entries.len() == 200000 {
            self.flush_entries();
        }
    }

    fn flush_entries(&mut self) {
        let mut writer = next_file(&self.output_dir.clone(), self.next_set_id.clone().fetch_add(1, Ordering::Relaxed));
        for entry in self.entries.drain(0..self.entries.len()) {
            write_u16(&mut writer, entry.bb_map).unwrap();
            write_f32(&mut writer, if entry.active_player.is_white() { entry.score } else { -entry.score }).unwrap();
            write_u8(&mut writer, entry.active_player.0).unwrap();

            let kings = entry.white_king_pos as u16 | ((entry.black_king_pos as u16) << 8);
            write_u16(&mut writer, kings).unwrap();

            for i in 1i8..=5i8 {
                let bb_white = entry.bb[(i + 6) as usize];
                if bb_white != 0 {
                    write_u64(&mut writer, bb_white).unwrap();
                }

                let bb_black = entry.bb[(-i + 6) as usize];
                if bb_black != 0 {
                    write_u64(&mut writer, bb_black).unwrap();
                }
            }
        }
        write_u16(&mut writer, u16::MAX).unwrap();
        writer.flush().unwrap();
    }

    fn add_transformed_entry(&mut self, transform: fn(u16) -> u16, bb_map: u16, mut white_king_pos: u8, mut black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
        white_king_pos = transform(white_king_pos as u16) as u8;
        black_king_pos = transform(black_king_pos as u16) as u8;

        let mut bb_transformed = [0u64; 13];

        for i in 1i8..=5i8 {
            let bb_white = bb[(i + 6) as usize];
            let bb_white_target = bb_transformed.get_mut((i + 6) as usize).unwrap();
            for pos in BitBoard(bb_white) {
                let pos_transformed = transform(pos as u16);
                *bb_white_target |= 1 << pos_transformed;
            }

            let bb_black = bb[(-i + 6) as usize];
            let bb_black_target = bb_transformed.get_mut((-i + 6) as usize).unwrap();
            for pos in BitBoard(bb_black) {
                let pos_transformed = transform(pos as u16);
                *bb_black_target |= 1 << pos_transformed;
            }
        }

        self.add_entry(bb_map, white_king_pos, black_king_pos, active_player, score, bb_transformed);
    }
}

fn next_file(path: &str, set_nr: usize) -> BufWriter<FrameEncoder<File>> {
    let file_name = format!("{}/{}.lz4", path, set_nr);
    if Path::new(&file_name).exists() {
        panic!("Output file already exists: {}", file_name);
    }
    let file = File::create(&file_name).expect("Could not create output file");
    let encoder = FrameEncoder::new(file);
    BufWriter::with_capacity(1024 * 1024, encoder)
}

fn convert_test_pos(
    thread_count: usize, in_path: String, out_path: String, min_unconverted_id: usize, max_training_set_id: usize,
    use_game_result: bool, allow_transformations: bool,
) -> Result<usize, Error> {
    let mut threads = Vec::new();
    let next_set_id = Arc::new(AtomicUsize::new(1));

    for c in 1..=thread_count {
        let in_path2 = in_path.clone();
        let out_path2 = out_path.clone();
        let next_set_id2 = next_set_id.clone();
        threads.push(thread::spawn(move || {
            let mut output_writer = OutputWriter::new(out_path2, &next_set_id2, allow_transformations);

            for i in ((c + min_unconverted_id - 1)..=max_training_set_id).step_by(thread_count) {
                print!("{} ", i);
                stdout().flush().unwrap();

                read_from_fen_file(format!("{}/test_pos_{}.fen", in_path2, i).as_str(), &mut output_writer, use_game_result);
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("\nConversion finished");

    Ok(next_set_id.load(Ordering::Relaxed).saturating_sub(1))
}

fn read_from_fen_file(file_name: &str, writer: &mut OutputWriter, _use_game_result: bool) {
    let file = File::open(file_name).expect("Could not open test position file");
    let mut reader = BufReader::new(file);

    let mut line = String::with_capacity(256);
    loop {
        line.clear();

        match reader.read_line(&mut line) {
            Ok(read) => {
                if read == 0 {
                    return;
                }
            }

            Err(e) => panic!("Reading test position file failed: {}", e),
        };

        let parts = line.trim_end().split(' ').collect_vec();

        let score = if parts.len() == 7 {
            // 0..5 | 6
            // fen  | score
            let score = i32::from_str(parts[6]).expect("Could not parse score");
            score as f32 / SCORE_SCALE as f32

        } else if parts.len() == 11 {
            let score = i32::from_str(parts[10]).expect("Could not parse score");
            score as f32 / SCORE_SCALE as f32

        } else if parts.len() == 12 {
            let score = i32::from_str(parts[9]).expect("Could not parse score");
            score as f32 / SCORE_SCALE as f32

        } else if parts.len() == 13 {
            let score = i32::from_str(parts[12]).expect("Could not parse score");
            score as f32 / SCORE_SCALE as f32

        } else {
                panic!("Invalid test position entry: {}", line);
        };

        let fen: String = (parts[0..=5].join(" ") as String).replace('~', "");

        let (pieces, active_player, any_castling) = match parse_fen(fen.as_str()) {
            Ok(FenParseResult { pieces, active_player, castling_state, .. }) => (pieces, active_player, castling_state.any_castling()),
            Err(e) => panic!("could not parse FEN: {}", e),
        };

        let mut black_king_pos = 0;
        let mut white_king_pos = 0;
        let mut bb: [u64; 13] = [0; 13];

        for (pos, piece) in pieces.iter().enumerate() {
            if *piece == 0 {
                continue;
            }

            if *piece == -6 {
                black_king_pos = pos;
            } else if *piece == 6 {
                white_king_pos = pos;
            } else {
                bb[(*piece + 6) as usize] |= 1 << pos;
            }
        }

        let mut bb_map = 0;
        if any_castling {
            bb_map |= 1 << 15;
        }

        for i in 1i8..=5i8 {
            if bb[(i + 6) as usize] != 0 {
                bb_map |= 1 << (i - 1);
            }
            if bb[(-i + 6) as usize] != 0 {
                bb_map |= 1 << (i + 4);
            }
        }

        writer.write(bb_map, white_king_pos as u8, black_king_pos as u8, active_player, score, bb);
    }
}

pub fn read_samples<T: DataSamples>(samples: &mut T, start: usize, file_name: &str) {
    let file = File::open(file_name).unwrap_or_else(|_| panic!("Could not open test position file: {}", file_name));
    let decoder = lz4_flex::frame::FrameDecoder::new(file);
    let mut reader = BufReader::new(decoder);

    let mut total_samples = 0;

    let mut idx = start;

    loop {
        let bb_map = read_u16(&mut reader).unwrap();
        if bb_map == u16::MAX {
            break;
        }
        total_samples += 1;

        let score = read_f32(&mut reader).unwrap();
        let stm = read_u8(&mut reader).unwrap();
        samples.init(idx, score, stm);


        let kings = read_u16(&mut reader).unwrap();
        let white_king = kings & 0b111111;
        let black_king = kings >> 8;

        let white_no_queens = bb_map & (1 << 4) == 0;
        let black_no_queens = bb_map & (1 << 9) == 0;
        let white_no_rooks = bb_map & (1 << 3) == 0;
        let black_no_rooks = bb_map & (1 << 8) == 0;
        // let white_no_bishops = bb_map & (1 << 2) == 0;
        // let black_no_bishops = bb_map & (1 << 7) == 0;
        // let white_no_pawns = bb_map & (1 << 0) == 0;
        // let black_no_pawns = bb_map & (1 << 5) == 0;

        let white_king_col = white_king & 7;
        let black_king_col = black_king & 7;
        let mirror_white_pov = white_king_col > 3;
        let mirror_black_pov = black_king_col > 3;

        let transform_wpov = if mirror_white_pov {
            h_mirror_u16
        } else {
            |pos| pos
        };

        let transform_bpov = if mirror_black_pov {
            |pos| v_mirror_u16(h_mirror_u16(pos))
        } else {
            v_mirror_u16
        };

        let w_kingrel_bucket = king_bucket(transform_wpov(white_king));
        let b_kingrel_bucket = king_bucket(transform_bpov(black_king));

        let piece_bucket = if white_no_queens && black_no_queens {
            if white_no_rooks && black_no_rooks { 2 } else { 1 }
        } else {
            0
        };

        let w_bucket: u16 = piece_bucket * KING_BUCKETS as u16 + w_kingrel_bucket;
        let b_bucket: u16 = piece_bucket * KING_BUCKETS as u16 + b_kingrel_bucket;

        let bucket_size = 6 * 64 * 2;

        let (white_offset, black_offset) = (
                w_bucket * bucket_size as u16,
                b_bucket * bucket_size as u16,
        );

        let opp_offset = 64;

        samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(white_king) + white_offset);
        samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(black_king) + white_offset + opp_offset);

        samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(black_king) + black_offset);
        samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(white_king) + black_offset + opp_offset);

        for i in 1i8..=5i8 {
            if bb_map & (1 << (i - 1)) != 0 {
                // White pieces
                let bb = read_u64(&mut reader).unwrap();
                for pos in BitBoard(bb) {
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(pos as u16)
                        + white_offset);
                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(pos as u16)
                        + black_offset + opp_offset);
                }
            }

            if bb_map & (1 << (i as usize + 4)) != 0 {
                // Black pieces
                let bb = read_u64(&mut reader).unwrap();
                for pos in BitBoard(bb) {
                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(pos as u16)
                        + black_offset);
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(pos as u16)
                        + white_offset + opp_offset);
                }
            }
        }
        samples.finalize(idx);
        idx += 1;
    }

    if total_samples != SAMPLES_PER_SET {
        panic!("File does not contain the expected 200_000 samples, but: {}", total_samples);
    }
}

/// Rotates the given bitboard position by 90 degree counter-clock-wise
fn rotate90_ccw_u16(pos: u16) -> u16 {
    v_mirror_u16(mirror_diagonal_u16(pos))
}

/// Mirrors the given bitboard position diagonally
fn mirror_diagonal_u16(pos: u16) -> u16 {
    let row = pos / 8;
    let col = pos & 7;
    (col * 8) + row
}

/// Mirrors the given bitboard position index horizontally
fn h_mirror_u16(pos: u16) -> u16 {
    pos ^ 7
}

/// Mirrors the given bitboard position index vertically
fn v_mirror_u16(pos: u16) -> u16 {
    pos ^ 56
}
