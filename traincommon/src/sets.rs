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

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use itertools::{Itertools};
use std::cmp::{min};
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Error, Write};
use std::path::Path;
use std::str::FromStr;
use std::thread;
use lz4_flex::frame::FrameEncoder;
use velvet::bitboard::{BitBoard};
use velvet::fen::{parse_fen, FenParseResult};
use velvet::nn::{piece_idx, KING_BUCKETS, SCORE_SCALE, board_4};

pub const SAMPLES_PER_SET: usize = 200_000;

pub const K: f64 = 1.603;

pub trait DataSamples {
    fn init(&mut self, idx: usize, result: f32, stm: u8);
    fn add_wpov(&mut self, idx: usize, pos: u16);
    fn add_bpov(&mut self, idx: usize, pos: u16);
    fn finalize(&mut self, idx: usize);
}

pub fn convert_sets(thread_count: usize, caption: &str, in_path: &str, out_path: &str, min_id: usize, use_game_result: bool) -> usize {
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
        convert_test_pos(thread_count, in_path.to_string(), out_path.to_string(), min_unconverted_id, max_set_id, use_game_result)
            .expect("Could not convert test positions!");
    }

    max_set_id
}

fn convert_test_pos(
    thread_count: usize, in_path: String, out_path: String, min_unconverted_id: usize, max_training_set_id: usize,
    use_game_result: bool,
) -> Result<(), Error> {
    let mut threads = Vec::new();
    for c in 1..=thread_count {
        let in_path2 = in_path.clone();
        let out_path2 = out_path.clone();
        threads.push(thread::spawn(move || {
            for i in ((c + min_unconverted_id - 1)..=max_training_set_id).step_by(thread_count) {
                print!("{} ", i);
                stdout().flush().unwrap();

                let file = File::create(format!("{}/{}.lz4", out_path2, i)).expect("Could not create tensor data file");
                let encoder = FrameEncoder::new(file);
                let mut writer = BufWriter::with_capacity(1024 * 1024, encoder);

                read_from_fen_file(format!("{}/test_pos_{}.fen", in_path2, i).as_str(), &mut writer, use_game_result);
                writer.write_u16::<LittleEndian>(u16::MAX).unwrap();

                writer.flush().unwrap();
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("\nConversion finished");

    Ok(())
}

fn read_from_fen_file(file_name: &str, writer: &mut BufWriter<FrameEncoder<File>>, _use_game_result: bool) {
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

        // 0..5 | 6
        // fen  | score
        let parts = line.trim_end().split(' ').collect_vec();

        let score = if parts.len() == 7 {
            let score = i32::from_str(parts[5]).expect("Could not parse score");
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

        writer.write_u16::<LittleEndian>(bb_map).unwrap();
        writer.write_f32::<LittleEndian>(if active_player.is_white() { score } else { -score }).unwrap();
        writer.write_u8(active_player.0).unwrap();

        let kings = white_king_pos as u16 | ((black_king_pos as u16) << 8);
        writer.write_u16::<LittleEndian>(kings).unwrap();

        for i in 1i8..=5i8 {
            let bb_white = bb[(i + 6) as usize];
            if bb_white != 0 {
                writer.write_u64::<LittleEndian>(bb_white).unwrap();
            }

            let bb_black = bb[(-i + 6) as usize];
            if bb_black != 0 {
                writer.write_u64::<LittleEndian>(bb_black).unwrap();
            }
        }
    }
}

pub fn read_samples<T: DataSamples>(samples: &mut T, start: usize, file_name: &str, gen_mirror_pos: bool, rnd: &[u8]) {
    let file = File::open(file_name).unwrap_or_else(|_| panic!("Could not open test position file: {}", file_name));
    let decoder = lz4_flex::frame::FrameDecoder::new(file);
    let mut reader = BufReader::new(decoder);

    let mut total_samples = 0;

    let mut idx = start;

    loop {
        let mut bb_map = reader.read_u16::<LittleEndian>().unwrap();
        if bb_map == u16::MAX {
            break;
        }
        total_samples += 1;

        let score = reader.read_f32::<LittleEndian>().unwrap();
        let stm = reader.read_u8().unwrap();
        samples.init(idx, score, stm);

        let mut any_castling = false;
        if bb_map & (1 << 15) != 0 {
            bb_map ^= 1 << 15;
            any_castling = true;
        }

        let kings = reader.read_u16::<LittleEndian>().unwrap();
        let mut white_king = kings & 0b111111;
        let mut black_king = kings >> 8;

        let white_no_queens = bb_map & (1 << 4) == 0;
        let black_no_queens = bb_map & (1 << 9) == 0;
        let white_no_rooks = bb_map & (1 << 3) == 0;
        let black_no_rooks = bb_map & (1 << 8) == 0;
        // let white_no_bishops = bb_map & (1 << 2) == 0;
        // let black_no_bishops = bb_map & (1 << 7) == 0;
        let white_no_pawns = bb_map & (1 << 0) == 0;
        let black_no_pawns = bb_map & (1 << 5) == 0;

        let r= if rnd.is_empty() { 0 } else { rnd[idx] };
        let transform = if !any_castling && gen_mirror_pos && white_no_pawns && black_no_pawns {
            match r {
                0 => |pos| pos,
                1 => rotate90_ccw_u16,
                2 => h_mirror_u16,
                _ => mirror_diagonal_u16,
            }
        } else {
            |pos| pos
        };

        white_king = transform(white_king);
        black_king = transform(black_king);

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

        let w_kingrel_bucket = board_4(transform_wpov(white_king));
        let b_kingrel_bucket = board_4(transform_bpov(black_king));

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
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(transform(pos as u16))
                        + white_offset);
                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(transform(pos as u16))
                        + black_offset + opp_offset);
                }
            }

            if bb_map & (1 << (i as usize + 4)) != 0 {
                // Black pieces
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(transform(pos as u16))
                        + black_offset);
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(transform(pos as u16))
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
