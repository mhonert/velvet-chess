/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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
use itertools::Itertools;
use lz4_flex::frame::FrameEncoder;
use std::cmp::min;
use std::fs::File;
use std::io::{stdout, BufRead, BufReader, BufWriter, Error, Write};
use std::path::Path;
use std::str::FromStr;
use std::thread;
use velvet::bitboard::{v_mirror, v_mirror_u16, BitBoard};
use velvet::colors::Color;
use velvet::fen::{parse_fen, FenParseResult};
use velvet::nn::{bucket_size, piece_idx, INPUTS, KING_BUCKETS};
use velvet::pieces::{B, P, Q, R};

pub const SAMPLES_PER_SET: usize = 200_000;

#[derive(Clone, Debug)]
pub struct DataSample {
    pub wpov_inputs: Vec<u16>,
    pub bpov_inputs: Vec<u16>,
    pub result: f32,
    pub wtm: bool,
}

impl Default for DataSample {
    fn default() -> Self {
        DataSample { wpov_inputs: Vec::with_capacity(32), bpov_inputs: Vec::with_capacity(32), result: 0.0, wtm: false }
    }
}

pub fn convert_sets(thread_count: usize, caption: &str, in_path: &str, out_path: &str, min_id: usize) -> usize {
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
        convert_test_pos(thread_count, in_path.to_string(), out_path.to_string(), min_unconverted_id, max_set_id)
            .expect("Could not convert test positions!");
    }

    max_set_id
}

fn convert_test_pos(
    thread_count: usize, in_path: String, out_path: String, min_unconverted_id: usize, max_training_set_id: usize,
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
                let encoder = lz4_flex::frame::FrameEncoder::new(file);
                let mut writer = BufWriter::with_capacity(1024 * 1024, encoder);

                read_from_fen_file(format!("{}/test_pos_{}.fen", in_path2, i).as_str(), &mut writer);
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

fn read_from_fen_file(file_name: &str, writer: &mut BufWriter<FrameEncoder<File>>) {
    let file = File::open(file_name).expect("Could not open test position file");
    let mut reader = BufReader::new(file);

    loop {
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(read) => {
                if read == 0 {
                    return;
                }
            }

            Err(e) => panic!("Reading test position file failed: {}", e),
        };

        let parts = line.trim_end().split(' ').collect_vec();

        let result = if parts.len() == 7 {
            const SCORE_IDX: usize = 6;
            let score = i32::from_str(parts[SCORE_IDX]).expect("Could not parse score");
            score as f32 / 2048.0
        } else {
            panic!("Invalid test position entry: {}", line);
        };

        let fen: String = (parts[0..=5].join(" ") as String).replace("~", "");

        let (pieces, active_player) = match parse_fen(fen.as_str()) {
            Ok(FenParseResult { pieces, active_player, .. }) => (pieces, active_player),
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
        for i in 1i8..=5i8 {
            if bb[(i + 6) as usize] != 0 {
                bb_map |= 1 << (i - 1);
            }
            if bb[(-i + 6) as usize] != 0 {
                bb_map |= 1 << (i + 4);
            }
        }

        writer.write_u16::<LittleEndian>(bb_map).unwrap();

        writer.write_f32::<LittleEndian>(if active_player.is_white() { result } else { -result }).unwrap();
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

pub fn read_samples(samples: &mut [DataSample], start: usize, file_name: &str, flip_pawnless: bool, rnd: &[usize]) {
    let file = File::open(file_name).unwrap_or_else(|_| panic!("Could not open test position file: {}", file_name));
    let decoder = lz4_flex::frame::FrameDecoder::new(file);
    let mut reader = BufReader::new(decoder);

    let mut total_samples = 0;

    let mut idx = start;

    loop {
        let bb_map = reader.read_u16::<LittleEndian>().unwrap();
        if bb_map == u16::MAX {
            break;
        }
        total_samples += 1;

        samples[idx].result = reader.read_f32::<LittleEndian>().unwrap();
        samples[idx].wtm = Color(reader.read_u8().unwrap()).is_white();
        samples[idx].wpov_inputs.clear();
        samples[idx].bpov_inputs.clear();

        let kings = reader.read_u16::<LittleEndian>().unwrap();
        let white_king = kings & 0b111111;
        let black_king = kings >> 8;

        let white_no_queens = bb_map & (1 << 4) == 0;
        let black_no_queens = bb_map & (1 << 9) == 0;
        let white_no_rooks = bb_map & (1 << 3) == 0;
        let black_no_rooks = bb_map & (1 << 8) == 0;
        let white_no_bishops = bb_map & (1 << 2) == 0;
        let black_no_bishops = bb_map & (1 << 7) == 0;
        let white_no_knights = bb_map & (1 << 1) == 0;
        let black_no_knights = bb_map & (1 << 6) == 0;
        let white_no_pawns = bb_map & (1 << 0) == 0;
        let black_no_pawns = bb_map & (1 << 5) == 0;

        let white_king_col = white_king & 7;
        let black_king_col = black_king & 7;
        let mirror_white_pov = white_king_col > 3;
        let mirror_black_pov = black_king_col > 3;

        let (king_bucket, bucket_offset, max_piece_id) = if white_no_queens && black_no_queens {
            if white_no_rooks && black_no_rooks {
                if white_no_bishops && black_no_bishops && white_no_knights && black_no_knights {
                    (3, (bucket_size(Q) + bucket_size(R) + bucket_size(B)) * KING_BUCKETS, P)
                } else {
                    (2, (bucket_size(Q) + bucket_size(R)) * KING_BUCKETS, B)
                }
            } else {
                (1, bucket_size(Q) * KING_BUCKETS, R)
            }
        } else {
            (0, 0, Q)
        };
        let king_offset = king_bucket * 64;
        let bucket_size = bucket_size(max_piece_id);

        let (white_offset, black_offset) = (
            64 * 4
                + bucket_offset as u16
                + board_eighth(h_mirror_if(mirror_white_pov, white_king)) * bucket_size as u16,
            64 * 4
                + bucket_offset as u16
                + board_eighth(v_mirror(h_mirror_if(mirror_black_pov, black_king) as usize) as u16)
                    * bucket_size as u16,
        );

        let opp_offset = (INPUTS / 2) as u16;

        let vmirror = white_no_pawns && black_no_pawns && flip_pawnless && rnd[idx] & 1 == 1;

        samples[idx].wpov_inputs.push(v_mirror_if(vmirror, h_mirror_if(mirror_white_pov, white_king)) + king_offset);
        samples[idx]
            .wpov_inputs
            .push(v_mirror_if(vmirror, h_mirror_if(mirror_white_pov, black_king)) + king_offset + opp_offset);

        samples[idx]
            .bpov_inputs
            .push(v_mirror_if(vmirror, v_mirror_u16(h_mirror_if(mirror_black_pov, black_king))) + king_offset);
        samples[idx].bpov_inputs.push(
            v_mirror_u16(v_mirror_if(vmirror, h_mirror_if(mirror_black_pov, white_king))) + king_offset + opp_offset,
        );

        for i in 1i8..=5i8 {
            if bb_map & (1 << (i - 1)) != 0 {
                // White pieces
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    samples[idx].wpov_inputs.push(
                        piece_idx(i) * 64
                            + v_mirror_if(vmirror, h_mirror_if(mirror_white_pov, pos as u16))
                            + white_offset,
                    );
                    samples[idx].bpov_inputs.push(
                        piece_idx(i) * 64
                            + v_mirror_if(vmirror, v_mirror_u16(h_mirror_if(mirror_black_pov, pos as u16)))
                            + black_offset
                            + opp_offset,
                    );
                }
            }

            if bb_map & (1 << (i as usize + 4)) != 0 {
                // Black pieces
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    samples[idx].bpov_inputs.push(
                        piece_idx(i) * 64
                            + v_mirror_if(vmirror, v_mirror_u16(h_mirror_if(mirror_black_pov, pos as u16)))
                            + black_offset,
                    );
                    samples[idx].wpov_inputs.push(
                        piece_idx(i) * 64
                            + v_mirror_if(vmirror, h_mirror_if(mirror_white_pov, pos as u16))
                            + white_offset
                            + opp_offset,
                    );
                }
            }
        }
        idx += 1;
    }

    if total_samples != SAMPLES_PER_SET {
        panic!("File does not contain the expected 200_000 samples, but: {}", total_samples);
    }
}

fn h_mirror_if(mirror: bool, pos: u16) -> u16 {
    if !mirror {
        pos
    } else {
        h_mirror(pos)
    }
}

fn v_mirror_if(mirror: bool, pos: u16) -> u16 {
    if !mirror {
        pos
    } else {
        v_mirror_u16(pos)
    }
}

/// Mirrors the given bitboard position index horizontally
pub fn h_mirror(pos: u16) -> u16 {
    pos ^ 7
}

fn board_eighth(pos: u16) -> u16 {
    let row = pos / 8;
    let col = pos & 3;
    (row / 2) * 2 + (col / 2)
}
