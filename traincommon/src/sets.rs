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

use std::cell::RefCell;
use std::cmp::{min, Ordering};
use std::fs::File;
use std::io::{stdout, BufReader, BufWriter, Error, Write, Read};
use std::path::Path;
use std::sync::{Arc, atomic, mpsc};
use std::sync::atomic::{AtomicUsize};
use std::sync::mpsc::{SyncSender};
use std::thread;
use lz4_flex::frame::FrameEncoder;
use rand::prelude::SliceRandom;
use rand::{thread_rng};
use velvet::bitboard::{BitBoard};
use velvet::colors::{BLACK, Color, WHITE};
use velvet::fen::{create_from_fen};
use velvet::moves::Move;
use velvet::nn::{piece_idx, SCORE_SCALE, king_bucket, INPUTS, BUCKET_SIZE};
use velvet::nn::io::{read_f32, read_i16, read_u16, read_u32, read_u64, read_u8, write_f32, write_u16, write_u64, write_u8};
use velvet::scores::{is_eval_score, is_mate_or_mated_score, MAX_EVAL};
use velvet::syzygy::ProbeTB;
use velvet::syzygy::tb::TBResult;

pub const SAMPLES_PER_SET: usize = 200_000;

pub const K: f64 = 1.603;

pub trait DataSamples {
    fn init(&mut self, idx: usize, result: f32, stm: u8);
    fn add_wpov(&mut self, idx: usize, pos: u16);
    fn add_bpov(&mut self, idx: usize, pos: u16);
    fn finalize(&mut self, idx: usize);
}

pub fn convert_sets(thread_count: usize, caption: &str, in_path: &str, out_path: &str, use_game_result: bool, allow_transformations: bool) -> usize {
    let mut already_converted = false;
    let mut max_set_id = 0;
    let mut min_set_id = usize::MAX;
    let mut max_converted_set_id = 0;

    for id in 1..usize::MAX {
        if Path::new(&format!("{}/{}.lz4", out_path, id)).exists() {
            max_converted_set_id = max_converted_set_id.max(id);
            already_converted = true;
        } else if Path::new(&format!("{}/test_pos_{}.bin", in_path, id)).exists() {
            max_set_id = max_set_id.max(id);
            min_set_id = min(id, min_set_id);
        } else {
            break;
        }
    }

    if !already_converted {
        println!("Converting {} {} sets ...", (max_set_id - min_set_id + 1), caption);
        max_set_id = convert_test_pos(thread_count, in_path.to_string(), out_path.to_string(), min_set_id, max_set_id, use_game_result, allow_transformations)
            .expect("Could not convert test positions!");
    } else {
        println!("Skipping conversion, because target folder is not empty");
        max_set_id = max_converted_set_id;
    }

    max_set_id
}


struct OutputWriter {
    output_dir: String,
    next_set_id: Arc<AtomicUsize>,
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
    thread_local! {
        static ENTRIES: RefCell<Vec<Entry>> = RefCell::new(Vec::with_capacity(200_000));
    }

    fn new(output_dir: String, next_set_id: &Arc<AtomicUsize>, allow_transformations: bool) -> Self {
        OutputWriter{output_dir, next_set_id: next_set_id.clone(), allow_transformations}
    }

    pub fn write(&self, bb_map: u16, white_king_pos: u8, black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
        self.add_entry(bb_map, white_king_pos, black_king_pos, active_player, score, bb);

        if !self.allow_transformations {
            return;
        }

        let any_castling_rights = bb_map & (0b1111 << 12) != 0;
        if any_castling_rights {
            return;
        }

        let white_has_pawns = bb_map & (1 << 0) != 0;
        let black_has_pawns = bb_map & (1 << 5) != 0;
        if white_has_pawns || black_has_pawns {
            return;
        }

        self.add_transformed_entry(rotate90_ccw_u16, bb_map, white_king_pos, black_king_pos, active_player, score, bb);
        self.add_transformed_entry(mirror_diagonal_u16, bb_map, white_king_pos, black_king_pos, active_player, score, bb);
    }

    fn add_entry(&self, bb_map: u16, white_king_pos: u8, black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
        OutputWriter::ENTRIES.with(|cell| {
            let mut entries = cell.borrow_mut();
            entries.push(Entry{bb_map, score, active_player, white_king_pos, black_king_pos, bb});
            if entries.len() == 200000 {
                self.flush_entries(&mut entries);
            }
        });
    }

    fn flush_entries(&self, entries: &mut Vec<Entry>) {
        entries.shuffle(&mut thread_rng());
        let mut writer = next_file(&self.output_dir.clone(), self.next_set_id.clone().fetch_add(1, atomic::Ordering::Relaxed));
        for entry in entries.drain(0..entries.len()) {
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

    fn add_transformed_entry(&self, transform: fn(u16) -> u16, bb_map: u16, mut white_king_pos: u8, mut black_king_pos: u8, active_player: Color, score: f32, bb: [u64; 13]) {
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

    pub fn flush_send_entries(&self, tx: &SyncSender<Option<Entry>>) {
        OutputWriter::ENTRIES.with(|cell| {
            let mut entries = cell.borrow_mut();

            let len = entries.len();
            for entry in entries.drain(0..len) {
                tx.send(Some(entry)).expect("could not send entry");
            }
        });
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
    let next_input_set_id = Arc::new(AtomicUsize::new(min_unconverted_id));

    let (tx, rx) = mpsc::sync_channel::<Option<Entry>>(16*1024*1024);

    let output_writer = Arc::new(OutputWriter::new(out_path, &next_set_id, allow_transformations));

    println!("Starting {} threads", thread_count);

    for _ in 1..=thread_count {
        let in_path2 = in_path.clone();
        let tx2 = tx.clone();
        let output_writer2 = output_writer.clone();
        let next_input_set_id2 = next_input_set_id.clone();

        threads.push(thread::spawn(move || {

            loop {
                let i = next_input_set_id2.fetch_add(1, atomic::Ordering::Relaxed);
                if i > max_training_set_id {
                    break;
                }

                read_from_bin_fen_file(&output_writer2, format!("{}/test_pos_{}.bin", in_path2, i).as_str(), use_game_result);

                print!("{} ", i);
                stdout().flush().unwrap();
            }
            output_writer2.flush_send_entries(&tx2);

            tx2.send(None).expect("could not send stop entry");
        }));
    }

    let mut remaining = thread_count;

    while remaining > 0 {
        match rx.recv().expect("could not receive entry") {
            Some(entry) => {
                output_writer.add_entry(entry.bb_map, entry.white_king_pos, entry.black_king_pos, entry.active_player, entry.score, entry.bb);
            }
            None => {
                remaining -= 1;
            }
        }
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("\nConversion finished");

    Ok(next_set_id.load(atomic::Ordering::Relaxed).saturating_sub(1))
}

fn read_from_bin_fen_file(output_writer: &Arc<OutputWriter>, file_name: &str, _use_game_result: bool) {
    let file = File::open(file_name).expect("Could not open test position file");
    let mut reader = BufReader::new(file);

    let version = match read_u8(&mut reader) {
        Ok(v) => v,
        Err(e) => {
            println!("Could not read {}: {}", file_name, e);
            return;
        }
    };

    // assert_eq!(version, 1, "unexpected position file version");
    // temporary workaround for reading training data in the initial file format, which does not have a version header,
    // but instead directly starts with the FEN length of the first position
    let mut first_byte = if version == 1 {
        None
    } else {
        Some(version)
    };
    let is_old_data = first_byte.is_some();

    loop {
        let fen_len = if let Some(b) = first_byte {
            first_byte = None;
            b as usize
        } else {
            match read_u8(&mut reader) {
                Ok(len) => len as usize,
                Err(_) => {
                    return;
                }
            }
        };

        assert!(fen_len < 256, "fen_len exceeded safety limit");

        let mut fen_bytes = vec![0; fen_len];
        reader.read_exact(&mut fen_bytes).expect("could not read FEN");

        let fen = String::from_utf8(fen_bytes).expect("could not convert FEN");

        let mut board = create_from_fen(&fen);
        let game_result = read_i16(&mut reader).expect("could not read game result");
        let move_count = read_u16(&mut reader).expect("could not read move count") as usize;

        let mut gives_check = false;
        let mut moves = Vec::with_capacity(move_count);

        let mut last_tb_result = None;

        let (lower_limit, upper_limit) = match game_result.cmp(&0) {
            Ordering::Less => (i16::MIN, 300),
            Ordering::Equal => (-500, 500),
            Ordering::Greater => (-300, i16::MAX)
        };

        for _ in 1..=move_count {
            let m = Move::from_u32(read_u32(&mut reader).expect("could not read move"));
            board.perform_move(m.unpack());
            let score = if let Some(tb_result) = board.probe_wdl() {
                let (tb_score, tb_result) = match tb_result {
                    TBResult::Draw => (0, 0),
                    TBResult::Win => (m.score(), board.active_player().score(1)),
                    TBResult::Loss => (m.score(), board.active_player().score(-1)),
                    TBResult::CursedWin => (0, 0),
                    TBResult::BlessedLoss => (0, 0)
                };
                last_tb_result = Some(tb_result);

                tb_score

            } else if let Some(last_tb_result) = last_tb_result {
                if last_tb_result != 0 {
                    m.score()
                } else {
                    0
                }
            } else {
                m.score()
            };
            moves.push((m, score.clamp(lower_limit, upper_limit)));
        }

        let end_full_move_count = board.fullmove_count();

        board = create_from_fen(&fen);

        let skip_all = (!is_old_data && end_full_move_count > 140) || (is_old_data && (game_result == 0 || end_full_move_count > 80));

        let final_eval_score = moves.iter().map(|(_, score)| *score).filter(|&score| is_eval_score(score)).last().unwrap_or(game_result * MAX_EVAL);
        for (i, &(m, raw_score)) in moves.iter().enumerate() {
            if skip_all || is_mate_or_mated_score(raw_score) || !m.is_quiet() || gives_check {
                board.perform_move(m.unpack());
                gives_check = board.is_in_check(board.active_player());
                continue;
            }

            let mut adj_score = if !is_eval_score(raw_score) {
                raw_score
            } else {
                moves.iter().map(|(_, score)| *score).nth(i + 16).unwrap_or(final_eval_score)
            };
            adj_score = adj_score.clamp(raw_score - 100, raw_score + 100);

            let scaled_score = ((raw_score as f32 * 0.8) + (adj_score as f32 * 0.2)) / SCORE_SCALE as f32;

            let active_player = board.active_player();

            let mut black_king_pos = 0;
            let mut white_king_pos = 0;
            let mut bb: [u64; 13] = [0; 13];

            for pos in 0..64 {
                let piece = board.get_item(pos);
                if piece == 0 {
                    continue;
                }

                if piece == -6 {
                    black_king_pos = pos;
                } else if piece == 6 {
                    white_king_pos = pos;
                } else {
                    bb[(piece + 6) as usize] |= 1 << pos;
                }
            }

            let mut bb_map = 0;
            if board.can_castle_king_side(WHITE) {
                bb_map |= 1 << 15;
            }
            if board.can_castle_queen_side(WHITE) {
                bb_map |= 1 << 14;
            }
            if board.can_castle_king_side(BLACK) {
                bb_map |= 1 << 13;
            }
            if board.can_castle_queen_side(BLACK) {
                bb_map |= 1 << 12;
            }

            for i in 1i8..=5i8 {
                if bb[(i + 6) as usize] != 0 {
                    bb_map |= 1 << (i - 1);
                }
                if bb[(-i + 6) as usize] != 0 {
                    bb_map |= 1 << (i + 4);
                }
            }

            board.perform_move(m.unpack());

            gives_check = board.is_in_check(board.active_player());

            if gives_check || board.halfmove_count() < 20 {
                continue;
            }

            output_writer.write(bb_map, white_king_pos as u8, black_king_pos as u8, active_player, scaled_score, bb);
        }
    }
}

pub fn read_samples<T: DataSamples>(samples: &mut T, start: usize, file_name: &str, add_feature_abstractions: bool) {
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

        let white_no_pawns = bb_map & (1 << 0) == 0;
        let black_no_pawns = bb_map & (1 << 5) == 0;

        let h_mirror_white_pov = (white_king & 7) > 3;
        let h_mirror_black_pov = (black_king & 7) > 3;

        let no_pawns = white_no_pawns && black_no_pawns;
        let v_mirror_white_pov = no_pawns && (white_king / 8) > 3;
        let v_mirror_black_pov = no_pawns && (v_mirror_u16(black_king)) / 8 > 3;


        let transform_wpov = if h_mirror_white_pov {
            if v_mirror_white_pov {
                |pos| v_mirror_u16(h_mirror_u16(pos))
            } else {
                h_mirror_u16
            }
        } else if v_mirror_white_pov {
            v_mirror_u16
        } else {
            |pos| pos
        };

        let transform_bpov = if h_mirror_black_pov {
            if v_mirror_black_pov {
                h_mirror_u16
            } else {
                |pos| v_mirror_u16(h_mirror_u16(pos))
            }
        } else if v_mirror_black_pov {
            |pos| pos
        } else {
            v_mirror_u16
        };

        let w_bucket = king_bucket(transform_wpov(white_king));
        let b_bucket = king_bucket(transform_bpov(black_king));

        const BASE_OFFSET: u16 = 0;

        let opp_offset = 64;
        let (white_own_offset, white_opp_offset, black_own_offset, black_opp_offset) = (
            BASE_OFFSET + w_bucket * BUCKET_SIZE as u16,
            BASE_OFFSET + w_bucket * BUCKET_SIZE as u16 + opp_offset,
            BASE_OFFSET + b_bucket * BUCKET_SIZE as u16,
            BASE_OFFSET + b_bucket * BUCKET_SIZE as u16 + opp_offset,
        );

        samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(white_king) + white_own_offset);
        samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(black_king) + white_opp_offset);

        samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(black_king) + black_own_offset);
        samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(white_king) + black_opp_offset);

        if add_feature_abstractions {
            samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(white_king) + INPUTS as u16);
            samples.add_wpov(idx, 5 * 64 * 2 + transform_wpov(black_king) + opp_offset + INPUTS as u16);

            samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(black_king) + INPUTS as u16);
            samples.add_bpov(idx, 5 * 64 * 2 + transform_bpov(white_king) + opp_offset + INPUTS as u16);
        }

        let mut piece_counts = [0; 10];
        for i in 1i8..=5i8 {
            if bb_map & (1 << (i - 1)) != 0 {
                // White pieces
                let bb = read_u64(&mut reader).unwrap();
                for pos in BitBoard(bb) {
                    piece_counts[i as usize - 1] += 1;
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(pos as u16)
                        + white_own_offset);
                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(pos as u16)
                        + black_opp_offset);

                    if add_feature_abstractions {
                        samples.add_wpov(idx, piece_idx(i) * 64 * 2
                            + transform_wpov(pos as u16) + INPUTS as u16);
                        samples.add_bpov(idx, piece_idx(i) * 64 * 2
                            + transform_bpov(pos as u16)
                            + opp_offset + INPUTS as u16);
                    }
                }
            }

            if bb_map & (1 << (i as usize + 4)) != 0 {
                // Black pieces
                let bb = read_u64(&mut reader).unwrap();
                for pos in BitBoard(bb) {
                    piece_counts[i as usize + 4] += 1;

                    samples.add_bpov(idx, piece_idx(i) * 64 * 2
                        + transform_bpov(pos as u16)
                        + black_own_offset);
                    samples.add_wpov(idx, piece_idx(i) * 64 * 2
                        + transform_wpov(pos as u16)
                        + white_opp_offset);

                    if add_feature_abstractions {
                        samples.add_bpov(idx, piece_idx(i) * 64 * 2
                            + transform_bpov(pos as u16) + INPUTS as u16);
                        samples.add_wpov(idx, piece_idx(i) * 64 * 2
                            + transform_wpov(pos as u16)
                            + opp_offset + INPUTS as u16);
                    }
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
