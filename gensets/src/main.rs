/*
 * Velvet Chess Engine
 * Copyright (C) 2021 mhonert (https://github.com/mhonert)
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

use velvet::engine::{Engine, Message, LogLevel};
use velvet::magics::initialize_magics;
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use velvet::random::Random;
use std::io::{BufReader, BufRead, BufWriter, Write};
use std::fs::File;
use std::sync::mpsc;
use std::str::FromStr;
use std::process::exit;
use std::{thread};
use velvet::colors::{WHITE, Color};
use velvet::moves::NO_MOVE;
use std::sync::mpsc::Sender;
use clap::App;
use velvet::search::Search;
use velvet::gen_quiet_pos::GenQuietPos;
use velvet::fen::write_fen;
use std::cmp::{min};
use shakmaty_syzygy::{Tablebase, SyzygyError, Wdl, Dtz};
use shakmaty::{Chess, CastlingMode, Position, Setup};
use shakmaty::fen::Fen;
use std::convert::TryInto;
use std::collections::HashSet;
use std::path::Path;

#[derive(Clone, Debug)]
struct TestPos {
    fen: String,
    count: u16,
    end_count: u16,
    wp: f64,
    orig_wp: f64,
    white_score: i32,
    tb_result: bool,
    zobrist_hash: u64,
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage(
            "-i, --start-index=<START>              'Sets the start index for the generated training sets'
             -o, --opening-book=<FILE>              'Sets the opening book file'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -t  --table-base-path=<FILE>           'Sets the Syzygy tablebase path'")
        .get_matches();

    let start_index = i32::from_str(matches.value_of("start-index").unwrap()).expect("Start index must be an integer");

    let book_file = matches.value_of("opening-book").unwrap();

    let tb_path = matches.value_of("table-base-path").unwrap();

    let concurrency = match matches.value_of("concurrency") {
        Some(v) => usize::from_str(v).expect("Concurrency must be an integer >= 1"),
        None => {
            eprintln!("Missing -c (concurrency) option");
            exit(1);
        }
    };

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    initialize_magics();

    println!("Reading opening book: {} ...", book_file);
    let openings = read_openings(book_file);


    let (tx, rx) = mpsc::channel::<TestPos>();

    println!("Starting worker threads ...");
    let mut start = Instant::now();
    spawn_threads(&tx, concurrency, &openings, String::from(tb_path));

    let mut count = 0;
    println!("Waiting for generated test positions ...");

    let mut chunks = start_index;

    let file_name = format!("./data/test_pos_{}.fen", chunks);
    if Path::new(&file_name).exists() {
        panic!("Output file already exists: {}", file_name);
    }
    let mut file = File::create(&file_name).expect("Could not create output file");
    let mut writer = BufWriter::new(file);

    let mut sub_count: u64 = 0;

    for pos in rx {
        writeln!(&mut writer, "{} {} {} {}", pos.fen, pos.count, pos.end_count, pos.white_score).expect("Could not write position to file");
        count += 1;
        sub_count += 1;

        if count % 200_000 == 0 {
            chunks += 1;
            file = File::create(format!("./data/test_pos_{}.fen", chunks)).expect("Could not create output file");
            writer = BufWriter::new(file);
        }

        if sub_count >= 10_000 {
            let duration_secs = Instant::now().duration_since(start).as_millis() as f64 / 1000.0;
            if duration_secs > 0.0 {
                let per_minute = (sub_count as f64 / duration_secs) * 60.0;

                println!("- generated {} test positions ({:.2} per minute)", count, per_minute);
            }
            start = Instant::now();
            sub_count  = 0;
        }

    }

    println!("End");
}

fn spawn_threads(tx: &Sender<TestPos>, concurrency: usize, openings: &Vec<String>, tb_path: String) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.clone();
        let tb_path2 = tb_path.clone();
        thread::spawn(move || {
            find_test_positions(&tx2, &openings2, tb_path2);
        });
    }

}

fn find_test_positions(tx: &Sender<TestPos>, openings: &[String], tb_path: String) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let (_, erx1) = mpsc::channel::<Message>();
    let mut engine1 = Engine::new(erx1);
    engine1.set_log_level(LogLevel::Error);

    let (_, erx2) = mpsc::channel::<Message>();
    let mut engine2 = Engine::new(erx2);
    engine2.set_log_level(LogLevel::Error);

    let (_, erx3) = mpsc::channel::<Message>();
    let mut engine3 = Engine::new(erx3);
    engine3.set_log_level(LogLevel::Error);

    let mut tb = Tablebase::new();
    println!("Setting tablebase path to: {}", tb_path);
    tb.add_directory(tb_path).expect("Could not add tablebase path");

    let mut duplicate_check = HashSet::with_capacity(65536);

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        let mut positions = collect_quiet_pos(&tb, &mut rnd, opening.as_str(), &mut [&mut engine1, &mut engine2], &mut engine3);
        for pos in positions.iter_mut() {
            if duplicate_check.contains(&pos.zobrist_hash) {
                continue;
            }

            duplicate_check.insert(pos.zobrist_hash);
            tx.send(pos.clone()).expect("could not send test position");
        }

        duplicate_check.clear();
    }
}

fn read_openings(file_name: &str) -> Vec<String> {
    let file = File::open(file_name).expect("Could not open book file");
    let mut reader = BufReader::new(file);

    let mut openings = Vec::new();

    loop {
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(read) => if read == 0 {
                return openings;
            },

            Err(e) => panic!("could not read line from opening book: {}", e)
        };

        openings.push(line);
    }
}

fn collect_quiet_pos(tb: &Tablebase<Chess>, rnd: &mut Random, opening: &str, engines: &mut [&mut Engine], pos_engine: &mut Engine) -> Vec<TestPos> {
    let mut player_color = WHITE;

    let node_limit = 150 + (rnd.rand32() % 150) as u64;
    for engine in engines.iter_mut() {
        engine.reset();
        engine.set_position(opening.to_string(), Vec::new());
        engine.node_limit = node_limit;
    }

    let mut positions = Vec::new();
    let mut ply = 0;

    loop {
        ply += 1;
        let idx = idx_by_color(player_color);

        let best_move = engines[idx].find_best_move(3, 100, true);

        if best_move == NO_MOVE {
            return positions;
        }

        let score = best_move.score() * player_color as i32;

        if (ply >= 300 && score.abs() < 1000) || engines[idx].board.is_draw() {
            return positions;
        }

        engines[idx].board.perform_move(best_move);
        player_color = -player_color;

        engines[idx_by_color(player_color)].board.perform_move(best_move);

        if ply > 6 && rnd.rand32() % 100 <= 10 {
            if score.abs() >= 7000 {
                continue;
            }

            let initial_fen = write_fen(&engines[0].board);
            pos_engine.reset();
            pos_engine.set_position(initial_fen.to_string(), Vec::new());

            if !pos_engine.make_quiet_position() {
                continue;
            }

            let new_fen = write_fen(&pos_engine.board);

            pos_engine.node_limit = 100_000;
            let best_move = pos_engine.find_best_move(10, 0, true);
            if best_move == NO_MOVE {
                continue;
            }

            let mut score = best_move.score() * pos_engine.board.active_player() as i32;
            if score.abs() >= 7000 {
                continue;
            }

            if pos_engine.board.get_occupancy_bitboard().count_ones() <= 5 {
                let tb_pos: Chess = new_fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();
                match tb.probe_wdl(&tb_pos) {
                    Ok(_) => {
                        if tb.probe_dtz(&tb_pos).is_ok() {
                            let (mut result, move_count) = resolve_tb_match(&tb, &tb_pos);
                            result *= pos_engine.board.active_player();

                            let half_move_distance = move_count as i32 - pos_engine.board.halfmove_count as i32;

                            if result == 0 { // Draw
                                if half_move_distance == 0 {
                                    score = 0;
                                } else if half_move_distance < 100 {
                                    score /= 100 - half_move_distance
                                }
                            } else if result < 0 {
                                score = -3000 + half_move_distance;
                            } else if result > 0 {
                                score = 3000 - half_move_distance;
                            }
                        }
                    },

                    Err(e) => {
                        match e {
                            SyzygyError::Castling => {}
                            SyzygyError::TooManyPieces => {}
                            SyzygyError::MissingTable { .. } => {}
                            SyzygyError::ProbeFailed { error, .. } => {
                                println!("TB probe failed: {}", error);
                            }
                        }
                    }
                }
            }

            positions.push(TestPos{
                fen: new_fen,
                count: pos_engine.board.halfmove_count,
                end_count: 0,
                orig_wp: 0.0,
                wp: 0.0,
                white_score: score,
                tb_result: false,
                zobrist_hash: pos_engine.board.get_hash()
            });
        }
    }
}

fn resolve_tb_match(tb: &Tablebase<Chess>, in_pos: &Chess) -> (i8, u32) {
    let mut pos = in_pos.clone();
    let dtz = tb.probe_dtz(&pos).unwrap();
    let wdl = real_wdl(&tb, &pos, dtz).unwrap();

    let result = match wdl {
        Wdl::Loss => -1,
        Wdl::Win => 1,
        _ => 0,
    };

    loop {
        if pos.is_game_over() {
            break;
        }

        let (m, dtz) = tb.best_move(&pos).unwrap().expect("no move found!");
        if dtz == Dtz(0) {
            break;
        }

        pos = pos.play(&m).expect("invalid move");
    }

    let mut ply: u32 = (TryInto::<u32>::try_into(pos.fullmoves()).unwrap() - 1) * 2;
    if pos.turn() == shakmaty::Color::Black {
        ply += 1;
    }

    (result, ply)
}

fn real_wdl(tb: &Tablebase<Chess>, pos: &Chess, dtz: Dtz) -> Result<Wdl, SyzygyError> {
    if let Some(outcome) = pos.outcome() {
        return Ok(Wdl::from_outcome(outcome, pos.turn()));
    }

    let halfmoves = min(101, pos.halfmoves()) as i32;
    let before_zeroing = dtz.add_plies(halfmoves);

    if before_zeroing.0.abs() != 100 || halfmoves == 0 {
        return Ok(Wdl::from_dtz_after_zeroing(before_zeroing));
    }

    if halfmoves == 1 && dtz.0.abs() == 99 {
        return Ok(Wdl::from_dtz_after_zeroing(before_zeroing));
    }

    let best = tb.best_move(pos)?.expect("has moves");
    let mut after = pos.clone();
    after.play_unchecked(&best.0);
    Ok(-real_wdl(tb, &after, best.1)?)
}

fn idx_by_color(color: Color) -> usize {
    if color == WHITE {
        0
    } else {
        1
    }
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}