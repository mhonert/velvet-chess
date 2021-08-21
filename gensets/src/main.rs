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

use std::thread;
use std::cmp::min;
use std::collections::HashSet;
use std::convert::TryInto;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::Sender;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::App;
use shakmaty::{CastlingMode, Chess, Position, Setup};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Dtz, SyzygyError, Tablebase, Wdl};

use gen_quiet_pos::GenQuietPos;
use velvet::colors::WHITE;
use velvet::engine::LogLevel;
use velvet::fen::{create_from_fen, write_fen};
use velvet::magics::initialize_magics;
use velvet::moves::{Move, NO_MOVE};
use velvet::random::Random;
use velvet::search::{Search, SearchLimits};
use velvet::transposition_table::TranspositionTable;

pub mod gen_quiet_pos;

#[derive(Clone, Debug)]
struct TestPos {
    fen: String,
    count: u16,
    score: i32,
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
        writeln!(&mut writer, "{} {} {}", pos.fen, pos.count, pos.score).expect("Could not write position to file");
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

fn spawn_threads(tx: &Sender<TestPos>, concurrency: usize, openings: &[String], tb_path: String) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.to_owned();
        let tb_path2 = tb_path.clone();
        thread::spawn(move || {
            find_test_positions(&tx2, &openings2, tb_path2);
        });
    }
}

fn find_test_positions(tx: &Sender<TestPos>, openings: &[String], tb_path: String) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let mut tb = Tablebase::new();
    println!("Setting tablebase path to: {}", tb_path);
    tb.add_directory(tb_path).expect("Could not add tablebase path");

    let mut duplicate_check = HashSet::with_capacity(65536);

    let tt = Arc::new(TranspositionTable::new(256));

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        let mut positions = collect_quiet_pos(&tb, &mut duplicate_check, &mut rnd, opening.as_str(), tt.clone());
        for pos in positions.iter_mut() {
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

fn select_move(rnd: &mut Random, search: &mut Search, min_depth: i32) -> Move {
    let mut move_candidates = Vec::with_capacity(8);
    let mut min_score = i32::MIN;
    loop {
        let m = search.find_best_move(None, min_depth, &move_candidates);
        if m == NO_MOVE {
            break;
        }

        if !move_candidates.is_empty() && m.score() < min_score {
            break;
        }

        if move_candidates.is_empty() {
            min_score = m.score() - 25;
        }

        move_candidates.push(m.without_score());

        if move_candidates.len() >= 6 {
            break;
        }
    }

    if move_candidates.is_empty() {
        return NO_MOVE;
    }

    move_candidates[rnd.rand32() as usize % move_candidates.len()]
}

fn collect_quiet_pos(tb: &Tablebase<Chess>, duplicate_check: &mut HashSet<u64>, rnd: &mut Random, opening: &str, tt: Arc<TranspositionTable>) -> Vec<TestPos> {
    let mut player_color = WHITE;

    tt.clear();
    let board = create_from_fen(opening);

    let node_limit = 200 + (rnd.rand32() % 100) as u64;
    let limits = SearchLimits::new(None, Some(node_limit), None, None, None, None, Some(1000), None).unwrap();
    let mut search = Search::new(Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, board, 1);


    let mut positions = Vec::new();
    let mut ply = 0;

    loop {
        ply += 1;

        let best_move = select_move(rnd, &mut search, 3);

        if best_move == NO_MOVE {
            return positions;
        }

        let score = best_move.score() * player_color as i32;

        if (ply >= 300 && score.abs() < 500) || search.board.is_draw() {
            return positions;
        }

        search.board.perform_move(best_move);
        player_color = -player_color;

        if ply > 6 && rnd.rand32() % 100 <= 10 {
            if score.abs() >= 7000 {
                continue;
            }

            let prev_board = search.board.clone();

            if !search.make_quiet_position() {
                search.board = prev_board;
                continue;
            }

            let pos_hash = search.board.get_hash();

            if duplicate_check.contains(&pos_hash) {
                search.board = prev_board;
                continue;
            }

            let new_fen = write_fen(&search.board);

            search.set_node_limit(50000);
            let best_move = search.find_best_move(None, 9, &[]);
            search.set_node_limit(node_limit);
            if best_move == NO_MOVE {
                search.board = prev_board;
                continue;
            }

            let mut score = best_move.score() * search.board.active_player() as i32;
            if score.abs() >= 7000 {
                search.board = prev_board;
                continue;
            }

            let mut tb_result = false;
            if search.board.get_occupancy_bitboard().count_ones() <= 5 {
                let tb_pos: Chess = new_fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();
                match tb.probe_wdl(&tb_pos) {
                    Ok(_) => {
                        if tb.probe_dtz(&tb_pos).is_ok() {
                            let (mut result, move_count) = resolve_tb_match(tb, &tb_pos);
                            result *= search.board.active_player();

                            let half_move_distance = move_count as i32 - search.board.halfmove_count() as i32;

                            if result == 0 { // Draw
                                if half_move_distance == 0 {
                                    score = 0;
                                } else if half_move_distance < 100 {
                                    score /= 100 - half_move_distance
                                }
                            } else if result < 0 {
                                score = -4000 + half_move_distance;
                            } else if result > 0 {
                                score = 4000 - half_move_distance;
                            }
                            tb_result = true;
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

            let pos_count = search.board.halfmove_count();
            if !tb_result {
                let (lt_score, _) = playout_match(tb, score, 8, &mut search);
                search.set_node_limit(node_limit);
                score = lt_score;
            }

            positions.push(TestPos{
                fen: new_fen,
                count: pos_count,
                score,
                zobrist_hash: pos_hash
            });

            duplicate_check.insert(pos_hash);
            search.board = prev_board;
        }
    }
}

fn playout_match(tb: &Tablebase<Chess>, start_score: i32, start_depth: i32, search: &mut Search) -> (i32, u16) {
    let mut lt_score = start_score as f64;
    let mut lt_score_distance = start_depth as u16;
    let mut min_depth = start_depth;

    for i in 0..30 {
        let best_move = search.find_best_move(None, min_depth, &[]);
        if best_move == NO_MOVE || search.board.is_draw() {
            break;
        }

        search.board.perform_move(best_move);

        if search.board.get_occupancy_bitboard().count_ones() <= 5 {
            let new_fen = write_fen(&search.board);
            let tb_pos: Chess = new_fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();
            match tb.probe_wdl(&tb_pos) {
                Ok(_) => {
                    if tb.probe_dtz(&tb_pos).is_ok() {
                        let (mut result, move_count) = resolve_tb_match(tb, &tb_pos);
                        result *= search.board.active_player();

                        let half_move_distance = move_count as i32 - search.board.halfmove_count() as i32;

                        lt_score_distance += half_move_distance as u16;
                        lt_score *= 1.0 - 1.0 / lt_score_distance as f64;
                        lt_score += result as f64 * 4000.0 * (1.0 / lt_score_distance as f64);
                        break;
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

        let mut score = best_move.score() * -search.board.active_player() as i32;
        if score.abs() >= 7000 {
            if score.is_negative() {
                score += 4000;
            } else {
                score -= 4000;
            }
            lt_score *= 1.0 - 1.0 / lt_score_distance as f64;
            lt_score += score as f64 * (1.0 / lt_score_distance as f64);
            break
        }
        lt_score *= 1.0 - 1.0 / lt_score_distance as f64;
        lt_score += score as f64 * (1.0 / lt_score_distance as f64);

        if min_depth > 4 && i >= 3 && i % 2 == 1 {
            min_depth -= 1;
        }

        lt_score_distance += 1
    }

    (lt_score as i32, lt_score_distance - start_depth as u16)
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

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
