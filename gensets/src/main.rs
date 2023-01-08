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

use itertools::Itertools;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::App;
use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Outcome, Position};
use shakmaty_syzygy::Tablebase;

use crate::chess960::CHESS960_FENS;
use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, read_fen, write_fen, START_POS};
use velvet::history_heuristics::HistoryHeuristics;
use velvet::move_gen::MoveGenerator;
use velvet::moves::{Move, NO_MOVE};
use velvet::nn::init_nn_params;
use velvet::random::Random;
use velvet::scores::{is_mate_score, MATE_SCORE, MAX_SCORE, MIN_SCORE};
use velvet::search::{PrincipalVariation, Search};
use velvet::time_management::SearchLimits;
use velvet::transposition_table::{TranspositionTable, MAX_DEPTH};

mod chess960;

#[derive(Clone, Debug)]
struct TestPos {
    fen: String,
    score: i32,
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage(
            "-i, --start-index=<START>              'Sets the start index for the generated training sets'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -f, --frc=<BOOL>                       'Generates training sets for Chess960'
             -t  --table-base-path=<FILE>           'Sets the Syzygy tablebase path'",
        )
        .get_matches();

    let start_index = i32::from_str(matches.value_of("start-index").unwrap()).expect("Start index must be an integer");

    let tb_path = matches.value_of("table-base-path").unwrap();

    let concurrency = match matches.value_of("concurrency") {
        Some(v) => usize::from_str(v).expect("Concurrency must be an integer >= 1"),
        None => {
            eprintln!("Missing -c (concurrency) option");
            exit(1);
        }
    };

    let chess960 = match matches.value_of("frc") {
        Some(v) => bool::from_str(v).expect("frc must be a boolean"),
        None => false,
    };

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let openings = gen_openings(chess960);

    let (tx, rx) = mpsc::channel::<TestPos>();

    init_nn_params();
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
        writeln!(&mut writer, "{} {}", pos.fen, pos.score,).expect("Could not write position to file");
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
            sub_count = 0;
        }
    }

    println!("End");
}

fn gen_openings(chess960: bool) -> Vec<String> {
    let mut openings = Vec::with_capacity(500000);
    let hh = HistoryHeuristics::new();
    let mut move_gen = MoveGenerator::new();

    if chess960 {
        for w_fen in CHESS960_FENS.iter() {
            for b_fen in CHESS960_FENS.iter() {
                let fen = mix(w_fen, b_fen);
                openings.push(fen);
            }
        }
    } else {
        let chess_start_pos = [START_POS];
        for fen in chess_start_pos.iter() {
            let mut board = create_from_fen(fen);
            play_opening(4, &hh, &mut move_gen, &mut board, &mut openings);
        }
    }

    openings.sort_unstable();
    openings.dedup();

    println!("Generated {} openings", openings.len());

    openings
}

fn mix(white: &str, black: &str) -> String {
    // "bbqnnrkr/pppppppp/8/8/8/8/PPPPPPPP/BBQNNRKR w HFhf - 0 1",
    let w_pieces = white.split_terminator(' ').take(1).collect::<String>();
    let white_pieces: String = w_pieces.split('/').skip(4).take(4).join("/");
    let black_pieces: String = black.split('/').take(4).join("/");

    let white_castling: String = white.split(' ').skip(2).take(1).join("").chars().take(2).collect();
    let black_castling: String = black.split(' ').skip(2).take(1).join("").chars().skip(2).take(2).collect();

    format!("{}/{} w {}{} - 0 1", black_pieces, white_pieces, white_castling, black_castling)
}

fn play_opening(
    remaining_plies: i32, hh: &HistoryHeuristics, move_gen: &mut MoveGenerator, board: &mut Board,
    openings: &mut Vec<String>,
) {
    if remaining_plies == 0 {
        openings.push(write_fen(board));
        return;
    }

    move_gen.enter_ply(board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

    while let Some(m) = move_gen.next_root_move(hh, board) {
        let (previous_piece, move_state) = board.perform_move(m);
        if board.is_in_check(board.active_player().flip()) {
            // Invalid move
            board.undo_move(m, previous_piece, move_state);
            continue;
        }

        play_opening(remaining_plies - 1, hh, move_gen, board, openings);

        board.undo_move(m, previous_piece, move_state);
    }

    move_gen.leave_ply();
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

    let tt = TranspositionTable::new(16);
    let stop = Arc::new(AtomicBool::new(false));

    let limits = SearchLimits::new(None, Some(10), None, None, None, None, None, None, None).unwrap();
    let mut search =
        Search::new(stop, Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, create_from_fen(START_POS), false);

    let (_tx, rx) = mpsc::channel::<Message>();

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        collect_quiet_pos(Some(&rx), tx, &mut rnd, opening.as_str(), &tb, &mut search);
    }
}

fn select_move(rx: Option<&Receiver<Message>>, rnd: &mut Random, search: &mut Search, min_depth: i32) -> Move {
    let mut move_candidates = Vec::with_capacity(4);
    let mut min_score = i32::MIN;
    let mut latest_move = NO_MOVE;

    loop {
        let (m, _) = search.find_best_move(rx, min_depth, &move_candidates);
        if m == NO_MOVE {
            break;
        }

        if !move_candidates.is_empty() && m.score() < min_score {
            break;
        }

        if move_candidates.is_empty() {
            min_score = m.score() - 2;
        }

        latest_move = m;

        if !move_candidates.is_empty() || rnd.rand32() & 1 == 1 {
            break;
        }

        move_candidates.push(m.without_score());
    }

    latest_move
}

fn collect_quiet_pos(
    rx: Option<&Receiver<Message>>, tx: &Sender<TestPos>, rnd: &mut Random, opening: &str, tb: &Tablebase<Chess>,
    search: &mut Search,
) {
    read_fen(&mut search.board, opening).unwrap();

    let mut num = 0;

    while num < 120 {
        num += 1;

        if search.board.is_draw() {
            return;
        }

        let selected_move = select_move(rx, rnd, search, 2);
        if selected_move == NO_MOVE {
            return;
        }

        let (previous_piece, move_state) = search.board.perform_move(selected_move);

        if num > 8 && selected_move.score().abs() < (MATE_SCORE - MAX_DEPTH as i32 * 2) && selected_move.is_quiet() {
            let mut qs_pv = PrincipalVariation::default();
            search.set_stopped(false);
            search.board.active_player().score(search.quiescence_search::<true>(
                rx,
                search.board.active_player(),
                MIN_SCORE,
                MAX_SCORE,
                0,
                None,
                &mut qs_pv,
            ));

            if qs_pv.is_empty() {
                let bm = search.find_best_move(rx, 8, &[]);
                if !bm.0.is_quiet() {
                    continue;
                }
                let base_score = search.board.active_player().score(bm.0.score());
                let (mut score, divider) = eval_pos(rx, tb, search, base_score as i32, 1);
                score /= divider;

                if score.abs() <= 4000 {
                    search.board.undo_move(selected_move, previous_piece, move_state);
                    tx.send(TestPos { fen: write_fen(&search.board), score }).expect("Could not send position");
                    search.board.perform_move(selected_move);
                }
            }
        }
    }
}

fn eval_pos(
    rx: Option<&Receiver<Message>>, tb: &Tablebase<Chess>, search: &mut Search, prev_score: i32, depth: i64,
) -> (i32, i32) {
    if search.board.is_draw() {
        return (((prev_score as i64 * depth) / (depth + 1)) as i32, 2);
    }

    if search.board.occupancy_bb().piece_count() <= 5 {
        if let Some(tb_hit) = tablebase_result(tb, &write_fen(&search.board)) {
            let tb_result = tb_hit.0;
            let tb_score = tb_result * 8000 - (tb_hit.1 * 10 * tb_result.signum());

            let divider = if tb_result == 0 { 2 } else { 1 };
            return (((prev_score as i64 * depth + tb_score as i64) / (depth + 1)) as i32, divider);
        }
    }

    let search_depth = 4 - log2(depth as u32).min(3);
    let (m, _) = search.find_best_move(rx, search_depth, &[]);
    if m == NO_MOVE {
        return (prev_score, 1);
    }

    let pos_score = search.board.active_player().score(m.score());

    if is_mate_score(pos_score) {
        return (((prev_score as i64 * depth + pos_score as i64) / (depth + 1)) as i32, 1);
    }

    let (previous_piece, move_state) = search.board.perform_move(m);
    let (eval_score, divider) = eval_pos(rx, tb, search, pos_score, depth + 1);
    search.board.undo_move(m, previous_piece, move_state);

    let score = ((prev_score as i64 * depth + eval_score as i64) / (depth + 1)) as i32;

    (score, divider)
}

#[inline]
fn log2(i: u32) -> i32 {
    (32 - i.leading_zeros()) as i32 - 1
}

fn tablebase_result(tb: &Tablebase<Chess>, fen: &str) -> Option<(i32, i32)> {
    let mut pos: Chess = fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();
    let mut distance = 0;

    loop {
        match pos.outcome() {
            None => {}
            Some(outcome) => {
                return Some((outcome_to_result(outcome), distance));
            }
        }
        match tb.best_move(&pos) {
            Ok(result) => match result {
                None => panic!("Missing best move from table base"),
                Some((m, _)) => {
                    pos = pos.play(&m).expect("TB returned invalid move");
                }
            },
            Err(e) => {
                println!("Tablebase probe failed: {}", e);
                return None;
            }
        }
        distance += 1;
        if distance > 100 {
            return Some((0, distance));
        }
    }
}

fn outcome_to_result(outcome: Outcome) -> i32 {
    match outcome {
        Outcome::Decisive { winner } => {
            if winner.is_white() {
                1
            } else {
                -1
            }
        }
        Outcome::Draw => 0,
    }
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e),
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
