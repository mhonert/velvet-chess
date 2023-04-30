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

use std::cmp::Ordering;
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
use shakmaty::{CastlingMode, Chess, Setup};
use shakmaty_syzygy::Tablebase;

use crate::chess960::CHESS960_FENS;
use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, read_fen, write_fen, START_POS};
use velvet::history_heuristics::HistoryHeuristics;
use velvet::move_gen::MoveGenerator;
use velvet::moves::{NO_MOVE};
use velvet::nn::init_nn_params;
use velvet::random::Random;
use velvet::scores::{is_eval_score, MAX_SCORE, MIN_SCORE};
use velvet::search::{PrincipalVariation, Search};
use velvet::time_management::SearchLimits;
use velvet::transposition_table::{TranspositionTable};

mod chess960;

#[derive(Clone, Debug)]
struct TestPos {
    fen: String,
    score: i32,
    tb_result: Option<i32>,
    tb_result_ply: i32,
    tb_result_dtz: i32,
    ply: i32,
    game_result: i32,
    game_result_ply: i32,
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage(
            "-i, --start-index=<START>       'Sets the start index for the generated training sets'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -f, --frc=<BOOL>                       'Generates training sets for Chess960'
             -r, --rnd-moves=<NUM>                  'Sets the number of random moves'
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

    let random_moves = match matches.value_of("rnd-moves") {
        Some(v) => u8::from_str(v).expect("rnd-moves must be a positive integer >= 0"),
        None => {
            eprintln!("Missing -r (rnd-moves) option");
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
    spawn_threads(&tx, concurrency, &openings, random_moves, String::from(tb_path));

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
        writeln!(&mut writer, "{} {} {} {} {} {} {} {}", pos.fen, pos.ply, pos.tb_result.unwrap_or(0xFF),
                 pos.tb_result_ply, pos.tb_result_dtz, pos.game_result, pos.game_result_ply, pos.score).expect("Could not write position to file");
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

    if chess960 {
        for w_fen in CHESS960_FENS.iter() {
            for b_fen in CHESS960_FENS.iter() {
                let fen = mix(w_fen, b_fen);
                openings.push(fen);
            }
        }
    } else {
        openings.push(String::from(START_POS));
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

fn spawn_threads(tx: &Sender<TestPos>, concurrency: usize, openings: &[String], random_moves: u8, tb_path: String) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.to_owned();
        let tb_path2 = tb_path.clone();
        thread::spawn(move || {
            find_test_positions(&tx2, &openings2, random_moves, tb_path2);
        });
    }
}

fn find_test_positions(tx: &Sender<TestPos>, openings: &[String], random_moves: u8, tb_path: String) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let mut tb = Tablebase::new();
    println!("Setting tablebase path to: {}", tb_path);
    tb.add_directory(tb_path).expect("Could not add tablebase path");

    let tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    let limits = SearchLimits::new(None, Some(10), None, None, None, None, None, None, None).unwrap();
    let mut search =
        Search::new(stop, Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, create_from_fen(START_POS), false);

    let (_tx, rx) = mpsc::channel::<Message>();

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        collect_quiet_pos(Some(&rx), tx, opening.as_str(), random_moves, &tb, &mut search, &mut rnd);
    }
}

fn collect_quiet_pos(
    rx: Option<&Receiver<Message>>, tx: &Sender<TestPos>, opening: &str, random_moves: u8, tb: &Tablebase<Chess>,
    search: &mut Search, rnd: &mut Random,
) {
    read_fen(&mut search.board, opening).unwrap();

    let mut ply = 0;
    for _ in 0..random_moves {
        if !play_random_move(rnd, &search.hh, &mut search.movegen, &mut search.board) {
            return;
        }
        ply += 1;
    }

    let mut positions: Vec<TestPos> = Vec::new();

    let mut game_result = 0;

    let mut last_score = 0;

    loop {
        ply += 1;

        if search.board.is_draw() {
            break;
        }

        let mut tb_result = None;
        let mut tb_result_ply = 0;
        let mut tb_result_dtz = 0;
        if search.board.piece_count() <= 6 {
            if let Some((result, dtz)) = tablebase_result(tb, write_fen(&search.board).as_str()) {
                tb_result = Some(result);
                tb_result_ply = ply;
                tb_result_dtz = dtz;
                for pos in positions.iter_mut().rev() {
                    if pos.game_result != 0xFF {
                        pos.game_result = result;
                    }
                }
            }
        }

        // search.set_node_limit(5000);
        let (selected_move, _) = search.find_best_move(rx, 8, &[]);
        if selected_move == NO_MOVE {
            break;
        }

        let score = search.board.active_player().score(selected_move.score());

        if ply >= 12 && score.abs() <= 3000 && selected_move.is_quiet() && !search.board.is_in_check(search.board.active_player()) {
            let mut qs_pv = PrincipalVariation::default();
            search.set_stopped(false);
            search.quiescence_search::<true>(
                rx,
                search.board.active_player(),
                MIN_SCORE,
                MAX_SCORE,
                0,
                None,
                &mut qs_pv,
            );

            if qs_pv.is_empty() {
                let fen = write_fen(&search.board);
                positions.push(TestPos { fen: fen.clone(), score, tb_result, tb_result_ply, tb_result_dtz, game_result: 0xFF, ply, game_result_ply: 0 });
            }
        }
        last_score = score;
        if last_score.abs() > 3000 {
            break;
        }

        let _ = search.board.perform_move(selected_move);
    }

    if last_score < -3000 {
        game_result = -1;
    } else if last_score > 3000 {
        game_result = 1;
    }

    for pos in positions.iter_mut().rev() {
        if ply >= 160 && game_result == 0 {
            break;
        }

        if pos.game_result == 0xFF {
            pos.game_result = game_result;
        }
        pos.game_result_ply = ply;

        tx.send(pos.clone()).expect("Could not send position");
    }
}

fn play_random_move(
    rnd: &mut Random, hh: &HistoryHeuristics, move_gen: &mut MoveGenerator, board: &mut Board,
) -> bool {
    if board.is_in_check(board.active_player()) || board.is_draw() {
        return false;
    }

    move_gen.enter_ply(board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

    let mut candidate_moves = Vec::new();

    while let Some(m) = move_gen.next_root_move(hh, board) {
        let (previous_piece, move_state) = board.perform_move(m);
        if board.is_in_check(board.active_player().flip()) || board.is_draw() {
            board.undo_move(m, previous_piece, move_state);
            continue;
        }
        candidate_moves.push(m);

        board.undo_move(m, previous_piece, move_state);
    }
    move_gen.leave_ply();

    if candidate_moves.is_empty() {
        return false;
    }

    let m = candidate_moves[rnd.rand32() as usize % candidate_moves.len()];
    board.perform_move(m);

    true
}

fn tablebase_result(tb: &Tablebase<Chess>, fen: &str) -> Option<(i32, i32)> {
    let pos: Chess = fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();
    if pos.board().pieces().count() > 6 {
        return None;
    }

    let win_white_pov = if pos.turn().is_white() { 1 } else { -1 };

    match tb.probe_dtz(&pos) {
        Ok(result) => {
            let r = result.ignore_rounding().0;
            match r.cmp(&0) {
                Ordering::Less => {
                    if r < -100 {
                        Some((0, 100))
                    } else {
                        Some((-win_white_pov, -r))
                    }
                },

                Ordering::Greater => {
                    if r > 100 {
                        Some((0, 100))
                    } else {
                        Some((win_white_pov, r))
                    }
                },

                Ordering::Equal => {
                    Some((0, 0))
                }
            }
        }
        Err(_) => {
            None
        }
    }
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e),
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
