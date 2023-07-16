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

use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, Parser, Subcommand};

use crate::chess960::CHESS960_FENS;
use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, read_fen, write_fen, START_POS};
use velvet::history_heuristics::HistoryHeuristics;
use velvet::move_gen::MoveGenerator;
use velvet::moves::{NO_MOVE};
use velvet::nn::init_nn_params;
use velvet::random::Random;
use velvet::scores::{MAX_SCORE, MIN_SCORE};
use velvet::search::{PrincipalVariation, Search};
use velvet::syzygy;
use velvet::time_management::SearchLimits;
use velvet::transposition_table::{TranspositionTable};
use crate::rescore::rescore;

mod chess960;
mod rescore;

#[derive(Clone, Debug)]
enum Command {
    AddTestPos(String, i32),
    Terminate
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args, Debug)]
pub struct ExtractArgs {
    input_dir: String
}

#[derive(Args, Debug)]
pub struct RescoreArgs {
    input_pattern: String,
    tb_path: String,
    concurrency: usize,
}

#[derive(Args, Debug)]
struct GenerateArgs {
    start_index: usize,
    concurrency: usize,
    frc: bool,
    rnd_moves: u8,
    opening_file: Option<String>,
    tb_path: String
}

#[derive(Subcommand, Debug)]
enum Commands {
    Generate(GenerateArgs),
    Rescore(RescoreArgs),
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Commands::Generate(args) => generate(args),
        Commands::Rescore(args) => rescore(args),
    }
}


fn generate(args: GenerateArgs) {
    let start_index = args.start_index;

    let tb_path = args.tb_path;

    if !syzygy::tb::init(tb_path.clone()) {
        eprintln!("could not initialize tablebases using path: {}", tb_path.as_str());
        exit(1);
    } else {
        let count = syzygy::tb::max_piece_count();
        if count == 0 {
            println!("debug no tablebases found");
        } else {
            println!("debug found {}-men tablebases", syzygy::tb::max_piece_count());
        }
    }

    let concurrency = args.concurrency;

    let random_moves = args.rnd_moves;

    let chess960 = args.frc;

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let openings = if let Some(opening_file) = args.opening_file {
        read_openings(&opening_file)
    } else {
        gen_openings(chess960)
    };

    let (tx, rx) = mpsc::channel::<Command>();

    init_nn_params();
    println!("Starting worker threads ...");
    let mut start = Instant::now();
    spawn_threads(&tx, concurrency, Arc::new(openings), random_moves);

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

    for cmd in rx {
        match cmd {
            Command::AddTestPos(fen, score) => {
                writeln!(&mut writer, "{} {}", fen, score).expect("Could not write position to file");
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
            Command::Terminate => {
                break;
            }
        }
    }

    println!("End");
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

        openings.push(line.trim().to_string());
    }
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

fn spawn_threads(tx: &Sender<Command>, concurrency: usize, openings: Arc<Vec<String>>, random_moves: u8) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.clone();
        thread::spawn(move || {
            find_test_positions(&tx2, openings2, random_moves);
        });
    }
}

fn find_test_positions(tx: &Sender<Command>, openings: Arc<Vec<String>>, random_moves: u8) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    let limits = SearchLimits::new(None, Some(10), None, None, None, None, None, None, None).unwrap();
    let mut search =
        Search::new(stop, Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, create_from_fen(START_POS), false);

    search.set_tb_probe_root(false);

    let (_tx, rx) = mpsc::channel::<Message>();

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        collect_quiet_pos(Some(&rx), tx, opening.as_str(), random_moves, &mut search, &mut rnd);
    }
}

fn collect_quiet_pos(
    rx: Option<&Receiver<Message>>, tx: &Sender<Command>, opening: &str, random_moves: u8,
    search: &mut Search, rnd: &mut Random
) {
    read_fen(&mut search.board, opening).unwrap();

    let mut ply = search.board.halfmove_count() as i32;
    for _ in 0..random_moves {
        if !play_random_move(rnd, &search.hh, &mut search.movegen, &mut search.board) {
            return;
        }
        ply += 1;
    }

    loop {
        ply += 1;
        if ply > 300 {
            break;
        }

        if search.board.is_draw() {
            break;
        }

        search.set_node_limit(10000);
        let (mut selected_move, _) = search.find_best_move(rx, 8, &[]);
        if selected_move == NO_MOVE {
            break;
        }

        let mut score = search.board.active_player().score(selected_move.score());

        if selected_move.is_quiet() && ((ply <= 30 && rnd.rand32() & 3 == 0) || (ply <= 60 && rnd.rand32() & 7 == 0) || (ply <= 90 && rnd.rand32() & 15 == 0)) {
            let (alt_selected_move, _) = search.find_best_move(rx, 8, &[selected_move.without_score()]);
            let alt_score = search.board.active_player().score(alt_selected_move.score());
            if alt_selected_move != NO_MOVE && alt_selected_move.score() >= selected_move.score() {
                // println!("Used alternative move: {}", write_fen(&search.board));
                // println!(" - {} = {}", UCIMove::from_move(&search.board, selected_move), score);
                // println!(" - {} = {}", UCIMove::from_move(&search.board, alt_selected_move), alt_score);
                score = alt_score;
                selected_move = alt_selected_move;
            }
        }

        if ply >= 10 && score.abs() <= 1000 && selected_move.is_quiet() && !search.board.is_in_check(search.board.active_player()) {
            let mut qs_pv = PrincipalVariation::default();
            search.set_stopped(false);
            search.quiescence_search::<true>(
                None,
                search.board.active_player(),
                MIN_SCORE,
                MAX_SCORE,
                0,
                None,
                &mut qs_pv,
            );

            if qs_pv.is_empty() && search.board.halfmove_clock() < 42 {
                let fen = write_fen(&search.board);
                tx.send(Command::AddTestPos(fen.clone(), score)).expect("Could not send position");
            }
        }

        let _ = search.board.perform_move(selected_move);
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
        let captured_piece_id = board.get_item(m.end()).abs();
        if captured_piece_id < m.piece_id()
            && board.has_negative_see(
            board.active_player().flip(),
            m.start(),
            m.end(),
            m.piece_id(),
            captured_piece_id,
            0,
            board.occupancy_bb(),
        ) {
            continue;
        }

        let (previous_piece, removed_piece_id) = board.perform_move(m);
        if board.is_in_check(board.active_player().flip()) || board.is_draw() {
            board.undo_move(m, previous_piece, removed_piece_id);
            continue;
        }
        candidate_moves.push(m);

        board.undo_move(m, previous_piece, removed_piece_id);
    }
    move_gen.leave_ply();

    if candidate_moves.is_empty() {
        return false;
    }

    let m = candidate_moves[rnd.rand32() as usize % candidate_moves.len()];
    board.perform_move(m);

    true
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e),
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
