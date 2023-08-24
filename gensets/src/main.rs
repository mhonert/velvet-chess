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
use std::io::{BufRead, BufReader};
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
use velvet::moves::{Move, NO_MOVE};
use velvet::nn::init_nn_params;
use velvet::pieces::EMPTY;
use velvet::random::Random;
use velvet::scores::{MAX_SCORE, MIN_SCORE};
use velvet::search::{PrincipalVariation, Search};
use velvet::syzygy;
use velvet::time_management::SearchLimits;
use velvet::transposition_table::{TranspositionTable};
use crate::rescore::rescore;
use crate::writer::OutputWriter;

mod chess960;
mod rescore;
mod writer;

#[derive(Clone, Debug)]
enum Command {
    AddTestPos(String, i16),
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
    // frc: bool,
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

    let chess960 = false; //args.frc;

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
    spawn_threads(&tx, concurrency, Arc::new(openings), random_moves);

    println!("Creating output writers");
    let mut writers = vec![
        OutputWriter::new("1000"),
        OutputWriter::new("2000"),
        OutputWriter::new("3000"),
        OutputWriter::new("4000"),
    ];

    let mut count: usize = 0;
    let mut sub_count: usize = 0;

    println!("Setting CTRL-C handler");

    ctrlc::set_handler(move || tx.send(Command::Terminate).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let start = Instant::now();
    let mut start_batch = Instant::now();

    println!("Waiting for generated test positions ...");
    for cmd in rx {
        match cmd {
            Command::AddTestPos(fen, score) => {
                writers[score.unsigned_abs() as usize / 1000].add(fen, score);
                sub_count += 1;
                count += 1;
                if sub_count >= 10_000 {
                    let batch_duration_secs = Instant::now().duration_since(start_batch).as_millis() as f64 / 1000.0;
                    if batch_duration_secs > 0.0 {
                        let batch_per_minute = (sub_count as f64 / batch_duration_secs) * 60.0;
                        let duration_secs = Instant::now().duration_since(start).as_millis() as f64 / 1000.0;
                        let per_minute = (count as f64 / duration_secs) * 60.0;

                        println!("- generated {} test positions (curr. {:.2} per minute / avg. {:.2} per minute)", count, batch_per_minute, per_minute);
                    }
                    start_batch = Instant::now();
                    sub_count = 0;
                }
            }

            Command::Terminate => {
                println!("Stopping training set generation ...");
                for writer in writers.iter_mut() {
                    writer.terminate();
                }
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

    let mut prev_capture_pos: i32 = -1;

    let mut random_moves = 0;
    let mut random_prob = 1;
    let mut allow_random_moves = true;

    loop {
        ply += 1;
        if ply > 400 {
            break;
        }

        if search.board.is_draw() {
            break;
        }

        let max_candidate_count = if ply <= 80 { 2 } else { 1 };
        let (max_score, candidate_moves) = find_candidate_moves(search, rx, 4, max_candidate_count, prev_capture_pos);
        if candidate_moves.is_empty() {
            break;
        }

        let selected_move = candidate_moves[rnd.rand32() as usize % candidate_moves.len()];

        if max_score < 4000 && ply >= 10 && is_quiet(&mut search.board, selected_move) && search.board.halfmove_clock() < 42 {
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

            if qs_pv.is_empty() {
                let nodes = if search.board.occupancy_bb().count() <= 8 {
                    allow_random_moves = false;
                    2000
                } else {
                    1000
                };
                search.set_node_limit(nodes);
                let (scored_move, _) = search.find_best_move(rx, 8, &[]);
                let score = search.board.active_player().score(scored_move.score());
                if score.abs() < 4000 && is_quiet(&mut search.board, scored_move) {
                    let fen = write_fen(&search.board);
                    tx.send(Command::AddTestPos(fen.clone(), score)).expect("Could not send position");
                }
            }
        }

        let (_, removed_piece_id) = search.board.perform_move(selected_move.unpack());
        if removed_piece_id != EMPTY {
            prev_capture_pos = selected_move.end();
        } else {
            prev_capture_pos = -1;
            if allow_random_moves && ply >= 25 && random_moves < 2 && !search.board.is_in_check(search.board.active_player()) && (rnd.rand32() % random_prob) == 0 {
                random_moves += 1;
                play_random_move(rnd, &search.hh, &mut search.movegen, &mut search.board);
            }
        }

        random_prob += 1;
    }
}

fn is_quiet(board: &mut Board, m: Move) -> bool {
    if !m.is_quiet() {
        return false;
    }
    if board.is_in_check(board.active_player()) {
        return false;
    }

    let upm = m.unpack();

    let (own_piece, removed_piece_id) = board.perform_move(upm);
    let mut quiet = removed_piece_id == EMPTY;

    quiet &= !board.is_in_check(board.active_player());

    board.undo_move(upm, own_piece, removed_piece_id);

    quiet
}

fn find_candidate_moves(search: &mut Search, rx: Option<&Receiver<Message>>, min_depth: i32, max_candidate_count: usize, prev_capture_pos: i32) -> (i16, Vec<Move>) {
    let mut candidates = Vec::with_capacity(max_candidate_count);
    let mut found_recapture = false;
    let mut max_score = i16::MIN;
    search.set_node_limit(100);
    for _ in 0..max_candidate_count {
        let (selected_move, _) = search.find_best_move(rx, min_depth, &candidates);
        if selected_move == NO_MOVE {
            break;
        }
        let score = search.board.active_player().score(selected_move.score());
        max_score = max_score.max(score);
        let is_recapture = selected_move.end() == prev_capture_pos;

        if found_recapture && !is_recapture {
            continue;
        }
        candidates.push(selected_move.without_score());

        found_recapture |= is_recapture;
    }
    (max_score, candidates)
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

        let upm = m.unpack();
        let (previous_piece, removed_piece_id) = board.perform_move(upm);
        if board.is_in_check(board.active_player().flip()) || board.is_draw() {
            board.undo_move(upm, previous_piece, removed_piece_id);
            continue;
        }
        candidate_moves.push(m);

        board.undo_move(upm, previous_piece, removed_piece_id);
    }
    move_gen.leave_ply();

    if candidate_moves.is_empty() {
        return false;
    }

    let m = candidate_moves[rnd.rand32() as usize % candidate_moves.len()];
    board.perform_move(m.unpack());

    true
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e),
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
