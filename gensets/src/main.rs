/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::process::exit;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::{Args, Parser, Subcommand};

use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, read_fen, write_fen, START_POS};
use velvet::init::init;
use velvet::moves::{Move, NO_MOVE};
use velvet::random::Random;
use velvet::search::{Search};
use velvet::syzygy;
use velvet::time_management::SearchLimits;
use velvet::transposition_table::{TranspositionTable};
use crate::writer::{NextIDSource, OutputWriter};

mod writer;

#[derive(Clone, Debug)]
enum Command {
    UpdateCount(usize),
    RequestTermination,
    ThreadTerminated
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
pub struct ConvertArgs {
    input_file: String,
}

#[derive(Args, Debug)]
struct GenerateArgs {
    concurrency: usize,
    opening_file: String,
    tb_path: String
}

#[derive(Subcommand, Debug)]
enum Commands {
    Generate(GenerateArgs),
}

fn main() {
    let args = Cli::parse();
    match args.command {
        Commands::Generate(args) => generate(args),
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

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let openings = read_openings(&args.opening_file);

    let (tx, rx) = mpsc::channel::<Command>();

    init();
    println!("Starting worker threads ...");
    let stop = Arc::new(AtomicBool::default());
    let id_source = Arc::new(NextIDSource::new());
    spawn_threads(&tx, concurrency, Arc::new(openings), id_source, stop.clone());

    let mut count = 0;
    let mut sub_count = 0;
    let mut running = concurrency;

    println!("Setting CTRL-C handler");

    ctrlc::set_handler(move || tx.send(Command::RequestTermination).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let start = Instant::now();
    let mut start_batch = Instant::now();

    println!("Waiting for generated test positions ...");
    for cmd in rx {
        match cmd {
            Command::UpdateCount(update_count) => {
                sub_count += update_count;
                count += update_count;
                let batch_duration_secs = Instant::now().duration_since(start_batch).as_millis() as f64 / 1000.0;
                if batch_duration_secs >= 10.0 {
                    let batch_per_minute = (sub_count as f64 / batch_duration_secs) * 60.0;
                    let duration_secs = Instant::now().duration_since(start).as_millis() as f64 / 1000.0;
                    let per_minute = (count as f64 / duration_secs) * 60.0;

                    println!("- generated {} test positions (curr. {:.2} per minute / avg. {:.2} per minute)", count, batch_per_minute, per_minute);
                    start_batch = Instant::now();
                    sub_count = 0;
                }
            }

            Command::RequestTermination => {
                println!("Stopping training set generation ... ");
                stop.store(true, Ordering::Relaxed);
            }

            Command::ThreadTerminated => {
                running -= 1;
                if running == 0 {
                    break;
                }
                println!(" - {} thread(s) remaining", running);
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

fn spawn_threads(tx: &Sender<Command>, concurrency: usize, openings: Arc<Vec<String>>, id_source: Arc<NextIDSource>, stop: Arc<AtomicBool>) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.clone();
        let id_source2 = id_source.clone();
        let stop2 = stop.clone();
        thread::spawn(move || {
            find_test_positions(&tx2, openings2, id_source2, stop2);
        });
    }
}

fn find_test_positions(tx: &Sender<Command>, openings: Arc<Vec<String>>, id_source: Arc<NextIDSource>, stop: Arc<AtomicBool>) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let tt = TranspositionTable::new(32);

    let limits = SearchLimits::new(None, Some(10), None, None, None, None, None, None, None).unwrap();
    let mut search =
        Search::new(Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, create_from_fen(START_POS), false);
    search.set_tb_probe_root(false);

    let mut writer = OutputWriter::new(id_source);

    let (_tx, rx) = mpsc::channel::<Message>();

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        let update_count = collect_quiet_pos(Some(&rx), opening.as_str(), &mut search, &mut rnd, &mut writer);
        tx.send(Command::UpdateCount(update_count)).expect("could not send update count");

        if stop.load(Ordering::Relaxed) {
            break;
        }
    }

    writer.terminate();
    tx.send(Command::ThreadTerminated).expect("could not send thread terminated command");
}

fn collect_quiet_pos(
    rx: Option<&Receiver<Message>>, opening: &str,
    search: &mut Search, rnd: &mut Random, writer: &mut OutputWriter
) -> usize {
    read_fen(&mut search.board, opening).unwrap();

    let mut ply = search.board.halfmove_count() as i32;
    let start_ply = ply;

    let mut positions = Vec::new();

    let start_fen = write_fen(&search.board);

    let mut game_result = 0;

    loop {
        ply += 1;

        if search.board.is_repetition_draw() || search.board.is_fifty_move_draw() || search.board.is_insufficient_material_draw() {
            break;
        }

        let (best_score, candidate_moves) = find_candidate_moves(search, rx, (ply - start_ply) <= 16 && rnd.rand32() & 1 == 0);
        if candidate_moves.is_empty() {
            let active_player = search.board.active_player();
            if search.board.is_in_check(active_player) {
                game_result = active_player.score(-1);
            }
            break;
        }

        let best_move = candidate_moves[rnd.rand32() as usize % candidate_moves.len()];
        let score = search.board.active_player().score(best_score);

        positions.push(best_move.with_score(score).to_u32());

        search.board.perform_move(best_move);
    }

    let count = positions.len();
    writer.add(start_fen, game_result, positions);
    count
}

fn find_candidate_moves(search: &mut Search, rx: Option<&Receiver<Message>>, consider_alternative: bool) -> (i16, Vec<Move>) {
    let mut skip = Vec::with_capacity(2);
    let mut candidates = Vec::with_capacity(2);
    let mut best_score = i16::MIN;
    search.set_node_limit(5000);
    for _ in 0..2 {
        let (selected_move, _) = search.find_best_move_with_full_strength(rx, &skip);
        if selected_move == NO_MOVE {
            break;
        }

        let score = selected_move.score();

        if !candidates.is_empty() && score < best_score {
            break;
        }

        best_score = best_score.max(score);

        candidates.push(selected_move);
        if !consider_alternative {
            break;
        }

        skip.push(selected_move.without_score());
    }
    (best_score, candidates)
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e),
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
