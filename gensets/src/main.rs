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

use std::cmp::max;
use std::collections::HashSet;
use std::thread;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Sender, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::App;

use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{START_POS, create_from_fen, write_fen};
use velvet::history_heuristics::HistoryHeuristics;
use velvet::magics::initialize_magics;
use velvet::move_gen::MoveGenerator;
use velvet::moves::{Move, NO_MOVE};
use velvet::random::Random;
use velvet::scores::{MAX_SCORE, MIN_SCORE};
use velvet::search::{PrincipalVariation, Search, SearchLimits};
use velvet::transposition_table::TranspositionTable;

pub mod gen_quiet_pos;

#[derive(Clone, Debug)]
struct TestPos {
    fen: Option<String>,
    count: u16,
    prev_score: i32,
    score: i32,
    is_quiet: bool,
    mat_score: i32,
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage(
            "-i, --start-index=<START>              'Sets the start index for the generated training sets'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'")
        .get_matches();

    let start_index = i32::from_str(matches.value_of("start-index").unwrap()).expect("Start index must be an integer");

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

    // println!("Reading opening book: {} ...", book_file);
    // let openings = read_openings(book_file);
    let openings = gen_openings();

    let (tx, rx) = mpsc::channel::<TestPos>();

    println!("Starting worker threads ...");
    let mut start = Instant::now();
    spawn_threads(&tx, concurrency, &openings);

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
        writeln!(&mut writer, "{} {} {}", pos.fen.unwrap(), pos.count, pos.score).expect("Could not write position to file");
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

fn gen_openings() -> Vec<String> {
    let mut openings = Vec::with_capacity(120000);
    let hh = HistoryHeuristics::new();
    let mut board = create_from_fen(START_POS);
    let mut move_gen = MoveGenerator::new();

    play_opening(4, &hh, &mut move_gen, &mut board, &mut openings);

    openings.sort_unstable();
    openings.dedup();

    println!("Generated {} openings", openings.len());

    openings
}

fn play_opening(remaining_plies: i32, hh: &HistoryHeuristics, move_gen: &mut MoveGenerator, board: &mut Board, openings: &mut Vec<String>) {
    if remaining_plies == 0 {
        openings.push(write_fen(board));
        return;
    }

    move_gen.enter_ply(board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE);

    while let Some(m) = move_gen.next_legal_move(hh, board) {
        let (previous_piece, move_state) = board.perform_move(m);
        if board.is_in_check(-board.active_player()) {
            // Invalid move
            board.undo_move(m, previous_piece, move_state);
            continue;
        }

        play_opening(remaining_plies - 1, hh, move_gen, board, openings);

        board.undo_move(m, previous_piece, move_state);
    }

    move_gen.leave_ply();
}

fn spawn_threads(tx: &Sender<TestPos>, concurrency: usize, openings: &[String]) {
    for pos in 0..concurrency {
        thread::sleep(Duration::from_millis(pos as u64 * 10));
        let tx2 = tx.clone();
        let openings2 = openings.to_owned();
        thread::spawn(move || {
            find_test_positions(&tx2, &openings2);
        });
    }
}

fn find_test_positions(tx: &Sender<TestPos>, openings: &[String]) {
    let mut rnd = Random::new_with_seed(new_rnd_seed());

    let tt = Arc::new(TranspositionTable::new(128));


    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        let mut positions = collect_quiet_pos(&mut rnd, opening.as_str(), tt.clone());
        for pos in positions.iter_mut() {
            if !pos.is_quiet {
                continue;
            }

            tx.send(pos.clone()).expect("could not send test position");

            // println!("{:82}: {:5} -> {:5}", pos.fen.as_ref().unwrap(), pos.prev_score, pos.score);
        }
    }
}

fn select_move(rx: Option<&Receiver<Message>>, rnd: &mut Random, move_variety: bool, search: &mut Search, min_depth: i32) -> (Move, PrincipalVariation) {
    let mut move_candidates = Vec::with_capacity(4);
    let mut min_score = i32::MIN;
    let mut latest_move = NO_MOVE;
    let mut chosen_pv = PrincipalVariation::new();

    loop {
        let (m, pv) = search.find_best_move(rx, min_depth, &move_candidates);
        if m == NO_MOVE {
            break;
        }

        if !move_candidates.is_empty() && m.score() < min_score {
            break;
        }

        if move_candidates.is_empty() {
            min_score = m.score() - 30;
        }

        latest_move = m;
        chosen_pv = pv.clone();
        if !move_variety || rnd.rand32() & 1 == 1 || move_candidates.len() == 4 {
            break;
        }

        move_candidates.push(m.without_score());
    }

    (latest_move, chosen_pv)
}

fn collect_quiet_pos(rnd: &mut Random, opening: &str, tt: Arc<TranspositionTable>) -> Vec<TestPos> {

    tt.clear();

    let mut duplicate_check = HashSet::new();
    let board = create_from_fen(opening);

    let (_tx, rx) = mpsc::channel::<Message>();
    let node_limit = 20000 + (rnd.rand32() % 2000) as u64;
    let limits = SearchLimits::new(None, Some(node_limit), None, None, None, None, Some(200), None).unwrap();
    let mut search = Search::new(Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt.clone(), board, 1);


    let mut positions = Vec::new();
    let mut ply = 0;

    let mut move_variety = true;

    loop {
        ply += 1;

        search.set_node_limit(node_limit);
        let (selected_move, pv) = select_move(Some(&rx), rnd, move_variety, &mut search, 8);

        if selected_move == NO_MOVE {
            let (result, description) = if search.board.is_in_check(search.board.active_player()) {
                ((-search.board.active_player()) as i32, "Mate score")
            } else {
                (0, "Draw")
            };

            return adjust_scores(positions, result, search.board.halfmove_count(), String::from(description));
        }

        let score = selected_move.score() * search.board.active_player() as i32;
        move_variety = ply <= 8 && score.abs() <= 200;

        if search.board.is_insufficient_material_draw() {
            return adjust_scores(positions, 0, search.board.halfmove_count(), String::from("Insufficient material draw"));
        } else if search.board.is_repetition_draw() {
            return adjust_scores(positions, 0, search.board.halfmove_count(), String::from("Repetition draw"));
        } else if search.board.halfmove_clock() >= 30 && score.abs() <= 400 {
            return adjust_scores(positions, 0, search.board.halfmove_count(), String::from("Draw scores"));
        }

        if score.abs() >= 1500 {
            return adjust_scores(positions, score.signum(), search.board.halfmove_count(), String::from("Score >= 1500"));
        }

        if ply >= 8 && pv.moves().len() >= 7 && !search.board.is_in_check(search.board.active_player()) {
            let quiet_fen = extract_quiet_fen(Some(&rx), &mut duplicate_check, &mut search, &pv.moves());
            let is_quiet = quiet_fen.is_some();
            positions.push(TestPos{
                fen: quiet_fen,
                count: search.board.halfmove_count(),
                prev_score: score,
                score,
                is_quiet,
                mat_score: search.board.material_score(),
            });
        } else {
            positions.push(TestPos{
                fen: None,
                count: search.board.halfmove_count(),
                prev_score: score,
                score,
                is_quiet: false,
                mat_score: search.board.material_score(),
            });

        }

        search.board.perform_move(selected_move);
    }
}

fn extract_quiet_fen(rx: Option<&Receiver<Message>>, duplicate_check: &mut HashSet<String>, search: &mut Search, pv: &[Move]) -> Option<String> {

    let mut qs_pv = PrincipalVariation::new();
    search.quiescence_search(search.board.active_player(), MIN_SCORE, MAX_SCORE, 0, None, &mut qs_pv);

    if !qs_pv.moves().is_empty() {
        if let Some((m, rest_pv)) = pv.split_first() {
            let (previous_piece, move_state) = search.board.perform_move(*m);
            let result = extract_quiet_fen(rx, duplicate_check, search, rest_pv);
            search.board.undo_move(*m, previous_piece, move_state);

            return result;
        }
    }

    if search.board.is_in_check(search.board.active_player()) {
        return None;
    }

    // Reached end of PV
    let fen = write_fen(&search.board);
    if duplicate_check.contains(&fen) {
        return None;
    }
    duplicate_check.insert(fen.clone());

    Some(fen)
}

fn adjust_scores(mut positions: Vec<TestPos>, mut game_result: i32, end_count: u16, _result: String) -> Vec<TestPos> {
    if positions.len() <= 2 {
        return Vec::new();
    }

    game_result *= 2000;
    if game_result == 0 {
        let end = positions.len();
        for i in 0..positions.len() {
            positions[end - i - 1].score /= 2;
        }
    }

    for i in (1..positions.len()).rev() {
        if positions[i].mat_score != positions[i - 1].mat_score {
            for j in (max(0, i as i32 - 8) as usize)..i {
                positions[j].is_quiet = false;
            }
        }
    }

    let last_index = positions.len() - 1;
    let curr_score = positions[last_index].score;

    for _ in 0..4 {
        positions[last_index].score = (game_result * (end_count as i32 * 2) + curr_score * positions[last_index].count as i32) / (end_count as i32 * 2 + positions[last_index].count as i32);
        for i in (1..positions.len()).rev() {
            let last_score = positions[i].score;
            let curr_score = positions[i - 1].score;

            positions[i - 1].score = (last_score + curr_score) / 2;
        }
    }

    // println!("-----------------------------------------------------------");
    // println!("{}: {}", result, game_result);

    positions
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
