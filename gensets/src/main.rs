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

use std::cmp::{min};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::str::FromStr;
use std::sync::{Arc, mpsc};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use clap::App;
use shakmaty::{CastlingMode, Chess, Position};
use shakmaty::fen::{Fen};
use shakmaty_syzygy::{Dtz, Tablebase};

use gen_quiet_pos::GenQuietPos;
use velvet::board::Board;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, START_POS, write_fen};
use velvet::history_heuristics::HistoryHeuristics;
use velvet::move_gen::MoveGenerator;
use velvet::moves::{Move, NO_MOVE};
use velvet::random::Random;
use velvet::search::{PrincipalVariation, Search, SearchLimits};
use velvet::transposition_table::TranspositionTable;
use velvet::uci_move::UCIMove;

pub mod gen_quiet_pos;

#[derive(Clone, Debug)]
struct TestPos {
    fen: Option<String>,
    count: u16,
    prev_score: i32,
    score: i32,
    include: bool
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage(
            "-i, --start-index=<START>              'Sets the start index for the generated training sets'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -t  --table-base-path=<FILE>           'Sets the Syzygy tablebase path'")
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

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    // println!("Reading opening book: {} ...", book_file);
    // let openings = read_openings(book_file);
    let openings = gen_openings();

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

    move_gen.enter_ply(board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);

    while let Some(m) = move_gen.next_root_move(hh, board) {
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

    let tt = TranspositionTable::new(128);

    loop {
        let opening = openings[rnd.rand32() as usize % openings.len()].clone();

        let mut positions = collect_quiet_pos(&tb, &mut rnd, opening.as_str(), tt.clone());
        for pos in positions.iter_mut() {
            if !pos.include {
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
    let mut chosen_pv = PrincipalVariation::default();

    loop {
        let (m, pv) = search.find_best_move(rx, min_depth, &move_candidates);
        if m == NO_MOVE {
            break;
        }

        if !move_candidates.is_empty() && m.score() < min_score {
            break;
        }

        if move_candidates.is_empty() {
            min_score = m.score() - 10;
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

fn collect_quiet_pos(tb: &Tablebase<Chess>, rnd: &mut Random, opening: &str, tt: Arc<TranspositionTable>) -> Vec<TestPos> {

    tt.clear();

    let mut duplicate_check = HashSet::new();
    let board = create_from_fen(opening);

    let (_tx, rx) = mpsc::channel::<Message>();
    let node_limit = 10000 + (rnd.rand32() % 1000) as u64;
    let limits = SearchLimits::new(None, Some(node_limit), None, None, None, None, Some(1), None).unwrap();
    let mut search = Search::new(Arc::new(AtomicBool::new(false)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, board, 1, false);

    let mut positions = Vec::new();
    let mut ply = 0;

    let mut move_variety = true;

    loop {
        ply += 1;

        let is_candidate = ply >= 8 && !search.board.is_in_check(search.board.active_player());

        search.set_node_limit(node_limit);
        let (selected_move, pv) = select_move(Some(&rx), rnd, move_variety, &mut search, 10);

        if selected_move == NO_MOVE {
            let (result, description) = if search.board.is_in_check(search.board.active_player()) {
                ((-search.board.active_player()) as i32, "Mate score")
            } else {
                (0, "Draw")
            };

            return adjust_scores(positions, result, String::from(description));
        }

        let score = selected_move.score() * search.board.active_player() as i32;
        move_variety = ply <= 12 && score.abs() <= 200;

        if search.board.is_insufficient_material_draw() {
            return adjust_scores(positions, 0, String::from("Insufficient material draw"));
        } else if search.board.is_repetition_draw() {
            return adjust_scores(positions, 0, String::from("Repetition draw"));
        } else if search.board.is_fifty_move_draw() {
            return adjust_scores(positions, 0, String::from("50-move draw"));
        }

        if search.board.get_occupancy_bitboard().count_ones() <= 5 {
            return resolve_tb_match(tb, Some(&rx), &mut duplicate_check, &mut search, positions);
        }

        let fen = write_fen(&search.board);
        let is_quiet = is_candidate && pv.moves().len() >= 10 && search.is_quiet_pv(&pv.moves(), search.material_score());

        if !duplicate_check.contains(&fen) && score.abs() < 6000 {
            duplicate_check.insert(fen.clone());
            positions.push(TestPos {
                fen: Some(fen),
                count: search.board.halfmove_count(),
                prev_score: score,
                score,
                include: is_quiet && score.abs() <= 3000
            });
        }

        search.board.perform_move(selected_move);
    }
}


fn resolve_tb_match(tb: &Tablebase<Chess>, rx: Option<&Receiver<Message>>, duplicate_check: &mut HashSet<String>, search: &mut Search, mut positions: Vec<TestPos>) -> Vec<TestPos> {
    let start_fen = write_fen(&search.board);
    let mut pos: Chess = start_fen.parse::<Fen>().expect("Could not parse FEN").position(CastlingMode::Standard).unwrap();

    let mut result = 0;

    loop {
        if pos.is_game_over() {
            result = match pos.outcome().unwrap() {
                shakmaty::Outcome::Draw => 0,
                shakmaty::Outcome::Decisive { winner } => if winner.is_white() { 1 } else { -1 },
            };
            break;
        }

        let (m, dtz) = tb.best_move(&pos).unwrap().expect("no move found!");
        if dtz == Dtz(0) {
            break;
        }
        pos = pos.play(&m).expect("invalid move");

        let uci_move = UCIMove::from_uci(&m.to_uci(CastlingMode::Standard).to_string()).unwrap();

        let m = uci_move.to_move(&search.board);
        let (score, is_quiet) = search_tb_move(rx, search, m);

        let fen = write_fen(&search.board);

        if !duplicate_check.contains(&fen) && score.abs() < 6000 {
            duplicate_check.insert(fen.clone());
            positions.push(TestPos {
                fen: Some(fen),
                count: search.board.halfmove_count(),
                prev_score: score,
                score,
                include: is_quiet && score.abs() <= 3000
            });
        }

        search.board.perform_move(m);
    }

    adjust_scores(positions, result, String::from("TB"))
}

fn search_tb_move(rx: Option<&Receiver<Message>>, search: &mut Search, tb_move: Move) -> (i32, bool) {
    search.movegen.enter_ply(search.board.active_player(), NO_MOVE, NO_MOVE, NO_MOVE, NO_MOVE);
    let mut skipped_moves = Vec::new();
    while let Some(m) = search.movegen.next_root_move(&search.hh, &mut search.board) {
        if !m.is_same_move(tb_move) {
            skipped_moves.push(m.without_score())
        }
    }
    search.movegen.leave_ply();

    // Only search the table base move, skip all other moves
    let (scored_move, pv) = search.find_best_move(rx, 10, &skipped_moves);

    let is_quiet = !search.board.is_in_check(search.board.active_player()) && pv.moves().len() >= 10 && search.is_quiet_pv(&pv.moves(), search.material_score());

    (scored_move.score() * search.board.active_player() as i32, is_quiet)
}

fn adjust_scores(mut positions: Vec<TestPos>, game_result: i32, _result: String) -> Vec<TestPos> {
    // println!("-----------------------------------------------------------");
    // println!("{}: {}", result, game_result);

    if positions.is_empty() {
        return positions;
    }

    if game_result == 0 {
        positions.last_mut().unwrap().score = 0;

        for (i, pos) in positions.iter_mut().rev().take(32).enumerate() {
            pos.score /= 33 - i as i32;
        }
    }

    for i in 0..positions.len() {
        for j in i..min(i + 16, positions.len() - 1) {
            if game_result == 0 {
                if positions[j].prev_score.abs() < positions[i].score.abs() {
                    positions[i].score = (positions[i].score * 8 + positions[j].prev_score) / 9;
                }
                continue;
            }

            let diff = positions[j].prev_score - positions[i].score;
            if game_result.signum() == diff.signum() {
                positions[i].score = (positions[i].score * 8 + positions[j].prev_score) / 9;
            }
        }
        positions[i].include = positions[i].include && positions[i].score.abs() <= 3000;
    }

    positions
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}
