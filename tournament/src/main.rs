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
mod tournament;
mod config;
mod uci_engine;
mod pgn;
mod san;
mod affinity;

use std::collections::HashMap;
use std::env::args;
use std::sync::{Arc};
use anyhow::Context;
use core_affinity::CoreId;
use thread_priority::*;
use selfplay::openings::OpeningBook;
use selfplay::pentanomial::PentanomialCount;
use selfplay::selfplay::{ Outcome};
use velvet::board::Board;
use velvet::fen::{create_from_fen, read_fen, write_fen, START_POS};
use velvet::history_heuristics::{HistoryHeuristics, MoveHistory};
use velvet::init::init;
use velvet::move_gen::{is_valid_move, MoveList};
use velvet::moves::{Move, NO_MOVE};
use velvet::uci_move::UCIMove;
use crate::affinity::pin_thread;
use crate::config::EngineConfig;
use crate::pgn::{PgnGame};
use crate::san::move_to_san;
use crate::tournament::TournamentState;
use crate::uci_engine::UciEngine;

fn main() {
    if args().len() < 2 {
        println!("Usage: tournament <tournament-file.toml>");
        return;
    }

    init();

    let tournament_file = args().nth(1).expect("No tournament file parameter provided");
    let tournament_config = config::read_tournament_config(tournament_file).expect("Could not read tournament configuration");
    
    let mut engine_configs = config::read_engine_configs(&tournament_config.engines).expect("Could not read engine configurations");
    let openings = Arc::new(OpeningBook::new(&tournament_config.book));

    engine_configs.merge_default_options(&tournament_config.default_options);

    // Generate tournament ID from current date and time
    let tournament_id = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();

    // Create sub-directory for tournament data
    let tournament_path = format!("tournament_{}", tournament_id);
    std::fs::create_dir(&tournament_path).expect("Could not create tournament directory");

    println!("Velvet Tournament Tool");
    if let Some(nodes) = tournament_config.nodes {
        println!(" - Starting tournament {} with fixed nodes {}", tournament_id, nodes);
    } else {
        println!(" - Starting tournament {} with TC {}+{}", tournament_id, tournament_config.tc, tournament_config.inc);
    }

    let mut core_ids = core_affinity::get_core_ids().expect("Could not retrieve CPU core IDs");
    core_ids.sort();

    let mut core_pairs = Vec::new();
    for i in 0..core_ids.len() / 2 {
        core_pairs.push((core_ids[i], core_ids[i + core_ids.len() / 2]));
    }

    // Keep at least one "full" CPU core free
    // Assumes that HT is enabled and that there are two logical CPU cores per physical CPU core
    // (core_affinity library currently does not return the CPU core type)
    let mut reserved_core_count = 1;
    reserved_core_count += (core_pairs.len() - reserved_core_count) % tournament_config.engine_threads as usize;
    for _ in 0..reserved_core_count {
        core_pairs.pop();
    }

    println!(" - Using 2x{} logical CPU cores (reserved 2x{} logical CPU cores)", core_pairs.len(), reserved_core_count);

    // assert!(core_affinity::set_for_current(reserved_core_ids[0]), "could not set CPU core affinity");

    let nodes = tournament_config.nodes;
    let time = (tournament_config.tc * 1000.0) as i32;
    let inc = (tournament_config.inc * 1000.0) as i32;
    let state = TournamentState::new(&tournament_config, &engine_configs).expect("Could not create tournament state");
    let challenger = engine_configs.0.get(&tournament_config.challenger).expect("Could not find challenger engine in config").clone();
    
    let available_cores = core_pairs.iter().flat_map(|(a, b)| [*a, *b]).collect::<Vec<_>>();

    let handles: Vec<_> = available_cores.chunks(tournament_config.engine_threads as usize).map(|ids| {
        let ids = ids.to_vec();
        let thread_state = state.clone();
        let thread_openings = openings.clone();
        let thread_challenger = challenger.clone();
        let thread_tournament_path = tournament_path.clone();

        ThreadBuilder::default()
            .name(format!("Worker {:?}", ids))
            .priority(ThreadPriority::Max)
            .spawn(move |result| {
                if let Err(e) = result {
                    eprintln!("Could not set thread priority for worker thread running on {:?}: {}", ids, e);
                }
                run_thread(&ids, thread_state, thread_tournament_path.clone(), thread_openings, thread_challenger, time, inc, nodes);
            })
            .expect("could not spawn thread")
    }).collect();

    for handle in handles.into_iter() {
        handle.join().expect("could not join threads");
    }
}

fn run_thread(ids: &[CoreId], state: Arc<TournamentState>, tournament_path: String, openings: Arc<OpeningBook>, challenger_cfg: EngineConfig, time: i32, inc: i32, nodes: Option<i32>) {
    pin_thread(ids).expect("Could not set CPU core affinity for worker thread");

    let mut challenger = UciEngine::start(&challenger_cfg).unwrap_or_else(|_| panic!("Could not start challenger engine: {}", challenger_cfg.name));
    challenger.init().expect("Could not initialize engine");

    challenger.uci_newgame().expect("Could not start new game in engine");
    challenger.ready().expect("Could not ready engine");

    let pgn_file = format!("./{}/{}.pgn", tournament_path, ids[0].id);
    let mut pgn_writer = pgn::PgnWriter::new(&pgn_file).expect("Could not create PGN writer");

    let mut board = create_from_fen(START_POS);
    let mut engines: HashMap<u32, UciEngine> = HashMap::default();

    let mut controller = MatchController::default();

    while !state.stopped() {
        let mut p = PentanomialCount::default();

        let (opponent_cfg, round) = state.next_opponent().expect("No opponents found");

        let opponent = engines.entry(opponent_cfg.id).or_insert_with(|| {
            let mut engine = UciEngine::start(&opponent_cfg).with_context(|| opponent_cfg.name.clone()).expect("Could not start opponent engine");
            engine.init().expect("Could not initialize engine");

            engine
        });
        
        let opening = openings.get_random();

        read_fen(&mut board, &opening).expect("Could not read FEN");
        let start_move_count = board.fullmove_count();
        let mut pgn1 = PgnGame::new(challenger_cfg.name.clone(), opponent_cfg.name.clone(), time, inc, round, opening.clone(), start_move_count);
        let first_result = controller.play(&mut board, &mut pgn1, &opening, &mut [&mut challenger, opponent], time, inc, nodes).expect("Could not play UCI match");
        pgn1.set_result(first_result);

        read_fen(&mut board, &opening).expect("Could not read FEN");
        let mut pgn2 = PgnGame::new(opponent_cfg.name.clone(), challenger_cfg.name.clone(), time, inc, round, opening.clone(), start_move_count);
        let second_result = controller.play(&mut board, &mut pgn2, &opening, &mut [opponent, &mut challenger], time, inc, nodes).expect("Could not play UCI match");
        pgn2.set_result(second_result);

        pgn_writer.write_game(pgn1).expect("Could not write PGN game");
        pgn_writer.write_game(pgn2).expect("Could not write PGN game");
        pgn_writer.flush().expect("Could not flush PGN writer");

        p.add((first_result.invert(), second_result));

        state.update(opponent_cfg.id, p);
    }
}


#[derive(Default)]
struct MatchController {
    move_gen: MoveList,
    move_history: MoveHistory,
    hh: HistoryHeuristics,
    moves: Vec<Move>,
}

impl MatchController {
    pub fn play(&mut self, board: &mut Board, pgn: &mut PgnGame, opening: &str, engines: &mut [&mut UciEngine; 2], time: i32, inc: i32, nodes: Option<i32>) -> anyhow::Result<Outcome> {
        for engine in engines.iter_mut() {
            engine.uci_newgame().expect("Could not start new game in engine");
            engine.ready().expect("Could not ready engine");
        }

        let mut i = 0;
        let mut remaining_time = [time, time];
        let mut moves = String::with_capacity(2048);

        loop {
            self.generate_available_moves(board);
            let has_legal_moves = !self.moves.is_empty();
            if !has_legal_moves {
                return Ok(if i == 0 { Outcome::Loss } else { Outcome::Win });
            }

            let start = std::time::Instant::now();
            let bm = engines[i].go(opening, remaining_time[0], remaining_time[1], inc, &moves, nodes)?;
            if bm == "0000" {
                println!(" - Engine {} returned bestmove 0000", engines[i].name());
                return Ok(if i == 0 { Outcome::Loss } else { Outcome::Win });
            }
            let duration = start.elapsed().as_millis() as i32;
            
            let time_idx = board.active_player().idx();
            
            let next_remaining_time = remaining_time[time_idx] - duration;
            if next_remaining_time >= 0 {
                remaining_time[time_idx] = next_remaining_time + inc;
            } else if nodes.is_none() {
                println!(" - Engine {} ran out of time by {}ms", engines[i].name(), -next_remaining_time);
                return Ok(if i == 0 { Outcome::Loss } else { Outcome::Win });
            }

            let m = UCIMove::from_uci(&bm).context("Could not parse best move")?.to_move(board);
            if !is_valid_move(board, board.active_player(), m) {
                println!(" - Engine {} played invalid move: {} ({:?}) in position {}", engines[i].name(), bm, m, write_fen(board));
                return Ok(if i == 0 { Outcome::Loss } else { Outcome::Win });
            }
            board.perform_move(m);
            let gives_check = board.is_in_check(board.active_player());
            pgn.add_move(&move_to_san(m, &self.moves, gives_check));

            if board.is_insufficient_material_draw() || board.is_repetition_draw() || board.is_fifty_move_draw() {
                return Ok(Outcome::Draw);
            }

            if !moves.is_empty() {
                moves.push(' ');
            }
            moves.push_str(&bm);

            i = (i + 1) % 2;
        }
    }

    fn generate_available_moves(&mut self, board: &mut Board) {
        self.moves.clear();
        self.move_gen.init(board.active_player(), NO_MOVE, self.move_history);
        while let Some(m) = self.move_gen.next_root_move(&self.hh, board, false) {
            self.moves.push(m);
        }
    }
}
