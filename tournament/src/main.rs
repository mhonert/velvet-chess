/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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

use std::env::args;
use std::sync::{Arc, RwLock};
use core_affinity::CoreId;
use thread_priority::*;
use selfplay::openings::OpeningBook;
use selfplay::pentanomial::PentanomialCount;
use velvet::fen::{create_from_fen, read_fen, START_POS};
use velvet::init::init;
use velvet::nn::init_nn_params;
use crate::tournament::TournamentState;

const PAIRS: usize = 8;

const SPRT_ELO_BOUNDS: (f64, f64) = (0.0, 3.0);

const SPRT_ALPHA: f64 = 0.05;
const SPRT_BETA: f64 = 0.05;

fn main() {
    if args().len() < 2 {
        println!("Usage: tournament <tournament-file.toml>");
        return;
    }

    init();

    let tournament_file = args().nth(1).expect("No tournament file parameter provided");
    let tournament_config = config::read_tournament_config(tournament_file).expect("Could not read tournament configuration");
    
    let engine_config = config::read_engine_configs(tournament_config.engines).expect("Could not read engine configurations");
    let openings = Arc::new(OpeningBook::new(&tournament_config.book));
    
    println!("Velvet Tournament Tool");
    println!(" - Starting tournament with TC {}+{} ", tournament_config.tc, tournament_config.inc);

    let mut reserved_core_ids = Vec::new();
    let mut core_ids = core_affinity::get_core_ids().expect("Could not retrieve CPU core IDs");

    core_ids.sort();

    // Keep one "full" CPU core free
    // Assumes that HT is enabled and that there are two logical CPU cores per physical CPU core
    // (core_affinity library currently does not return the CPU core type)
    reserved_core_ids.push(core_ids.remove(0));
    reserved_core_ids.push(core_ids.remove(core_ids.len() / 2 + 1));

    println!(" - Using {} CPU cores", core_ids.len());

    // assert!(core_affinity::set_for_current(reserved_core_ids[0]), "could not set CPU core affinity");

    let state = TournamentState::new(tournament_config, engine_config).expect("Could not create tournament state");

    let handles: Vec<_> = core_ids.into_iter().map(|id| {
        let thread_state = state.clone();
        let thread_openings = openings.clone();

        ThreadBuilder::default()
            .name(format!("Worker {:?}", id))
            .priority(ThreadPriority::Max)
            .spawn(move |result| {
                if let Err(e) = result {
                    eprintln!("Could not set thread priority for worker thread running on {:?}: {}", id, e);
                }
                run_thread(id, thread_state.clone(), thread_openings.clone(), time, inc);
            })
            .expect("could not spawn thread")
    }).collect();

    for handle in handles.into_iter() {
        handle.join().expect("could not join threads");
    }
}

fn run_thread(id: CoreId, state: Arc<TournamentState>, openings: Arc<OpeningBook>, time: i32, inc: i32) {
    if !core_affinity::set_for_current(id) {
        eprintln!("Could not set CPU core affinity for worker thread running on {:?}", id);
    }

    let mut board = create_from_fen(START_POS);

    while !state.stopped() {
        let mut p = PentanomialCount::default();
        let mut time_losses = 0;

        // for _ in 0..PAIRS {
        //     let opening = openings.get_random();
        //     let opponent = state.next_opponent();
        //
        //     read_fen(&mut board, &opening).expect("Could not read FEN");
        //     // let (first_result, add_time_losses) = play_match(&mut board, &mut [&mut engine_a, &mut engine_b], time, inc);
        //     // if state.stopped() {
        //     //     return;
        //     // }
        //     // time_losses += add_time_losses;
        //     //
        //     // read_fen(&mut board, &opening).expect("Could not read FEN");
        //     // let (second_result, add_time_losses) = play_match(&mut board, &mut [&mut engine_b, &mut engine_a], time, inc);
        //     // if state.stopped() {
        //     //     return;
        //     // }
        //     // time_losses += add_time_losses;
        //
        //     // p.add((first_result, second_result.invert()));
        //     return;
        // }

        // let avg_depth = (engine_a.avg_depth() + engine_b.avg_depth()) / 2;
        // state.update(&p, avg_depth, time_losses);
    }
}
