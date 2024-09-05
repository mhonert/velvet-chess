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
mod sprt;

use std::env::args;
use std::sync::{Arc};
use core_affinity::CoreId;
use thread_priority::*;
use selfplay::openings::OpeningBook;
use selfplay::pentanomial::PentanomialCount;
use selfplay::selfplay::{play_match, SearchControl};
use velvet::fen::{create_from_fen, read_fen, START_POS};
use velvet::init::init;
use velvet::nn::init_nn_params;
use crate::sprt::SprtState;

const PAIRS: usize = 8;

const SPRT_ELO_BOUNDS: (f64, f64) = (0.0, 3.0);

const SPRT_ALPHA: f64 = 0.05;
const SPRT_BETA: f64 = 0.05;

fn main() {
    if args().len() < 3 {
        println!("Usage: sprt <book> <base-time> <feature-options...>");
        return;
    }

    init();
    init_nn_params();

    let book_file = args().nth(1).expect("No book file parameter provided");
    let time = args().nth(2).expect("No base time parameter provided").parse::<i32>().expect("Invalid base time parameter") * 1000;
    let inc = time / 100;
    
    let params: Vec<(String, i16, i16)> = args().skip(3).map(|f| {
        let parts: Vec<&str> = f.split(',').collect();
        (parts[0].to_string(), parts[1].parse::<i16>().expect("Invalid parameter value A"), parts[2].parse::<i16>().expect("Invalid parameter value B"))
    }).collect();

    println!("Velvet SPRT Tool");
    println!(" - Testing Elo gain using TC {}+{} for params {:?} ", time as f64 / 1000.0, inc as f64 / 1000.0, params);

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

    let openings = Arc::new(OpeningBook::new(&book_file));

    let state = Arc::new(SprtState::new(SPRT_ELO_BOUNDS.0, SPRT_ELO_BOUNDS.1));

    let handles: Vec<_> = core_ids.into_iter().map(|id| {
        let thread_state = state.clone();
        let thread_openings = openings.clone();
        let thread_params = params.clone();
        
        ThreadBuilder::default()
            .name(format!("Worker {:?}", id))
            .priority(ThreadPriority::Max)
            .spawn(move |result| {
                if let Err(e) = result {
                    eprintln!("Could not set thread priority for worker thread running on {:?}: {}", id, e);
                }
                run_thread(id, thread_state.clone(), thread_openings.clone(), thread_params.clone(), time, inc);
            })
            .expect("could not spawn thread")
    }).collect();

    for handle in handles.into_iter() {
        handle.join().expect("could not join threads");
    }
}

fn run_thread(id: CoreId, state: Arc<SprtState>, openings: Arc<OpeningBook>, params: Vec<(String, i16, i16)>, time: i32, inc: i32) {
    if !core_affinity::set_for_current(id) {
        eprintln!("Could not set CPU core affinity for worker thread running on {:?}", id);
    }

    let mut engine_a = SearchControl::new(time, inc);
    let mut engine_b = SearchControl::new(time, inc);

    let mut board = create_from_fen(START_POS);

    let params_a = params.iter().map(|(name, a, _)| (name.clone(), *a)).collect::<Vec<_>>();
    let params_b = params.iter().map(|(name, _, b)| (name.clone(), *b)).collect::<Vec<_>>();
    
    engine_a.set_params(&params_a);
    engine_b.set_params(&params_b);

    while !state.stopped() {
        engine_a.reset();
        engine_b.reset();

        let mut p = PentanomialCount::default();
        let mut time_losses = 0;

        for _ in 0..PAIRS {
            let opening = openings.get_random();

            read_fen(&mut board, &opening).expect("Could not read FEN");
            let (first_result, add_time_losses) = play_match(&mut board, &mut [&mut engine_a, &mut engine_b], time, inc);
            if state.stopped() {
                return;
            }
            time_losses += add_time_losses;

            read_fen(&mut board, &opening).expect("Could not read FEN");
            let (second_result, add_time_losses) = play_match(&mut board, &mut [&mut engine_b, &mut engine_a], time, inc);
            if state.stopped() {
                return;
            }
            time_losses += add_time_losses;

            p.add((first_result, second_result.invert()));
        }

        let avg_depth = (engine_a.avg_depth() + engine_b.avg_depth()) / 2;
        state.update(&p, avg_depth, time_losses);
    }
}
