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

use std::env::args;
use std::sync::{Arc, RwLock};
use core_affinity::CoreId;
use rand::prelude::ThreadRng;
use rand::Rng;
use thread_priority::{ThreadBuilder, ThreadPriority};
use selfplay::openings::OpeningBook;
use selfplay::pentanomial::PentanomialCount;
use selfplay::selfplay::{play_match, SearchControl};
use velvet::fen::{create_from_fen, read_fen, START_POS};
use velvet::init::init;
use velvet::nn::init_nn_params;

const TIME: i32 = 4000;
const INC: i32 = TIME / 100;

const PAIRS: usize = 4;

fn main() {
    init();
    init_nn_params();
    let book_file = args().nth(1).expect("No book file parameter provided");

    println!("Velvetune");

    let mut reserved_core_ids = Vec::new();
    let mut core_ids = core_affinity::get_core_ids().expect("Could not retrieve CPU core IDs");

    core_ids.sort();

    // Keep one "full" CPU core free
    // Assumes that the HT is enabled and that there are two logical CPU cores per physical CPU core
    // (core_affinity library currently does not return the CPU core type)
    reserved_core_ids.push(core_ids.remove(0));
    reserved_core_ids.push(core_ids.remove(core_ids.len() / 2 + 1));

    println!("Using {} CPU cores", core_ids.len());

    let openings = Arc::new(OpeningBook::new(&book_file));

    let mut tunable_params = ParamSet::default();
    let lr = 0.001;
    tunable_params.add("fp_base_margin".to_string(), 17.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("fp_margin_multiplier".to_string(), 22.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("razor_margin_multiplier".to_string(), 200.0, 0.0, 500.0, 25.0, lr);
    tunable_params.add("rfp_base_margin_improving".to_string(),  19.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("rfp_margin_multiplier_improving".to_string(),22.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("rfp_base_margin_not_improving".to_string(), 23.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("rfp_margin_multiplier_not_improving".to_string(),  28.0, 0.0, 100.0, 5.0, lr);
    tunable_params.add("lmr_base".to_string(), 256.0, 0.0, 512.0, 32.0, lr);
    tunable_params.add("lmr_divider".to_string(), 1024.0, 256.0, 4096.0, 64.0, lr);
    tunable_params.add("nmp_base".to_string(), 768.0, 32.0, 2048.0, 32.0, lr);
    tunable_params.add("nmp_divider".to_string(), 768.0, 32.0, 2048.0, 64.0, lr);

    let tuner_state = TunerState::new(tunable_params);

    let handles: Vec<_> = core_ids.into_iter().map(|id| {
        let thread_tuner_state = tuner_state.clone();
        let thread_openings = openings.clone();

        ThreadBuilder::default()
            .name(format!("Worker {:?}", id))
            .priority(ThreadPriority::Max)
            .spawn(move |result| {
                if let Err(e) = result {
                    eprintln!("Could not set thread priority for worker thread running on {:?}: {}", id, e);
                }
                run_thread(id, thread_tuner_state, thread_openings);
            })
            .expect("could not spawn thread")
    }).collect();

    for handle in handles.into_iter() {
        handle.join().expect("could not join threads");
    }
}

fn run_thread(id: CoreId, thread_tuner_state: Arc<RwLock<TunerState>>, thread_openings: Arc<OpeningBook>) {
    assert!(core_affinity::set_for_current(id), "could not set CPU core affinity for worker thread");

    let params = thread_tuner_state.read().expect("could not read-lock tuner state").clone_params();

    let mut base_params_a = params.clone();
    let mut base_params_b = params.clone();

    let mut engine_a = SearchControl::new(TIME, INC);
    let mut engine_b = SearchControl::new(TIME, INC);

    let mut board = create_from_fen(START_POS);
    
    println!("Running thread on core {:?}", id);

    loop {
        let mut p = PentanomialCount::default();
        engine_a.reset();
        engine_b.reset();
        pertubate(&mut base_params_a, &mut base_params_b);

        let params_a = base_params_a.0.iter().map(|p| (p.name.clone(), p.value.round() as i16)).collect::<Vec<_>>();
        let params_b = base_params_b.0.iter().map(|p| (p.name.clone(), p.value.round() as i16)).collect::<Vec<_>>();

        engine_a.set_params(&params_a);
        engine_b.set_params(&params_b);

        for _ in 0..PAIRS {
            let opening = thread_openings.get_random();

            read_fen(&mut board, &opening).expect("Could not read FEN");
            let (first_result, _) = play_match(&mut board, &mut [&mut engine_a, &mut engine_b], TIME, INC);

            read_fen(&mut board, &opening).expect("Could not read FEN");
            let (second_result, _) = play_match(&mut board, &mut [&mut engine_b, &mut engine_a],TIME, INC);

            p.add((first_result, second_result.invert()));
        }

        let gradient = p.gradient(); 
        println!(" - Gradient = {} / {:?}", gradient, p);

        // let avg_depth = (engine_a.avg_depth() + engine_b.avg_depth()) / 2;
        // let min_depth = engine_a.min_depth.min(engine_b.min_depth);
        // println!("Avg. depth: {} / Min. depth: {}", avg_depth, min_depth);
        thread_tuner_state.write().expect("Could not write lock tuner state").update(&base_params_a, &base_params_b, gradient, PAIRS * 2);
        let params = thread_tuner_state.read().expect("could not read-lock tuner state").clone_params();
        base_params_a = params.clone();
        base_params_b = params.clone();
    }
}

pub struct TunerState {
    params: ParamSet,
    games: usize,
    updates: usize,
}

impl TunerState {
    pub fn new(params: ParamSet) -> Arc<RwLock<TunerState>> {
        Arc::new(RwLock::new(TunerState{params, games: 0, updates: 0}))
    }

    pub fn clone_params(&self) -> ParamSet {
        self.params.clone()
    }

    pub fn update(&mut self, params_a: &ParamSet, params_b: &ParamSet, gradient: f64, count: usize) {
        self.games += count;
        self.updates += 1;
        println!("{} games / {} parameter updates", self.games, self.updates);
        self.params.update(params_a, params_b, gradient);
        println!("Current parameter values:");
        for param in self.params.0.iter() {
            println!("{} = {:.2}", param.name, param.value);
        }
        println!("__________________________________________________________________________________");
    }
}

#[derive(Clone, Debug, Default)]
pub struct TunableParam {
    name: String,
    value: f64,
    min: f64,
    max: f64,
    step: f64,
    lr: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ParamSet(Vec<TunableParam>);

impl ParamSet {
    pub fn add(&mut self, name: String, value: f64, min: f64, max: f64, step: f64, lr: f64) {
        self.0.push(TunableParam{name, value, min, max, step, lr});
    }

    pub fn update(&mut self, a: &ParamSet, b: &ParamSet, gradient: f64) {
        for ((param, param_a), param_b) in self.0.iter_mut().zip(a.0.iter()).zip(b.0.iter()) {
            let step = param_a.step;
            let delta = (param_a.value - param_b.value).signum() * step * gradient * param_a.lr;
            if delta != 0.0 {
                println!("Update {} from {} to {} (Delta = {}, Gradient = {}, param_a: {}, param_b: {})",
                         param.name, param.value, param.value + delta, delta, gradient, param_a.value, param_b.value);
            }
            
            // TODO: Implement proper SPSA
            param.value += delta;
            param.value = param.value.clamp(param.min, param.max);
        }
    }
}

pub fn pertubate(a: &mut ParamSet, b: &mut ParamSet) {
    let mut rng = ThreadRng::default();
    for (param_a, param_b) in a.0.iter_mut().zip(b.0.iter_mut()) {
        let delta = if rng.gen_bool(0.5) { -1.0 } else { 1.0 } * param_a.step;
        param_a.value += delta;
        param_a.value = param_a.value.clamp(param_a.min, param_a.max);
        param_b.value -= delta;
        param_b.value = param_b.value.clamp(param_b.min, param_b.max);

        // println!("Pertubated parameter {}: {} vs {}", param_a.name, param_a.value, param_b.value);
    }
}
