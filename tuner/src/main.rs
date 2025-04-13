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

const TIME: i32 = 60000;
const INC: i32 = TIME / 100;

const PAIRS: usize = 8;

const MAX_ITERATIONS: usize = 100000;
const A: usize = MAX_ITERATIONS / 10;

const SPSA_A: f64 = 1.0;
const SPSA_C: f64 = 1.0;
const SPSA_ALPHA: f64 = 0.602;
const SPSA_GAMMA: f64 = 0.101;

fn main() {
    init();
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
    tunable_params.add("fp_base_margin".to_string(), 17.0, 0.0, 100.0, 5.0);
    tunable_params.add("fp_margin_multiplier".to_string(), 22.0, 0.0, 100.0, 5.0);
    tunable_params.add("razor_margin_multiplier".to_string(), 200.0, 0.0, 500.0, 25.0);
    tunable_params.add("rfp_base_margin_improving".to_string(),  19.0, 0.0, 100.0, 5.0);
    tunable_params.add("rfp_margin_multiplier_improving".to_string(),22.0, 0.0, 100.0, 5.0);
    tunable_params.add("rfp_base_margin_not_improving".to_string(), 23.0, 0.0, 100.0, 5.0);
    tunable_params.add("rfp_margin_multiplier_not_improving".to_string(),  28.0, 0.0, 100.0, 5.0);
    tunable_params.add("nmp_base".to_string(), 768.0, 32.0, 2048.0, 32.0);
    tunable_params.add("nmp_divider".to_string(), 672.0, 32.0, 2048.0, 64.0);
    tunable_params.add("se_double_ext_margin".to_string(), 4.0, 1.0, 10.0, 1.0);
    tunable_params.add("se_double_ext_limit".to_string(), 12.0, 1.0, 20.0, 1.0);
    tunable_params.add("prob_cut_margin".to_string(), 150.0, 50.0, 300.0, 25.0);
    tunable_params.add("prob_cut_depth".to_string(), 4.0, 1.0, 10.0, 1.0);
    tunable_params.add("lmr_base".to_string(), 256.0, 0.0, 512.0, 32.0);
    tunable_params.add("lmr_divider".to_string(), 1024.0, 256.0, 4096.0, 64.0);
    tunable_params.add("lmp_max_depth".to_string(), 4.0, 1.0, 10.0, 1.0);
    tunable_params.add("lmp_improving_base".to_string(), 3.0, 1.0, 10.0, 1.0);
    tunable_params.add("lmp_improving_multiplier".to_string(), 65.0, 10.0, 100.0, 5.0);
    tunable_params.add("lmp_not_improving_base".to_string(), 2.0, 1.0, 10.0, 1.0);
    tunable_params.add("lmp_not_improving_multiplier".to_string(), 35.0, 10.0, 100.0, 5.0);

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

    let mut engine_a = SearchControl::new(TIME, INC);
    let mut engine_b = SearchControl::new(TIME, INC);

    let mut board = create_from_fen(START_POS);
    
    println!("Running thread on core {:?}", id);

    loop {
        let mut p = PentanomialCount::default();
        engine_a.reset();
        engine_b.reset();
        let step = thread_tuner_state.read().expect("Could not write lock tuner state").start_step();

        let params_a = step.params_a.0.iter().map(|p| (p.name.clone(), p.value.round() as i16)).collect::<Vec<_>>();
        let params_b = step.params_b.0.iter().map(|p| (p.name.clone(), p.value.round() as i16)).collect::<Vec<_>>();

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

        thread_tuner_state.write().expect("Could not write lock tuner state").finish_step(&step, gradient, PAIRS);
    }
}

pub struct StepState {
    params_a: ParamSet,
    params_b: ParamSet,
    a_t: f64,
    c_t: f64,
}

pub struct TunerState {
    params: ParamSet,
    game_pairs: usize,
    updates: usize,
}

impl TunerState {
    pub fn new(params: ParamSet) -> Arc<RwLock<TunerState>> {
        Arc::new(RwLock::new(TunerState{params, game_pairs: 0, updates: 0}))
    }

    pub fn clone_params(&self) -> ParamSet {
        self.params.clone()
    }

    pub fn start_step(&self) -> StepState {
        let mut a = self.params.clone();
        let mut b = self.params.clone();

        let a_t = SPSA_A / (self.game_pairs as f64 + A as f64).powf(SPSA_ALPHA);
        let c_t = SPSA_C / (self.game_pairs as f64 + 1.0).powf(SPSA_GAMMA);

        let mut rng = ThreadRng::default();
        for (param_a, param_b) in a.0.iter_mut().zip(b.0.iter_mut()) {
            let delta = if rng.gen_bool(0.5) { -1.0 } else { 1.0 } * param_a.step * c_t;

            param_a.value += (delta + 0.5).round();
            param_a.value = param_a.value.clamp(param_a.min, param_a.max);
            param_b.value -= (delta - 0.5).round();
            param_b.value = param_b.value.clamp(param_b.min, param_b.max);
        }

        StepState{params_a: a, params_b: b, a_t, c_t}
    }

    pub fn finish_step(&mut self, step: &StepState, gradient: f64, game_pairs: usize) {
        self.game_pairs += game_pairs;
        self.updates += 1;

        self.params.update(&step.params_a, &step.params_b, gradient, step.a_t, step.c_t);
        println!("{} games / {} parameter updates", self.game_pairs, self.updates);
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
}

#[derive(Clone, Debug, Default)]
pub struct ParamSet(Vec<TunableParam>);

impl ParamSet {
    pub fn add(&mut self, name: String, value: f64, min: f64, max: f64, step: f64) {
        self.0.push(TunableParam{name, value, min, max, step});
    }

    pub fn update(&mut self, a: &ParamSet, b: &ParamSet, gradient: f64, a_t: f64, c_t: f64) {
        println!("Gradient = {}, a_t = {}, c_t = {}", gradient, a_t, c_t);
        for ((param, param_a), param_b) in self.0.iter_mut().zip(a.0.iter()).zip(b.0.iter()) {
            let step = param_a.step;
            let delta = (param_a.value - param_b.value).signum() * step * gradient * a_t / c_t;
            if delta != 0.0 {
                println!("Update {} from {} to {} (Delta = {}, Gradient = {}, param_a: {}, param_b: {})",
                         param.name, param.value, param.value + delta, delta, gradient, param_a.value, param_b.value);
            }
            
            param.value += delta;
            param.value = param.value.clamp(param.min, param.max);
        }
    }

    pub fn to_uci_params(&self) -> Vec<(String, i16)> {
        self.0.iter().map(|p| (p.name.clone(), p.value.round() as i16)).collect()
    }
}
