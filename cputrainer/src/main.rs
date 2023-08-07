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

mod net;
mod sets;

use crate::net::{Network, NetworkStats, A32};
use crossbeam_channel::{bounded, Receiver, Sender};
use env_logger::{Env, Target};
use itertools::Itertools;
use log::{error, info};
use rand::prelude::{SliceRandom};
use rand::rngs::ThreadRng;
use std::env;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::thread;
use traincommon::sets::{convert_sets, read_samples, SAMPLES_PER_SET};
pub use velvet::nn::INPUTS;
use velvet::nn::{HL1_HALF_NODES};
use crate::sets::{CpuDataSamples, DataSample};

const BATCH_SIZE: usize = 4000;

const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";

const USAGE: &str = "Usage: cputrainer [quantize] [thread count]";

#[derive(Copy, Clone)]
enum Command {
    Test(Scope),
    Stats,
}

#[derive(Debug)]
enum Result {
    Test(f64, u32),
    Stats(NetworkStats),
}

#[derive(Copy, Clone)]
enum Scope {
    Full,
}

pub fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).target(Target::Stdout).init();

    if env::args().len() <= 2 {
        info!("{}", USAGE);
        return;
    }

    let thread_count = match usize::from_str(env::args().nth(2).unwrap().as_str()) {
        Ok(count) => count.max(1),
        Err(_) => {
            error!("{}", USAGE);
            return;
        }
    };

    match env::args().nth(1).unwrap().as_str() {
        "quantize" => {
            main_quantize(thread_count);
        }

        _ => {
            info!("{}: invalid commmand", USAGE);
        }
    };
}

pub fn main_quantize(thread_count: usize) {
    info!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(thread_count, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1, false, false);

    info!("Reading test sets ...");
    let mut test_set = CpuDataSamples(vec![DataSample::default(); SAMPLES_PER_SET * max_test_set_id]);
    let mut start = 0;
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str());
        start += SAMPLES_PER_SET;
    }
    let mut rng = ThreadRng::default();
    test_set.0.shuffle(&mut rng);
    let full_test_set = Arc::new(test_set);

    let (to_tx, to_rx) = bounded(thread_count);
    let (from_tx, from_rx) = bounded(thread_count);

    let net = Arc::new(RwLock::new(Network::new()));

    let batch_size = BATCH_SIZE;
    let samples_per_thread = batch_size / thread_count;
    info!("Samples per thread: {}", samples_per_thread);

    spawn_training_threads(
        thread_count,
        max_test_set_id,
        &from_tx,
        &to_rx,
        &net,
        &full_test_set,
    );

    net.write().unwrap().init_from_raw_file("./data/nets/velvet_base.nn");


    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let error = error_sum / error_count as f64;
    info!("Initial error         : {:1.10}", error);
    net.write().unwrap().zero_unused_weights();

    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let error = error_sum / error_count as f64;
    info!("Initial error (0)     : {:1.10}", error);

    let stats = stats_net(thread_count, &to_tx, &from_rx);
    info!("Network stats         : {:?}", stats);
    let qnn_file = "./data/nets/velvet_final.qnn";
    net.write().map(|n| n.save_quantized(&stats, qnn_file)).expect("could not quantize net");

    net.write().unwrap().init_from_qnn_file(&qnn_file);
    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let error = error_sum / error_count as f64;
    info!("Error of quantized net: {:1.10}", error);
}

fn test_net(command: Command, thread_count: usize, to_tx: &Sender<Command>, from_rx: &Receiver<Result>) -> (f64, u32) {
    (0..thread_count).for_each(|_| to_tx.send(command).expect("could not send Test command"));
    let mut errors = 0.;
    let mut counts = 0;
    for (error, count) in (0..thread_count).map(|_| match from_rx.recv().expect("could not receive Test result") {
        Result::Test(error, count) => (error, count),
        unexpected => panic!("Unexpected result: {:?}", unexpected),
    }) {
        errors += error;
        counts += count;
    }

    (errors, counts)
}

fn stats_net(thread_count: usize, to_tx: &Sender<Command>, from_rx: &Receiver<Result>) -> NetworkStats {
    (0..thread_count).for_each(|_| to_tx.send(Command::Stats).expect("could not send Test command"));
    let mut stats = NetworkStats::default();
    for thread_stats in (0..thread_count).map(|_| match from_rx.recv().expect("could not receive stats result") {
        Result::Stats(s) => s,
        unexpected => panic!("Unexpected result: {:?}", unexpected),
    }) {
        stats = stats.max(&thread_stats);
    }

    stats
}

fn spawn_training_threads(
    threads: usize, max_test_set_id: usize, tx: &Sender<Result>, rx: &Receiver<Command>,
    base_net: &Arc<RwLock<Box<Network>>> , full_test_set: &Arc<CpuDataSamples>,
) {
    for i in 0..threads {
        let thread_tx = tx.clone();
        let thread_rx = rx.clone();
        let thread_base_net = base_net.clone();

        let thread_full_test_set = full_test_set.clone();
        thread::Builder::new()
            .stack_size(1048576 * 32)
            .spawn(move || {
                train(
                    i,
                    threads,
                    max_test_set_id,
                    thread_tx,
                    thread_rx,
                    thread_base_net,
                    thread_full_test_set,
                )
            })
            .expect("Could not spawn training thread");
    }
}

fn train(
    thread_id: usize, thread_count: usize, max_test_set_id: usize, tx: Sender<Result>, rx: Receiver<Command>,
    base_net: Arc<RwLock<Box<Network>>>, full_test_set: Arc<CpuDataSamples>,
) {
    let thread_test_set_size = full_test_set.0.len() / thread_count;

    let thread_test_set =
        Vec::from(&full_test_set.0[(thread_id * thread_test_set_size)..((thread_id + 1) * thread_test_set_size)]);

    let thread_test_batch_size = thread_test_set_size / (max_test_set_id + 1);

    let mut white_ihidden_values = A32([0f32; HL1_HALF_NODES]);
    let mut black_ihidden_values = A32([0f32; HL1_HALF_NODES]);

    loop {
        match rx.recv().expect("Could not read training command") {
            Command::Test(scope) => {
                let set_ids = match scope {
                    Scope::Full => (0..max_test_set_id).into_iter().collect_vec(),
                };

                let (curr_error, count) = base_net
                    .read()
                    .map(|n| {
                        let mut curr_error = 0.;
                        let mut count = 0;
                        for &i in set_ids.iter() {
                            for sample in
                                thread_test_set.iter().skip(i * thread_test_batch_size).take(thread_test_batch_size)
                            {
                                curr_error += n.test(sample, &mut white_ihidden_values, &mut black_ihidden_values);
                                count += 1;
                            }
                        }
                        (curr_error, count)
                    })
                    .expect("Could not lock for test");

                tx.send(Result::Test(curr_error, count)).expect("Could not send Test result");
                continue;
            }

            Command::Stats => {
                let set_ids = (0..max_test_set_id).into_iter().collect_vec();

                let stats = base_net
                    .read()
                    .map(|n| {
                        let mut curr_stats = NetworkStats::default();
                        for &i in set_ids.iter() {
                            for sample in
                                thread_test_set.iter().skip(i * thread_test_batch_size).take(thread_test_batch_size)
                            {
                                curr_stats = curr_stats.max(&n.stats(
                                    sample,
                                    &mut white_ihidden_values,
                                    &mut black_ihidden_values,
                                ));
                            }
                        }
                        curr_stats
                    })
                    .expect("Could not lock for test");

                tx.send(Result::Stats(stats)).expect("Could not send Stats result");
                continue;
            }
        };
    }
}