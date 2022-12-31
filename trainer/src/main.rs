/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use crate::net::{HiddenGradients, InputGradients, NetMomentums, Network, NetworkStats, A32};
use crate::sets::{convert_sets, read_samples, DataSample, SAMPLES_PER_SET};
use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};
use env_logger::{Env, Target};
use itertools::Itertools;
use log::{error, info, trace};
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use rand::RngCore;
use std::env;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;
pub use velvet::nn::INPUTS;
use velvet::nn::{HL_HALF_NODES, HL_NODES};

const TEST_BATCH_SIZE: usize = SAMPLES_PER_SET * 10;

const BATCH_SIZE: usize = 4000;
const INIT_LR: f64 = 0.0004;

const SETS_PER_BATCH: usize = 8;

const MIN_TRAINING_SET_ID: usize = 11;

const FEN_TRAINING_SET_PATH: &str = "./data/train_fen/";
const LZ4_TRAINING_SET_PATH: &str = "./data/train_lz4";
const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";

const USAGE: &str = "Usage: trainer [train|quantize] [thread count]";

#[derive(Copy, Clone)]
enum Command {
    Train(usize),
    Test(Scope),
    Reshuffle,
    Stats,
}

#[derive(Debug)]
enum Result {
    Train(usize, f64),
    Test(f64, u32),
    Stats(NetworkStats),
}

#[derive(Copy, Clone)]
enum Scope {
    Only(usize),
    AllExcept([usize; 2]),
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
        "train" => {
            main_train(thread_count);
        }

        "quantize" => {
            main_quantize(thread_count);
        }

        _ => {
            info!("{}: invalid commmand", USAGE);
        }
    };
}

pub fn main_train(thread_count: usize) {
    info!("Scanning available training sets ...");
    let max_training_set_id =
        convert_sets(thread_count, "training", FEN_TRAINING_SET_PATH, LZ4_TRAINING_SET_PATH, MIN_TRAINING_SET_ID);

    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    info!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * SAMPLES_PER_SET
    );

    info!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(thread_count, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1);

    info!("Reading test sets ...");
    let mut test_set = vec![DataSample::default(); SAMPLES_PER_SET * max_test_set_id];
    let mut start = 0;
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str(), false, &[]);
        start += SAMPLES_PER_SET;
    }
    let mut rng = ThreadRng::default();
    test_set.shuffle(&mut rng);
    let full_test_set = Arc::new(test_set);

    let (to_tx, to_rx) = bounded(thread_count);
    let (from_tx, from_rx) = bounded(thread_count);

    let net = Arc::new(RwLock::new(Network::new()));
    // net.write().unwrap().init_from_base_file(&"./data/nets/velvet_base.nn".to_string());

    let samples_per_thread = (BATCH_SIZE / thread_count) / 2;
    info!("Samples per thread: {}", samples_per_thread);

    let id_source =
        Arc::new(RwLock::new(IDSource::new(&mut rng, MIN_TRAINING_SET_ID, max_training_set_id, SETS_PER_BATCH)));

    let input_gradients = Arc::new(InputGradients::default());
    let mut thread_hidden_gradients = Vec::with_capacity(thread_count);
    for _ in 0..thread_count {
        thread_hidden_gradients.push(Arc::new(RwLock::new(HiddenGradients::default())));
    }

    spawn_training_threads(
        thread_count,
        max_test_set_id,
        &from_tx,
        &to_rx,
        &net,
        &input_gradients,
        &thread_hidden_gradients,
        &full_test_set,
        &id_source,
    );

    let mut hidden_gradients = HiddenGradients::default();

    let mut lr = INIT_LR;
    let mut lr_reduction = 0.75;

    let mut no_improvements = 0;

    let mut best_sub_error = 1.0f64;

    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let mut best_full_error = error_sum / error_count as f64;

    let mut best_net = Network::new();

    let mut epoch = 1;
    let mut epoch_start = Instant::now();

    let mut idle_threads = thread_count;

    let mut curr_test_set_id = 0;
    let mut epoch_error_start = best_full_error;

    let mut curr_epoch_samples = 0;

    info!("Initial error: {:1.10}", epoch_error_start);
    info!("Start training network with {} inputs and {} hidden layer nodes", INPUTS, HL_NODES);

    let samples_per_epoch = (max_training_set_id - MIN_TRAINING_SET_ID) * SAMPLES_PER_SET;

    let train_start = Instant::now();

    let mut momentums = NetMomentums::new();

    for iteration in 1..usize::MAX {
        let start = Instant::now();
        let mut epoch_changed = false;
        let mut curr_iteration_samples = 0;

        let r = if no_improvements == 0 { 100 } else { 10 };
        let mut train_err = 0f64;
        for _ in 0..r {
            let mut curr_batch_samples = 0;
            let mut allocated_count = 0;
            let train_start = Instant::now();
            loop {
                while allocated_count < BATCH_SIZE && idle_threads > 0 {
                    to_tx.send(Command::Train(samples_per_thread)).expect("could not send training input");

                    allocated_count += samples_per_thread;
                    idle_threads -= 1;
                }

                for _ in 0..thread_count {
                    match from_rx.try_recv() {
                        Ok(Result::Train(sample_count, err)) => {
                            curr_batch_samples += sample_count;
                            train_err += err;
                            idle_threads += 1;
                        }
                        Err(e) => {
                            if !matches!(e, TryRecvError::Empty) {
                                panic!("Error receiving training results: {}", e)
                            }
                        }
                        _ => {}
                    };
                }

                if curr_batch_samples >= BATCH_SIZE && idle_threads == thread_count {
                    train_err /= curr_batch_samples as f64;
                    break;
                }
            }
            trace!("Train duration: {}", train_start.elapsed().as_secs_f64() * 1000.0);

            let update_start = Instant::now();
            hidden_gradients.reset();
            for h in thread_hidden_gradients.iter_mut() {
                h.write()
                    .map(|mut g| {
                        hidden_gradients.add_all(&g);
                        g.reset();
                    })
                    .expect("could not update hidden gradients");
            }
            trace!("Update hidden gradients duration: {}", update_start.elapsed().as_secs_f64() * 1000.0);

            let update_start = Instant::now();
            net.write().unwrap().update_weights(
                &input_gradients,
                &hidden_gradients,
                &mut momentums,
                lr,
                iteration,
                curr_batch_samples as u32,
            );
            curr_iteration_samples += curr_batch_samples;

            trace!("Update weights duration: {}", update_start.elapsed().as_secs_f64() * 1000.0);
        }

        // Test
        let fast_test_start = Instant::now();
        let (curr_sub_error, sub_counts) =
            test_net(Command::Test(Scope::Only(curr_test_set_id)), thread_count, &to_tx, &from_rx);
        trace!("Total fast test duration: {}", fast_test_start.elapsed().as_secs_f64() * 1000.0);

        if curr_sub_error / sub_counts as f64 <= best_sub_error {
            let full_test_start = Instant::now();
            let prev_test_set_id = curr_test_set_id;
            curr_test_set_id = next_test_set_id(max_test_set_id, curr_test_set_id);
            let (next_sub_error, next_sub_counts) =
                test_net(Command::Test(Scope::Only(curr_test_set_id)), thread_count, &to_tx, &from_rx);
            best_sub_error = next_sub_error / next_sub_counts as f64;

            let (mut full_error, mut full_counts) = test_net(
                Command::Test(Scope::AllExcept([prev_test_set_id, curr_test_set_id])),
                thread_count,
                &to_tx,
                &from_rx,
            );
            full_error += curr_sub_error + next_sub_error;
            full_counts += sub_counts + next_sub_counts;
            full_error /= full_counts as f64;

            trace!("Total test duration: {}", full_test_start.elapsed().as_secs_f64() * 1000.0);

            if full_error <= best_full_error {
                best_net.copy_weights(net.read().unwrap().as_ref());
                best_full_error = full_error;

                no_improvements = 0;
            }
        }
        trace!("Total iteration duration: {}", start.elapsed().as_secs_f64() * 1000.0);

        let samples_per_sec = curr_iteration_samples as f64 / start.elapsed().as_secs_f64();
        info!(
            "Epoch: {:02} [ {:2}% ] / Test ID: {}, LR: {:1.8}, BS: {} - Best err: {:1.10} / Train err: {:1.10} / Acc.: {:1.10} / {:.0} samples/sec / Elapsed: {:3.1}m",
            epoch, (curr_epoch_samples * 100 / samples_per_epoch).min(100), curr_test_set_id, lr, BATCH_SIZE, best_full_error, train_err, epoch_error_start - best_full_error, samples_per_sec,
            train_start.elapsed().as_secs_f64() / 60.0
        );

        curr_epoch_samples += curr_iteration_samples;
        if curr_epoch_samples >= samples_per_epoch {
            epoch_changed = true;
            epoch += 1;
            curr_epoch_samples = 0;
        }

        if epoch_changed {
            let epoch_duration = Instant::now().duration_since(epoch_start);
            info!("- Epoch {:02} finished in {} seconds", epoch - 1, epoch_duration.as_secs());

            epoch_error_start = best_full_error;

            let id = format!("{}", epoch - 1);
            best_net.save_raw(id.as_str());

            if lr < 0.0000001 && no_improvements > 1 {
                break;
            }

            lr *= lr_reduction;
            lr_reduction *= 0.9;
            if no_improvements > 0 {
                net.write().unwrap().copy_weights(&best_net);
            }

            curr_test_set_id = next_test_set_id(max_test_set_id, curr_test_set_id);
            let (sub_error, count) =
                test_net(Command::Test(Scope::Only(curr_test_set_id)), thread_count, &to_tx, &from_rx);
            best_sub_error = sub_error / count as f64;

            no_improvements += 1;

            epoch_start = Instant::now();
            (0..thread_count).for_each(|_| to_tx.send(Command::Reshuffle).expect("could not send Reshuffle command"));
        }
    }

    best_net.save_raw("final");
}

pub fn main_quantize(thread_count: usize) {
    info!("Scanning available training sets ...");
    let max_training_set_id =
        convert_sets(thread_count, "training", FEN_TRAINING_SET_PATH, LZ4_TRAINING_SET_PATH, MIN_TRAINING_SET_ID);

    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    info!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * SAMPLES_PER_SET
    );

    info!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(thread_count, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1);

    info!("Reading test sets ...");
    let mut test_set = vec![DataSample::default(); SAMPLES_PER_SET * max_test_set_id];
    let mut start = 0;
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str(), false, &[]);
        start += SAMPLES_PER_SET;
    }
    let mut rng = ThreadRng::default();
    test_set.shuffle(&mut rng);
    let full_test_set = Arc::new(test_set);

    let (to_tx, to_rx) = bounded(thread_count);
    let (from_tx, from_rx) = bounded(thread_count);

    let net = Arc::new(RwLock::new(Network::new()));

    let batch_size = BATCH_SIZE;
    let samples_per_thread = batch_size / thread_count;
    info!("Samples per thread: {}", samples_per_thread);

    let id_source =
        Arc::new(RwLock::new(IDSource::new(&mut rng, MIN_TRAINING_SET_ID, max_training_set_id, SETS_PER_BATCH)));

    let input_gradients = Arc::new(InputGradients::default());
    let mut thread_hidden_gradients = Vec::with_capacity(thread_count);
    for _ in 0..thread_count {
        thread_hidden_gradients.push(Arc::new(RwLock::new(HiddenGradients::default())));
    }

    spawn_training_threads(
        thread_count,
        max_test_set_id,
        &from_tx,
        &to_rx,
        &net,
        &input_gradients,
        &thread_hidden_gradients,
        &full_test_set,
        &id_source,
    );

    net.write().unwrap().init_from_base_file(&"./data/nets/velvet_base.nn".to_string());

    let stats = stats_net(thread_count, &to_tx, &from_rx);
    info!("Network stats     : {:?}", stats);

    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let error = error_sum / error_count as f64;
    info!("Initial error     : {:1.10}", error);

    let (input_scale, hidden_scale) = net.write().map(|n| n.save_quantized(&stats)).expect("could not quantize net");

    net.read().map(|n| n.zero_check(input_scale, hidden_scale)).expect("could not quantize net");

    net.write().unwrap().init_from_base_file(&"./data/nets/velvet_base.nn".to_string());
    net.write().map(|mut n| n.quantize(input_scale, hidden_scale)).expect("could not quantize net");

    let (error_sum, error_count) = test_net(Command::Test(Scope::Full), thread_count, &to_tx, &from_rx);
    let error = error_sum / error_count as f64;
    info!("Quantization error: {:1.10}", error);
}

fn next_test_set_id(max_test_set_id: usize, curr_test_set_id: usize) -> usize {
    (curr_test_set_id + 1) % (max_test_set_id * SAMPLES_PER_SET / TEST_BATCH_SIZE)
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
    base_net: &Arc<RwLock<Box<Network>>>, input_gradients: &Arc<InputGradients>,
    hidden_gradients: &Vec<Arc<RwLock<HiddenGradients>>>, full_test_set: &Arc<Vec<DataSample>>,
    id_source: &Arc<RwLock<IDSource>>,
) {
    for i in 0..threads {
        let thread_tx = tx.clone();
        let thread_rx = rx.clone();
        let thread_base_net = base_net.clone();
        let thread_input_gradients = input_gradients.clone();
        let thread_hidden_gradients = hidden_gradients[i].clone();

        let thread_full_test_set = full_test_set.clone();
        let thread_id_source = id_source.clone();
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
                    thread_input_gradients,
                    thread_hidden_gradients,
                    thread_full_test_set,
                    thread_id_source,
                )
            })
            .expect("Could not spawn training thread");
    }
}

fn train(
    thread_id: usize, thread_count: usize, max_test_set_id: usize, tx: Sender<Result>, rx: Receiver<Command>,
    base_net: Arc<RwLock<Box<Network>>>, input_gradients: Arc<InputGradients>,
    hidden_gradients: Arc<RwLock<HiddenGradients>>, full_test_set: Arc<Vec<DataSample>>,
    id_source: Arc<RwLock<IDSource>>,
) {
    let thread_test_set_size = full_test_set.len() / thread_count;

    let thread_test_set =
        Vec::from(&full_test_set[(thread_id * thread_test_set_size)..((thread_id + 1) * thread_test_set_size)]);

    let thread_test_batch_size = thread_test_set_size / (max_test_set_id + 1);

    let mut rng = ThreadRng::default();

    let sets_per_batch = id_source.read().unwrap().batch_id_count;
    let batch_size = sets_per_batch * SAMPLES_PER_SET;
    let mut training_samples = vec![DataSample::default(); batch_size];

    let mut shuffle_base = (0..batch_size).collect_vec();
    shuffle_base.shuffle(&mut rng);

    let mut shuffled_idx = Vec::new();

    let mut white_ihidden_values = A32([0f32; HL_HALF_NODES]);
    let mut black_ihidden_values = A32([0f32; HL_HALF_NODES]);

    loop {
        let samples_per_thread = match rx.recv().expect("Could not read training command") {
            Command::Train(samples) => samples,

            Command::Reshuffle => {
                shuffle_base.shuffle(&mut rng);
                continue;
            }

            Command::Test(scope) => {
                let set_ids = match scope {
                    Scope::Only(subset_id) => Vec::from([subset_id]),

                    Scope::AllExcept(excluded_ids) => {
                        (0..max_test_set_id).into_iter().filter(|id| !excluded_ids.contains(id)).collect_vec()
                    }

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

        if shuffled_idx.len() < samples_per_thread {
            let sets = id_source.write().unwrap().next_batch(&mut rng);

            shuffled_idx = shuffle_base.clone();

            let mut start = 0;
            for &id in sets.iter() {
                read_samples(
                    &mut training_samples,
                    start,
                    format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id).as_str(),
                    true,
                    &shuffled_idx,
                );
                start += SAMPLES_PER_SET;
            }
        }

        let train_err = base_net
            .read()
            .map(|n| {
                hidden_gradients
                    .write()
                    .map(|mut hg| {
                        let mut errors = 0f64;
                        for idx in shuffled_idx.drain(shuffled_idx.len() - samples_per_thread..) {
                            let error = n.train(
                                &training_samples[idx],
                                &input_gradients,
                                &mut hg,
                                &mut white_ihidden_values,
                                &mut black_ihidden_values,
                            );
                            errors += error as f64;
                        }
                        errors
                    })
                    .expect("Could not lock hidden_gradients")
            })
            .expect("Could not read base net in training thread");

        tx.send(Result::Train(samples_per_thread, train_err)).expect("Could not send Train result");
    }
}

struct IDSource {
    ids: Vec<usize>,
    min_id: usize,
    max_id: usize,
    batch_id_count: usize,
}

impl IDSource {
    pub fn new(rng: &mut dyn RngCore, min_id: usize, max_id: usize, batch_id_count: usize) -> Self {
        IDSource { ids: shuffled_ids(rng, min_id, max_id), min_id, max_id, batch_id_count }
    }

    pub fn next_batch(&mut self, rng: &mut dyn RngCore) -> Vec<usize> {
        if self.ids.len() < self.batch_id_count {
            self.ids.append(&mut shuffled_ids(rng, self.min_id, self.max_id));
        }
        self.ids.drain(self.ids.len() - self.batch_id_count..).collect_vec()
    }
}

fn shuffled_ids(rng: &mut dyn RngCore, min: usize, max: usize) -> Vec<usize> {
    let mut ids = (min..=max).collect_vec();
    ids.shuffle(rng);
    ids
}
