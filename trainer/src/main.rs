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

use crate::net::Network;
use crate::sets::{convert_sets, read_samples, DataSample, POS_PER_SET};
use crossbeam_channel::{bounded, Receiver, Sender};
use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::rngs::ThreadRng;
use std::cmp::max;
use std::env;
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;
use velvet::nn::HL_NODES;
pub use velvet::nn::INPUTS;

const SETS_PER_BATCH: usize = 5;
const BATCH_SIZE: usize = SETS_PER_BATCH * POS_PER_SET;
const USE_GAME_RESULT: bool = true;

const TEST_BATCH_SIZE: usize = 400000;

const MIN_TRAINING_SET_ID: usize = 11;

const FEN_TRAINING_SET_PATH: &str = "./data/train_fen/";
const LZ4_TRAINING_SET_PATH: &str = "./data/train_lz4";
const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";

#[derive(Copy, Clone)]
enum Command {
    Train,
    TrainNewInputs([usize; SETS_PER_BATCH]),
    Test(Option<usize>),
    UpdateNet,
}

#[derive(Debug)]
enum Result {
    Train((usize, bool)),
    Test(f64),
    UpdateNet,
}

pub fn main() {
    let usage = "Usage: trainer [train thread count]";

    if env::args().len() <= 1 {
        println!("{}", usage);
        return;
    }

    let thread_count = match usize::from_str(env::args().nth(1).unwrap().as_str()) {
        Ok(count) => max(1, count),
        Err(_) => {
            println!("{}", usage);
            return;
        }
    };

    let mut rng = ThreadRng::default();

    println!("Scanning available training sets ...");
    let max_training_set_id = convert_sets(
        thread_count,
        "training",
        FEN_TRAINING_SET_PATH,
        LZ4_TRAINING_SET_PATH,
        MIN_TRAINING_SET_ID,
        USE_GAME_RESULT,
    );

    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    println!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * POS_PER_SET
    );

    println!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(thread_count, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1, false);

    println!("Reading test sets ...");
    let mut test_set = Vec::with_capacity(POS_PER_SET * max_test_set_id);
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str());
    }
    test_set.shuffle(&mut rng);
    let full_test_set = Arc::new(test_set);

    let (to_tx, to_rx) = bounded(thread_count);
    let (from_tx, from_rx) = bounded(thread_count);

    let net = Arc::new(RwLock::new(Network::new()));

    let mut thread_nets: Vec<Arc<RwLock<Box<Network>>>> = Vec::new();
    for _ in 0..thread_count {
        thread_nets.push(Arc::new(RwLock::new(Network::new())));
    }

    spawn_training_threads(thread_count, &from_tx, &to_rx, &net, &thread_nets, &full_test_set);

    let mut lr = 0.002;
    let mut lr_reduction = 0.75;
    let mut lr_reduced = false;

    let mut no_improvements = 0;

    let mut best_sub_error = 1.0f64;
    let mut best_full_error = 1.0f64;

    let mut best_net = Network::new();

    let mut epoch = 1;
    let mut epoch_start = Instant::now();

    // Shuffling all data sets is too time consuming.
    // Instead, multiple random data sets will be merged and shuffled
    let mut ids = (MIN_TRAINING_SET_ID..max_training_set_id).collect_vec();
    ids.shuffle(&mut rng);

    let mut thread_requires_input = vec![true; thread_count];

    let mut curr_test_set_id = 0;
    let mut epoch_error_start = 0.0;

    let epoch_set_count = (((max_training_set_id - MIN_TRAINING_SET_ID) / SETS_PER_BATCH) * SETS_PER_BATCH) as i32;
    let mut processed_epoch_set_count: i32 = -(SETS_PER_BATCH as i32);

    println!("Start training network with {} inputs and {} hidden layer nodes", INPUTS, HL_NODES);

    for iteration in 1..usize::MAX {
        let start = Instant::now();
        let mut epoch_changed = false;
        for i in 0..thread_count {
            if thread_requires_input[i] {
                if ids.len() < SETS_PER_BATCH {
                    ids = (MIN_TRAINING_SET_ID..max_training_set_id).collect_vec();
                    ids.shuffle(&mut rng);
                    epoch += 1;
                    epoch_changed = true;
                    processed_epoch_set_count = -4;
                }

                let mut sets = [0; SETS_PER_BATCH];
                sets.fill_with(|| ids.pop().unwrap());
                processed_epoch_set_count += SETS_PER_BATCH as i32;

                to_tx.send(Command::TrainNewInputs(sets)).expect("could not send training input");
            } else {
                to_tx.send(Command::Train).expect("could not send training input");
            }
        }

        let mut sample_counts = 0;
        for i in 0..thread_count {
            match from_rx.recv().expect("could not receive training results") {
                Result::Train((sample_count, requires_input)) => {
                    sample_counts += sample_count;
                    thread_requires_input[i] = requires_input;
                }
                unexpected => panic!("Unexpected result: {:?}", unexpected),
            };
        }

        // Update weights
        net.write().unwrap().reset_training_gradients();
        for thread_net in thread_nets.iter() {
            thread_net
                .read()
                .and_then(|tn| {
                    net.write().unwrap().add_gradients(&tn);
                    Ok(())
                })
                .expect("could not lock thread net");
        }

        net.write().unwrap().update_weights(lr as f32, sample_counts, iteration);
        (0..thread_count).for_each(|_| to_tx.send(Command::UpdateNet).expect("could not send UpdateNet command"));
        (0..thread_count).for_each(|_| {
            from_rx.recv().expect("could not receive UpdateNet result");
        });

        // Test
        let curr_sub_error = test_net(Command::Test(Some(curr_test_set_id)), thread_count, &to_tx, &from_rx);

        if curr_sub_error < best_sub_error {
            curr_test_set_id = next_test_set_id(max_test_set_id, curr_test_set_id);
            best_sub_error = test_net(Command::Test(Some(curr_test_set_id)), thread_count, &to_tx, &from_rx);

            let full_error = test_net(Command::Test(None), thread_count, &to_tx, &from_rx);
            if full_error < best_full_error {
                best_net.copy(net.read().unwrap().as_ref());
                best_full_error = full_error;

                if iteration == 1 {
                    epoch_error_start = best_full_error;
                }
                no_improvements = 0;
            }
        }

        let samples_per_sec = sample_counts as f64 / start.elapsed().as_secs_f64();

        if epoch_changed {
            let epoch_duration = Instant::now().duration_since(epoch_start);
            println!("- Epoch {:02} finished in {} seconds", epoch - 1, epoch_duration.as_secs());
            epoch_error_start = best_full_error;

            let id = format!("{}", epoch - 1);
            best_net.save_raw(id.as_str());
            best_net.save_quantized(id.as_str());

            if lr_reduced || no_improvements > 0 {
                if lr <= 0.000001 {
                    break;
                }

                if lr_reduced || no_improvements > 1 {
                    if no_improvements > 2 {
                        net.write().unwrap().copy_weights(&best_net);
                        net.write().unwrap().reset_momentums();
                    }
                    lr *= lr_reduction;
                    lr_reduction *= 0.75;
                    lr_reduced = true;
                }
                curr_test_set_id = next_test_set_id(max_test_set_id, curr_test_set_id);
                best_sub_error = test_net(Command::Test(Some(curr_test_set_id)), thread_count, &to_tx, &from_rx);
            }
            no_improvements += 1;

            epoch_start = Instant::now();
        }

        println!(
            "Epoch: {:02} [ {:2}% ] / Test Set: {}, LR: {:1.8} - Best error: {:1.10} / Acc.: {:1.10} / samples per second: {:.1}",
            epoch, processed_epoch_set_count * 100 / epoch_set_count, curr_test_set_id, lr, best_full_error, epoch_error_start - best_full_error, samples_per_sec
        );
    }

    best_net.save_raw("final");
    best_net.save_quantized("final");
}

fn next_test_set_id(max_test_set_id: usize, curr_test_set_id: usize) -> usize {
    (curr_test_set_id + 1) % (max_test_set_id * POS_PER_SET / TEST_BATCH_SIZE)
}

fn test_net(command: Command, thread_count: usize, to_tx: &Sender<Command>, from_rx: &Receiver<Result>) -> f64 {
    (0..thread_count).for_each(|_| to_tx.send(command).expect("could not send Test command"));
    (0..thread_count)
        .map(|_| match from_rx.recv().expect("could not receive Test result") {
            Result::Test(error) => error,
            unexpected => panic!("Unexpected result: {:?}", unexpected),
        })
        .sum::<f64>()
        / thread_count as f64
}

fn spawn_training_threads(
    threads: usize, tx: &Sender<Result>, rx: &Receiver<Command>, base_net: &Arc<RwLock<Box<Network>>>,
    nets: &[Arc<RwLock<Box<Network>>>], full_test_set: &Arc<Vec<DataSample>>,
) {
    for i in 0..threads {
        let thread_tx = tx.clone();
        let thread_rx = rx.clone();
        let thread_base_net = base_net.clone();
        let thread_net = nets[i].clone();
        let thread_full_test_set = full_test_set.clone();
        thread::Builder::new()
            .stack_size(1048576 * 16)
            .spawn(move || train(i, threads, thread_tx, thread_rx, thread_base_net, thread_net, thread_full_test_set))
            .expect("Could not spawn training thread");
    }
}

fn train(
    thread_num: usize, thread_count: usize, tx: Sender<Result>, rx: Receiver<Command>,
    base_net: Arc<RwLock<Box<Network>>>, net: Arc<RwLock<Box<Network>>>, full_test_set: Arc<Vec<DataSample>>,
) {
    let thread_test_size = TEST_BATCH_SIZE / thread_count;
    let thread_train_size = BATCH_SIZE / thread_count;

    let thread_full_test_size = full_test_set.len() / thread_count;

    let thread_full_test_set =
        Vec::from(&full_test_set[(thread_num * thread_full_test_size)..((thread_num + 1) * thread_full_test_size)]);
    let mut test_sub_set = Vec::new();

    let mut rng = ThreadRng::default();

    let mut samples = Vec::with_capacity(2 * POS_PER_SET);
    let mut remaining_samples: &mut [DataSample] = &mut samples;

    let mut curr_subset_id = usize::MAX;

    loop {
        match rx.recv().expect("Could not read training command") {
            Command::Train => (),

            Command::TrainNewInputs(ids) => {
                samples.clear();
                for &id in ids.iter() {
                    read_samples(&mut samples, format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id).as_str());
                }
                remaining_samples = &mut samples;
            }

            Command::UpdateNet => {
                net.write().unwrap().copy_weights(base_net.read().unwrap().as_ref());
                tx.send(Result::UpdateNet).expect("Could not send UpdateNet result");
                continue;
            }

            Command::Test(subset) => {
                let test_set = if let Some(subset_id) = subset {
                    if subset_id != curr_subset_id {
                        curr_subset_id = subset_id;
                        let subset_start = curr_subset_id * TEST_BATCH_SIZE;
                        test_sub_set = Vec::from(
                            &full_test_set[(subset_start + thread_num * thread_test_size)
                                ..(subset_start + (thread_num + 1) * thread_test_size)],
                        );
                    }

                    &test_sub_set
                } else {
                    &thread_full_test_set
                };

                let curr_error = net
                    .write()
                    .map(|mut n| {
                        let mut curr_error = 0.;
                        for sample in test_set.iter() {
                            curr_error += n.test(sample) as f64;
                        }
                        curr_error /= test_set.len() as f64;
                        curr_error
                    })
                    .expect("Could not lock for test");

                tx.send(Result::Test(curr_error)).expect("Could not send Test result");
                continue;
            }
        }

        let (batch, remaining_samples2) = remaining_samples.partial_shuffle(&mut rng, thread_train_size as usize);
        remaining_samples = remaining_samples2;

        net.write()
            .and_then(|mut n| {
                n.reset_training_gradients();
                for sample in batch.iter() {
                    n.train(sample);
                }
                Ok(())
            })
            .expect("Could not lock net in training thread");

        let requires_input = remaining_samples.is_empty();
        tx.send(Result::Train((batch.len(), requires_input))).expect("Could not send Train result");
    }
}
