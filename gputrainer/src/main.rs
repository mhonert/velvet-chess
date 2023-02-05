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

mod sets;
mod layer;

use byteorder::{LittleEndian, WriteBytesExt};
use itertools::Itertools;
use log::{info};
use rand::prelude::{SliceRandom, ThreadRng};
use std::env;
use std::fs::File;
use std::io::{BufWriter, Error};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use env_logger::{Env, Target};
use tch::nn::{Linear, ModuleT, Optimizer, VarStore};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Reduction, Tensor};
use traincommon::idsource::IDSource;
use traincommon::sets::{convert_sets, read_samples};
use velvet::nn::{HL_HALF_NODES, HL_NODES, SCORE_SCALE};
pub use velvet::nn::INPUTS;
use crate::layer::{InputLayer, input_layer};
use crate::sets::{DataSample, GpuDataSamples};

const TEST_SET_SIZE: usize = 400_000;

const BATCH_SIZE: i64 = 32000;
const SETS_PER_BATCH: usize = 4;

const INIT_LR: f64 = 0.001;

const K: f64 = 1.603;
const K_DIV: f64 = K / (400.0 / SCORE_SCALE as f64);

const MIN_TRAINING_SET_ID: usize = 11;
const FEN_TRAINING_SET_PATH: &str = "./data/train_fen/";
const LZ4_TRAINING_SET_PATH: &str = "./data/train_lz4";
const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";
const SAMPLES_PER_SET: usize = 200_000;

trait Mode {
    fn print_info(&self);
    fn forward_t(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor, train: bool) -> Tensor;
    fn save_raw(&self, id: &str, vs: &VarStore);
}

struct EvalNet {
    inputs: InputLayer,
    hidden: Linear,
}

impl EvalNet {
    fn new(vs: &nn::Path) -> Self {
        EvalNet{
            inputs: input_layer(vs / "input", INPUTS as i64, HL_HALF_NODES as i64),
            hidden: nn::linear(vs / "output", HL_NODES as i64, 1, Default::default()),
        }
    }
}

impl Mode for EvalNet {
    fn print_info(&self) {
        println!("Training neural network: {} x 2x{} x 1", INPUTS, HL_NODES);
    }

    fn forward_t(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor, train: bool) -> Tensor {
        let hidden_acc = self.inputs.forward(white_xs, black_xs, stms).clamp_min(0.0f64);

        self.hidden.forward_t(&hidden_acc, train)
            .multiply_scalar(K_DIV)
            .sigmoid()
    }

    fn save_raw(&self, id: &str, vs: &VarStore) {
        let vars = vs.variables();

        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8('V' as i8).unwrap();

        let hidden_layers = 1;
        writer.write_i8(hidden_layers).unwrap(); // Number of hidden layers

        let input_bias = Tensor::cat(&[vars.get("input.own_bias").unwrap(), vars.get("input.opp_bias").unwrap()], 0);
        write_layer(&mut writer, vars.get("input.weight").unwrap(), &input_bias, 1.0)
            .expect("Could not write layer");

        for i in 1..hidden_layers {
            write_layer(
                &mut writer,
                vars.get(format!("hidden{}.weight", i).as_str()).unwrap(),
                vars.get(format!("hidden{}.bias", i).as_str()).unwrap(),
                1.0,
            )
            .expect("Could not write layer");
        }

        write_layer(&mut writer, vars.get("output.weight").unwrap(), vars.get("output.bias").unwrap(), 1.0)
            .expect("Could not write layer");
    }
}

fn config_optimizer(vs: &VarStore, initial_lr: f64) -> Optimizer {
    let mut opt = nn::AdamW::default().build(vs, initial_lr).unwrap();
    opt.set_weight_decay(0.01);

    opt
}

pub fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).target(Target::Stdout).init();

    if env::args().len() <= 2 {
        info!("Usage: trainer [psq|eval] [reader thread count]");
        return;
    }

    let device = Device::cuda_if_available();
    let vs = VarStore::new(device);

    let mut best_net = Vec::new();

    let net = Box::new(EvalNet::new(&vs.root()));

    let data_reader_threads = match usize::from_str(env::args().nth(2).unwrap().as_str()) {
        Ok(count) => count.max(1),
        Err(_) => {
            info!("Usage: trainer [psq|eval] [reader thread count]");
            return;
        }
    };

    net.print_info();

    let available_parallelism = thread::available_parallelism().unwrap().get();
    info!("Scanning available training sets ...");
    let max_training_set_id = convert_sets(
        available_parallelism,
        "training",
        FEN_TRAINING_SET_PATH,
        LZ4_TRAINING_SET_PATH,
        MIN_TRAINING_SET_ID,
    );

    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    info!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * SAMPLES_PER_SET
    );

    info!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(available_parallelism, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1);

    info!("Reading test sets ...");
    let mut test_set = GpuDataSamples(vec![DataSample::default(); SAMPLES_PER_SET * max_test_set_id]);
    let mut start = 0;
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str(), false, &[]);
        start += SAMPLES_PER_SET;
    }
    let mut rng = ThreadRng::default();

    let id_source =
        Arc::new(RwLock::new(IDSource::new(&mut rng, MIN_TRAINING_SET_ID, max_training_set_id, SETS_PER_BATCH)));

    info!("Using {} random samples out of {} for validation", TEST_SET_SIZE, test_set.0.len());

    let stop_readers = Arc::new(AtomicBool::default());

    let (tx, rx) = mpsc::sync_channel::<(Tensor, Tensor, Tensor, Tensor)>(data_reader_threads * 2);
    spawn_data_reader_threads(data_reader_threads, device, &tx, &id_source, &stop_readers);

    let mut best_error = f64::MAX;

    let mut epoch = 1;

    let mut epoch_changed = false;
    let mut epoch_start = Instant::now();

    let samples_per_epoch = training_set_count * SAMPLES_PER_SET;
    let mut curr_epoch_samples = 0;

    let mut lr = INIT_LR;
    let mut lr_reduction = 0.75;
    let mut no_improvements = true;
    let mut already_reduced = false;

    let mut init_test_set = true;

    let mut opt = config_optimizer(&vs, lr);

    let start_time = Instant::now();

    let mut epoch_error_start = 1.0;

    let train_start = Instant::now();
    opt.set_lr(lr);

    let mut test_wpov_xs = Tensor::empty(&[], (Kind::Float, device));
    let mut test_bpov_xs = Tensor::empty(&[], (Kind::Float, device));
    let mut test_stms = Tensor::empty(&[], (Kind::Float, device));
    let mut test_ys = Tensor::empty(&[], (Kind::Float, device));

    loop {

        if init_test_set {
            test_set.0.shuffle(&mut rng);

            let (new_test_wpov_xs, new_test_bpov_xs, new_test_stms, new_test_ys) = to_sparse_tensors(&test_set.0[0..TEST_SET_SIZE], device);
            test_wpov_xs = new_test_wpov_xs;
            test_bpov_xs = new_test_bpov_xs;
            test_stms = new_test_stms;
            test_ys = new_test_ys;

            test_ys = test_ys.multiply_scalar(K_DIV).sigmoid();

            epoch_error_start = tch::no_grad(|| {
                f64::from(
                    &net.forward_t(&test_wpov_xs, &test_bpov_xs, &test_stms, false)
                        .mse_loss(&test_ys, Reduction::Mean),
                )
            });

            best_error = epoch_error_start;

            init_test_set = false;
        }

        let mut train_loss = 0.0;
        let mut batch_count = 0;

        let mut train_count = 0;

        let iter_start = Instant::now();
        while train_count < SAMPLES_PER_SET * 16 {
            let (wpov_data_batch, bpov_data_batch, stm_data_batch, label_batch) = rx.recv().expect("Could not receive test positions");
            train_count += label_batch.numel();

            curr_epoch_samples += label_batch.numel();

            let loss = net.forward_t(&wpov_data_batch, &bpov_data_batch, &stm_data_batch, true)
                .mse_loss(&(label_batch.multiply_scalar(K_DIV).sigmoid()), Reduction::Mean);

            opt.backward_step(&loss);
            tch::no_grad(|| {
                vs.variables_.lock().map(|mut v| {
                    const BOUND: f64 = 127.0 / 128.0;

                    let t = v.named_variables.get_mut("output.weight").unwrap();
                    *t = t.clamp_(-BOUND, BOUND);

                    let t = v.named_variables.get_mut("output.bias").unwrap();
                    *t = t.clamp_(-BOUND, BOUND);
                }).expect("weights clipped");
            });

            train_loss += f64::from(&loss);

            batch_count += 1;
        }

        if curr_epoch_samples >= samples_per_epoch {
            epoch_changed = true;
        }

        let test_error = tch::no_grad(|| {
            f64::from(
                &net.forward_t(&test_wpov_xs, &test_bpov_xs, &test_stms, false)
                    .mse_loss(&test_ys, Reduction::Mean),
            )
        });

        train_loss /= batch_count as f64;

        if test_error <= best_error {
            best_error = test_error;
            net.save_raw(&format!("{}", epoch), &vs);
            no_improvements = false;
            vs.save_to_stream(&mut best_net).expect("could not write net");
        }

        let samples_per_sec =
            (train_count as f64 * 1000.0) / Instant::now().duration_since(iter_start).as_millis() as f64;


        info!(
            "Epoch: {:02} [ {:2}% ] / LR: {:1.8}, BS: {} - Best err: {:1.10} / Train err: {:1.10} / Acc.: {:1.10} / {:.0} samples/sec / Elapsed: {:3.1}m",
            epoch, (curr_epoch_samples * 100 / samples_per_epoch).min(100), lr, BATCH_SIZE, best_error, train_loss, epoch_error_start - best_error, samples_per_sec,
            train_start.elapsed().as_secs_f64() / 60.0
        );

        if epoch_changed {
            let epoch_duration = Instant::now().duration_since(epoch_start);
            let samples_per_sec = (curr_epoch_samples as f64 * 1000.0) / epoch_duration.as_millis() as f64;
            info!(
                "- Epoch finished: {:.2} samples/s (epoch duration: {} seconds)",
                samples_per_sec,
                epoch_duration.as_secs()
            );

            if lr < 0.0000001 && no_improvements {
                break;
            }

            if no_improvements || already_reduced || epoch >= 5 {
                // if !already_reduced {
                //     vs.load_from_stream(Cursor::new(&best_net)).expect("could not read net");
                // }
                lr *= lr_reduction;
                lr_reduction *= 0.9;
                already_reduced = true;
            }

            no_improvements = true;

            opt.set_lr(lr);
            epoch_changed = false;
            epoch_error_start = best_error;
            curr_epoch_samples = 0;
            epoch_start = Instant::now();
            epoch += 1;
            init_test_set = true;
        }
    }

    info!("- Stopping reader threads ...");
    stop_readers.store(true, Ordering::Release);
    thread::sleep(Duration::from_millis(50));
    while !matches!(rx.try_recv(), Err(_)) {
        thread::sleep(Duration::from_millis(10));
    }

    info!("- Training finished in {:.1} minutes", Instant::now().duration_since(start_time).as_secs() as f64 / 60.0);
}

fn to_sparse_tensors(samples: &[DataSample], device: Device) -> (Tensor, Tensor, Tensor, Tensor) {
    let mut ys = Vec::with_capacity(samples.len());
    let mut stms = Vec::with_capacity(samples.len());

    let mut wpov_indices = Vec::with_capacity(1_200_000);
    let mut bpov_indices = Vec::with_capacity(1_200_000);

    for (j, sample) in samples.iter().enumerate() {
        ys.push(sample.result);
        stms.push(sample.wtm);
        wpov_indices.resize(wpov_indices.len() + sample.wpov_inputs.len(), j as i64);
        bpov_indices.resize(bpov_indices.len() + sample.bpov_inputs.len(), j as i64);
    }

    for sample in samples.iter() {
        wpov_indices.extend_from_slice(&sample.wpov_inputs);
        bpov_indices.extend_from_slice(&sample.bpov_inputs);
    }

    let wpov_value_count = wpov_indices.len() as i64 / 2;
    let bpov_value_count = bpov_indices.len() as i64 / 2;

    let wpov_indices = Tensor::of_slice(&wpov_indices).to(device).view((2, wpov_value_count));
    let bpov_indices = Tensor::of_slice(&bpov_indices).to(device).view((2, bpov_value_count));
    let wpov_values = Tensor::ones(&[wpov_value_count], (Kind::Float, device));
    let bpov_values = Tensor::ones(&[bpov_value_count], (Kind::Float, device));

    let wpov_xs = Tensor::sparse_coo_tensor_indices_size(
        &wpov_indices,
        &wpov_values,
        &[samples.len() as i64, INPUTS as i64],
        (Kind::Float, device),
    )
    .coalesce();

    let bpov_xs = Tensor::sparse_coo_tensor_indices_size(
        &bpov_indices,
        &bpov_values,
        &[samples.len() as i64, INPUTS as i64],
        (Kind::Float, device),
    )
        .coalesce();

    (wpov_xs,
     bpov_xs,
     Tensor::of_slice(&stms).to(device).view((samples.len() as i64, 1)),
     Tensor::of_slice(&ys).to(device).view((samples.len() as i64, 1)))
}

fn write_layer(writer: &mut BufWriter<File>, weights: &Tensor, biases: &Tensor, scale: f32) -> Result<(), Error> {
    let size = weights.size();

    writer.write_i32::<LittleEndian>(size[0] as i32)?;
    writer.write_i32::<LittleEndian>(size[1] as i32)?;

    let ws: Vec<f32> = weights.into();
    for &weight in ws.iter() {
        writer.write_f32::<LittleEndian>(weight * scale)?;
    }

    let bs: Vec<f32> = biases.into();
    for &bias in bs.iter() {
        writer.write_f32::<LittleEndian>(bias * scale)?;
    }

    Ok(())
}

fn spawn_data_reader_threads(
    threads: usize, device: Device, tx: &SyncSender<(Tensor, Tensor, Tensor, Tensor)>,
    id_source: &Arc<RwLock<IDSource>>,
    is_stopped: &Arc<AtomicBool>,
) {
    for _ in 0..threads {
        let tx2 = tx.clone();
        let thread_id_source = id_source.clone();
        let is_stopped = is_stopped.clone();
        thread::spawn(move || {
            let mut rng = ThreadRng::default();

            let sets_per_batch = thread_id_source.read().unwrap().per_batch_count();
            let batch_size = sets_per_batch * SAMPLES_PER_SET;
            let mut training_samples = GpuDataSamples(vec![DataSample::default(); batch_size]);

            let mut shuffle_base = (0..batch_size).collect_vec();
            shuffle_base.shuffle(&mut rng);

            let mut t_start = training_samples.0.len() - BATCH_SIZE as usize;
            let mut t_end = training_samples.0.len();
            let mut count = 0;
            let mut iter = 0;
            loop {
                if count < BATCH_SIZE as usize {
                    iter += 1;
                    if iter & 15 == 0 {
                        shuffle_base.shuffle(&mut rng);
                    }
                    let sets = thread_id_source.write().unwrap().next_batch(&mut rng);


                    let mut start = 0;
                    for &id in sets.iter() {
                        read_samples(
                            &mut training_samples,
                            start,
                            format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id).as_str(),
                            true,
                            &shuffle_base,
                        );
                        start += SAMPLES_PER_SET;
                    }

                    training_samples.0.shuffle(&mut rng);
                    count = start;

                    t_start = training_samples.0.len() - BATCH_SIZE as usize;
                    t_end = training_samples.0.len();
                }

                let (wpov_xs, bpov_xs, stms, ys) = to_sparse_tensors(&training_samples.0[t_start..t_end], device);
                tx2.send((wpov_xs, bpov_xs, stms, ys)).expect("Could not send training batch");
                if is_stopped.load(Ordering::Acquire) {
                    return;
                }
                t_start -= BATCH_SIZE as usize;
                t_end -= BATCH_SIZE as usize;
                count -= BATCH_SIZE as usize;
            }
        });
    }
}

