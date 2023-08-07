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

use log::{info};
use rand::prelude::{Distribution, SliceRandom, ThreadRng};
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
use rand::distributions::Uniform;
use tch::nn::{Linear, ModuleT, Optimizer, VarStore};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Reduction, Tensor};
use traincommon::idsource::IDSource;
use traincommon::sets::{convert_sets, K, read_samples};
use velvet::nn::{HL1_HALF_NODES, HL1_NODES, MAX_RELU, SCORE_SCALE};
pub use velvet::nn::INPUTS;
use velvet::nn::io::{write_f32, write_u16, write_u8};
use crate::layer::{InputLayer, input_layer};
use crate::sets::{DataSample, GpuDataSamples};

const TEST_BATCH_SIZE: usize = 200_000;

const BATCH_SIZE: i64 = 32000;
const SETS_PER_BATCH: usize = 4;

const INIT_LR: f64 = 0.001;
const INITIAL_PATIENCE: usize = 8;

const VALIDATION_STEP_SIZE: usize = 200_000_000;

const K_DIV: f64 = K / (400.0 / SCORE_SCALE as f64);

const MIN_TRAINING_SET_ID: usize = 1;
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
    output: Linear,
}

impl EvalNet {
    fn new(vs: &nn::Path) -> Self {
        EvalNet{
            inputs: input_layer(vs / "input", INPUTS as i64, HL1_HALF_NODES as i64),
            output: nn::linear(vs / "output", HL1_NODES as i64, 1, Default::default()),
        }
    }
}

impl Mode for EvalNet {
    fn print_info(&self) {
        println!("Training neural network: {} x 2x{} x 1", INPUTS, HL1_HALF_NODES);
    }

    fn forward_t(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor, train: bool) -> Tensor {
        let acc = self.inputs.forward(white_xs, black_xs, stms).clamp(0.0, MAX_RELU as f64).square();

        (self.output.forward_t(&acc, train))
            .multiply_scalar(K_DIV)
            .sigmoid()
    }

    fn save_raw(&self, id: &str, vs: &VarStore) {
        let vars = vs.variables();

        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        write_u8(&mut writer, b'V').unwrap();

        let hidden_layers = 1;
        write_u8(&mut writer, hidden_layers).unwrap(); // Number of hidden layers

        let input_bias = Tensor::cat(&[vars.get("input.own_bias").unwrap(), vars.get("input.opp_bias").unwrap()], 0);
        write_raw(&mut writer, vars.get("input.weight").unwrap(), 1.0).expect("Could not write layer");
        write_raw(&mut writer, &input_bias, 1.0).expect("Could not write layer");

        write_raw(&mut writer, vars.get("output.weight").unwrap(), 1.0).expect("Could not write layer");
        write_raw(&mut writer, vars.get("output.bias").unwrap(), 1.0).expect("Could not write layer");
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
    let mut vs = VarStore::new(device);

    let data_reader_threads = match usize::from_str(env::args().nth(2).unwrap().as_str()) {
        Ok(count) => count.max(1),
        Err(_) => {
            info!("Usage: trainer [psq|eval] [reader thread count]");
            return;
        }
    };


    let available_parallelism = thread::available_parallelism().unwrap().get();
    info!("Scanning available training sets ...");
    let max_training_set_id = convert_sets(
        available_parallelism,
        "training",
        FEN_TRAINING_SET_PATH,
        LZ4_TRAINING_SET_PATH,
        MIN_TRAINING_SET_ID,
        true,
        true
    );

    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    info!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * SAMPLES_PER_SET
    );

    info!("Scanning available test sets ...");
    let max_test_set_id = convert_sets(available_parallelism, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1, false, false);

    info!("Reading test sets ...");
    let mut test_set = GpuDataSamples(vec![DataSample::default(); SAMPLES_PER_SET * max_test_set_id]);
    let mut start = 0;
    for i in 1..=max_test_set_id {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str());
        start += SAMPLES_PER_SET;
    }
    let mut rng = ThreadRng::default();

    let id_source =
        Arc::new(RwLock::new(IDSource::new(&mut rng, MIN_TRAINING_SET_ID, max_training_set_id, SETS_PER_BATCH)));

    info!("Using validation set with {} samples", test_set.0.len());

    let stop_readers = Arc::new(AtomicBool::default());

    let (tx, rx) = mpsc::sync_channel::<(Tensor, Tensor, Tensor, Tensor)>(data_reader_threads * 2);
    spawn_data_reader_threads(data_reader_threads, device, &tx, &id_source, &stop_readers);

    let mut best_error = f64::MAX;

    let mut epoch = 1;

    let net = Box::new(EvalNet::new(&vs.root()));

    net.print_info();

    let mut epoch_changed = false;
    let mut epoch_start = Instant::now();

    let samples_per_epoch = training_set_count * SAMPLES_PER_SET;
    let mut curr_epoch_samples = 0;

    let mut calc_validation_error = true;

    let start_time = Instant::now();

    let train_start = Instant::now();


    let mut lr = INIT_LR;
    let mut opt = config_optimizer(&vs, lr);

    let mut accum = 0.;

    let mut epoch_validation_threshold = VALIDATION_STEP_SIZE;

    let mut missed_improvements = 0;
    let mut best_net_epoch_id = 0;
    let mut total_samples = 0;
    let mut norm_epoch = 1;

    let mut patience = INITIAL_PATIENCE;

    loop {
        if calc_validation_error {
            let validation_start = Instant::now();
            let validation_error = tch::no_grad(|| {
                let mut err = 0f64;

                let mut count = 0;
                for batch in test_set.0.chunks_exact(TEST_BATCH_SIZE) {
                    let (test_wpov_xs, test_bpov_xs, test_stms, mut test_ys) = to_sparse_tensors(batch, device);

                    test_ys = test_ys.multiply_scalar_(K_DIV).sigmoid();
                    err += f64::from(
                        &net.forward_t(&test_wpov_xs, &test_bpov_xs, &test_stms, false)
                            .mse_loss(&test_ys, Reduction::Mean),
                    );
                    count += 1;
                }

                err / count as f64
            });

            calc_validation_error = false;

            if validation_error <= best_error || norm_epoch <= 2 {
                best_net_epoch_id = epoch;
                if best_error != f64::MAX {
                    accum += best_error - validation_error;
                }
                best_error = validation_error;
                net.save_raw(&format!("{}", epoch), &vs);
                vs.save(&format!("data/nets/{}.vs", epoch)).expect("could not save net variables");
                missed_improvements = 0;
            } else {
                missed_improvements += 1;
            }

            if missed_improvements >= patience {
                if patience > 6 {
                    patience /= 2;
                } else if patience > 2 {
                    patience -= 1;
                }
                missed_improvements = 0;
                lr *= 0.4;
                opt.set_lr(lr);

                if lr <= 0.00000166 {
                    break;
                }

                if best_net_epoch_id != 0 {
                    match vs.load(&format!("data/nets/{}.vs", best_net_epoch_id)) {
                        Ok(_) => {}
                        Err(e) => {
                            info!("Could not load previous best net: {}", e);
                        }
                    }
                }
            }

            info!("Calculated validation error in {} seconds: {}", validation_start.elapsed().as_secs_f64(), validation_error);
        }

        let mut train_loss = 0.0;
        let mut batch_count = 0;

        let mut train_count = 0;

        let iter_start = Instant::now();
        while train_count < SAMPLES_PER_SET * 16 {
            let (wpov_data_batch, bpov_data_batch, stm_data_batch, label_batch) = rx.recv().expect("Could not receive test positions");

            let sample_count = label_batch.numel();
            train_count += sample_count;

            curr_epoch_samples += sample_count;
            total_samples += sample_count;

            let loss = net.forward_t(&wpov_data_batch, &bpov_data_batch, &stm_data_batch, true)
                .mse_loss(&(label_batch.multiply_scalar(K_DIV).sigmoid()), Reduction::Mean);

            opt.backward_step(&loss);

            train_loss += f64::from(&loss);

            batch_count += 1;
        }

        if curr_epoch_samples >= samples_per_epoch {
            epoch_changed = true;
        }

        if total_samples >= epoch_validation_threshold {
            epoch_validation_threshold = total_samples + VALIDATION_STEP_SIZE;
            calc_validation_error = true;
            norm_epoch += 1;
        }

        train_loss /= batch_count as f64;

        let samples_per_sec =
            (train_count as f64 * 1000.0) / Instant::now().duration_since(iter_start).as_millis() as f64;

        info!(
            "Norm. Epoch: {:02} / Set Epoch: {:02} [ {:2}% ] / LR: {:1.8}, BS: {} - Best err: {:1.10} / Train err: {:1.10} / Acc.: {:1.10} / {:.0} samples/sec / Elapsed: {:3.1}m",
            norm_epoch, epoch, (curr_epoch_samples * 100 / samples_per_epoch).min(100), lr, BATCH_SIZE, best_error, train_loss, accum, samples_per_sec,
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

            epoch += 1;

            epoch_changed = false;
            curr_epoch_samples = 0;
            epoch_start = Instant::now();

            accum = 0.;
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

fn write_raw(writer: &mut BufWriter<File>, weights: &Tensor, scale: f32) -> Result<(), Error> {
    for size in weights.size() {
        write_u16(writer, size as u16)?;
    }

    let ws: Vec<f32> = weights.into();
    for &weight in ws.iter() {
        write_f32(writer, weight * scale)?;
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

            let distribution = Uniform::new_inclusive::<u8, u8>(0, 3);
            let mut transform_rnd = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                transform_rnd.push(distribution.sample(&mut rng));
            }

            let mut t_start = training_samples.0.len() - BATCH_SIZE as usize;
            let mut t_end = training_samples.0.len();
            let mut count = 0;
            let mut curr_epoch = 1;
            loop {
                if count < BATCH_SIZE as usize {
                    let (epoch, sets) = thread_id_source.write().unwrap().next_batch(&mut rng);
                    if epoch != curr_epoch {
                        curr_epoch = epoch;
                        for v in transform_rnd.iter_mut() {
                            *v = (*v + 1) & 3;
                        }
                    }

                    let mut start = 0;
                    for &id in sets.iter() {
                        read_samples( &mut training_samples, start, format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id).as_str());
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

