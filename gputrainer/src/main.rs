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

use log::{error, info, warn};
use rand::prelude::{Distribution, SliceRandom, ThreadRng};
use std::fs::File;
use std::io::{BufReader, BufWriter, Error};
use std::process::exit;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use env_logger::{Env, Target};
use rand::distributions::Uniform;
use tch::nn::{Optimizer, VarStore};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor, IndexOp};
use clap::{Args, Parser, Subcommand};
use traincommon::idsource::IDSource;
use traincommon::sets::{convert_sets, K, read_samples};
use velvet::nn::{BUCKET_SIZE, BUCKETS, HL1_HALF_NODES, HL1_NODES, MAX_RELU, SCORE_SCALE};
pub use velvet::nn::INPUTS;
use velvet::nn::io::{read_f32, read_u16, read_u8, write_f32, write_u16, write_u8};
use velvet::syzygy;
use crate::layer::{InputLayer, input_layer, output_layer, OutputLayer};
use crate::sets::{DataSample, GpuDataSamples};

const TEST_BATCH_SIZE: usize = 200_000;

const BATCH_SIZE: i64 = 32000;
const SETS_PER_BATCH: usize = 8;

const INIT_LR: f64 = 0.001;
const INITIAL_PATIENCE: usize = 8;

const NORM_EPOCH_SIZE: usize = 200_000_000;

const K_DIV: f64 = K / (400.0 / SCORE_SCALE as f64);

const ERR_EXP: f64 = 2.6;

const FEN_TRAINING_SET_PATH: &str = "./data/train_fen/";
const LZ4_TRAINING_SET_PATH: &str = "./data/train_lz4";
const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";
const SAMPLES_PER_SET: usize = 200_000;

// const INPUT_FEATURES: i64 = INPUTS as i64 + BUCKET_SIZE as i64 * PIECE_BUCKETS as i64;
// const INPUT_FEATURES: i64 = INPUTS as i64;
//const INPUT_FEATURES: i64 = INPUTS as i64 + BUCKET_SIZE as i64;
const FEATURE_ABSTRACTIONS: i64 = BUCKET_SIZE as i64;

const TRAINING_STATE_FILE: &str = "./data/nets/training.state";
const AF_TRAINING_STORE_FILE: &str = "./data/nets/af_training.net";
const TRAINING_STORE_FILE: &str = "./data/nets/training.net";

trait Mode {
    fn print_info(&self);
    fn forward(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor) -> Tensor;
    fn save(&self, id: &str, vs: &VarStore, epoch: usize, lr: f64, patience: usize, best_error: f64);

    fn input_feature_count(&self) -> i64;
}

struct EvalNet {
    inputs: InputLayer,
    output: OutputLayer,
    use_feature_abstractions: bool,
    input_feature_count: i64,
}

impl EvalNet {
    fn new(vs: &nn::Path, use_feature_abstractions: bool) -> Self {
        let input_feature_count = if use_feature_abstractions {
            INPUTS as i64 + FEATURE_ABSTRACTIONS
        } else {
            INPUTS as i64
        };

        EvalNet{
            inputs: input_layer(vs / "input", input_feature_count, HL1_HALF_NODES as i64),
            output: output_layer(vs / "output", HL1_NODES as i64),
            use_feature_abstractions,
            input_feature_count,
        }
    }
}

impl Mode for EvalNet {
    fn print_info(&self) {
        info!("Training neural network: {} x 2x{} x 1 (feature abstractions: {})", self.input_feature_count, HL1_HALF_NODES, self.use_feature_abstractions);
    }

    fn forward(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor) -> Tensor {
        let acc = self.inputs.forward(white_xs, black_xs, stms).clamp(0.0, MAX_RELU as f64).square();

        (self.output.forward(&acc))
            .multiply_scalar(K_DIV)
            .sigmoid()
    }

    fn save(&self, id: &str, vs: &VarStore, epoch: usize, lr: f64, patience: usize, best_error: f64) {
        let vars = vs.variables();

        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        write_u8(&mut writer, b'V').unwrap();

        let hidden_layers = 1;
        write_u8(&mut writer, hidden_layers).unwrap(); // Number of hidden layers

        let input_bias = Tensor::cat(&[vars.get("input.own_bias").unwrap(), vars.get("input.opp_bias").unwrap()], 0);
        let inputs = vars.get("input.weight").unwrap();
        if self.use_feature_abstractions {
            write_raw(&mut writer, &merge_abstract_features(inputs)).expect("Could not write layer");
        } else {
            write_raw(&mut writer, inputs).expect("Could not write layer");
        };
        write_raw(&mut writer, &input_bias).expect("Could not write layer");

        write_raw(&mut writer, vars.get("output.weight").unwrap()).expect("Could not write layer");
        write_raw(&mut writer, vars.get("output.bias").unwrap()).expect("Could not write layer");

        save_training_state(epoch, lr, patience, best_error, self.use_feature_abstractions);

        if self.use_feature_abstractions {
            vs.save(AF_TRAINING_STORE_FILE).expect("could not save net variables");

            let vs2 = VarStore::new(Device::Cpu);

            let mut inputs2 = input_layer(vs2.root() / "input", INPUTS as i64, HL1_HALF_NODES as i64);
            let mut output2 = output_layer(vs2.root() / "output", HL1_NODES as i64);

            inputs2.copy_from(&merge_abstract_features(&vars.get("input.weight").unwrap()), vars.get("input.own_bias").unwrap(), vars.get("input.opp_bias").unwrap());
            output2.copy_from(vars.get("output.weight").unwrap(), vars.get("output.bias").unwrap());

            vs2.save(TRAINING_STORE_FILE).expect("could not save net variables");
        } else {
            vs.save(TRAINING_STORE_FILE).expect("could not save net variables");
        }
    }

    fn input_feature_count(&self) -> i64 {
        self.input_feature_count
    }
}

fn merge_abstract_features(all_features: &Tensor) -> Tensor {
    let real_features = all_features.i((.., ..INPUTS as i64));
    let abstracted_features = all_features.i((.., INPUTS as i64..));

    let abstracted_features_repeated = abstracted_features.i((.., (0..BUCKET_SIZE as i64))).repeat([1, BUCKETS as i64]);
    real_features + abstracted_features_repeated
}

fn config_optimizer(vs: &VarStore, initial_lr: f64) -> Optimizer {
    let mut opt = nn::AdamW::default().build(vs, initial_lr).unwrap();
    opt.set_weight_decay(0.01);

    opt
}

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}
#[derive(Args, Debug)]
struct TrainArgs {
    /// Whether to use feature abstractions or not
    #[arg(short, long, default_value_t = false)]
    use_feature_abstractions: bool,

    /// Number of training data reader threads
    #[arg(short, long, default_value_t = 5)]
    reader_threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train(TrainArgs),
}

pub fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).target(Target::Stdout).init();

    if !syzygy::tb::init(String::from("/mnt/tb")) {
        error!("could not initialize tablebases");
        exit(1);
    } else {
        let count = syzygy::tb::max_piece_count();
        if count == 0 {
            warn!("debug no tablebases found");
        } else {
            info!("debug found {}-men tablebases", syzygy::tb::max_piece_count());
        }
    }

    let start = Instant::now();
    let args = Cli::parse();
    match args.command {
        Commands::Train(args) => {
            let restart_without_feature_abstractions = train(args.reader_threads, args.use_feature_abstractions, start);
            if restart_without_feature_abstractions {
                train(args.reader_threads, false, start);
            }
        },
    }
}

fn train(data_reader_threads: usize, mut use_feature_abstractions: bool, start: Instant) -> bool {
    let device = Device::cuda_if_available();
    let mut vs = VarStore::new(device);

    let available_parallelism = thread::available_parallelism().unwrap().get();
    info!("Scanning available training sets ...");
    let training_set_count = convert_sets(
        available_parallelism,
        "training",
        FEN_TRAINING_SET_PATH,
        LZ4_TRAINING_SET_PATH,
        true,
        true
    );

    info!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * SAMPLES_PER_SET
    );

    info!("Scanning available test sets ...");
    let test_set_count = convert_sets(available_parallelism, "test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, false, true);
    info!("Using {} test sets with a total of {} positions", test_set_count, test_set_count * 200_000);

    info!("Duration: {:?} / {} per second", start.elapsed(), (training_set_count + (test_set_count * SAMPLES_PER_SET)) as f64 / start.elapsed().as_secs_f64());

    info!("Reading test sets ...");
    let test_set = read_test_sets(use_feature_abstractions, test_set_count);

    let mut rng = ThreadRng::default();

    let id_source =
        Arc::new(RwLock::new(IDSource::new(&mut rng, 1, training_set_count, SETS_PER_BATCH)));

    info!("Using validation set with {} samples", test_set.0.len());

    let stop_readers = Arc::new(AtomicBool::default());

    let net = Box::new(EvalNet::new(&vs.root(), use_feature_abstractions));

    let (tx, rx) = mpsc::sync_channel::<(Tensor, Tensor, Tensor, Tensor)>(data_reader_threads * 2);
    spawn_data_reader_threads(data_reader_threads, device, &tx, &id_source, &stop_readers, use_feature_abstractions, net.input_feature_count());

    let (mut epoch, mut lr, mut patience, mut best_error) = if let Some((epoch, lr, patience, best_error, used_feature_abstractions)) = load_training_state() {
        info!("Continue training from existing training state ...");
        if !used_feature_abstractions && use_feature_abstractions != used_feature_abstractions {
            use_feature_abstractions = used_feature_abstractions;
            info!("Override use_feature_abstractions to {}", use_feature_abstractions);
        }
        let file = if use_feature_abstractions { AF_TRAINING_STORE_FILE } else { TRAINING_STORE_FILE };

        vs.load(file).expect("Could not load training store");
        (epoch, lr, patience, best_error)
    } else {
        info!("Start new training ...");
        (1, INIT_LR, INITIAL_PATIENCE, f64::MAX)
    };

    net.print_info();

    let mut calc_validation_error = true;
    let mut curr_epoch_samples = 0;

    let mut opt = config_optimizer(&vs, lr);

    let mut epoch_validation_threshold = NORM_EPOCH_SIZE;

    let mut missed_improvements = 0;
    let mut total_samples = 0;

    let input_feature_count = net.input_feature_count();

    let mut restart_without_feature_abstractions = false;

    loop {
        if calc_validation_error {
            let validation_start = Instant::now();
            let validation_error = tch::no_grad(|| {
                let mut err = 0f64;

                let mut count = 0;
                for batch in test_set.0.chunks_exact(TEST_BATCH_SIZE) {
                    let (test_wpov_xs, test_bpov_xs, test_stms, mut test_ys) = to_sparse_tensors(batch, device, input_feature_count);

                    test_ys = test_ys.multiply_scalar_(K_DIV).sigmoid();
                    let result = &net.forward(&test_wpov_xs, &test_bpov_xs, &test_stms);
                    let mean = (result - test_ys).abs().float_power_tensor_scalar(ERR_EXP).mean(Kind::Float);
                    err += f64::try_from(mean).unwrap();
                    count += 1;
                }

                err / count as f64
            });

            calc_validation_error = false;

            if validation_error <= best_error || epoch <= 4 {
                best_error = validation_error;
                net.save(&format!("{}", epoch), &vs, epoch, lr, patience, best_error);

                missed_improvements = 0;
            } else {
                missed_improvements += 1;
            }

            if missed_improvements >= patience {
                patience = (patience / 2).max(2);
                missed_improvements = 0;
                lr *= 0.4;
                opt.set_lr(lr);

                if lr <= 0.00000166 {
                    break;
                }

                match vs.load(TRAINING_STORE_FILE) {
                    Ok(_) => {}
                    Err(e) => {
                        info!("Could not load previous best net: {}", e);
                    }
                }

                if use_feature_abstractions {
                    net.save(&format!("{}", epoch), &vs, epoch, lr, patience, best_error);
                    info!("Restart without feature abstractions ...");
                    restart_without_feature_abstractions = true;
                    break;
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

            total_samples += sample_count;
            curr_epoch_samples += sample_count;

            let result = net.forward(&wpov_data_batch, &bpov_data_batch, &stm_data_batch);
            let loss = (result - &(label_batch.multiply_scalar(K_DIV).sigmoid())).abs().float_power_tensor_scalar(ERR_EXP).mean(Kind::Float);

            opt.backward_step(&loss);

            train_loss += f64::try_from(&loss).unwrap();

            batch_count += 1;
        }

        if total_samples >= epoch_validation_threshold {
            epoch_validation_threshold = total_samples + NORM_EPOCH_SIZE;
            curr_epoch_samples = 0;
            calc_validation_error = true;
            epoch += 1;
        }

        train_loss /= batch_count as f64;

        let samples_per_sec =
            (train_count as f64 * 1000.0) / Instant::now().duration_since(iter_start).as_millis() as f64;

        info!(
            "Norm. Epoch: {:02} [ {:2}% ] / LR: {:1.8}, BS: {} - Best err: {:1.10} / Train err: {:1.10} / {:.0} samples/sec / Elapsed: {:3.1}m",
            epoch, (curr_epoch_samples * 100 / NORM_EPOCH_SIZE).min(100), lr, BATCH_SIZE, best_error, train_loss, samples_per_sec,
            start.elapsed().as_secs_f64() / 60.0
        );
    }

    info!("- Stopping reader threads ...");
    stop_readers.store(true, Ordering::Release);
    thread::sleep(Duration::from_millis(50));
    while !matches!(rx.try_recv(), Err(_)) {
        thread::sleep(Duration::from_millis(10));
    }

    if !restart_without_feature_abstractions {
        info!("- Training finished in {:.1} minutes", Instant::now().duration_since(start).as_secs() as f64 / 60.0);
    }

    restart_without_feature_abstractions
}

fn read_test_sets(use_feature_abstractions: bool, test_set_count: usize) -> GpuDataSamples {
    let mut test_set = GpuDataSamples(vec![DataSample::default(); SAMPLES_PER_SET * test_set_count]);
    let mut start = 0;
    for i in 1..=test_set_count {
        read_samples(&mut test_set, start, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str(), use_feature_abstractions);
        start += SAMPLES_PER_SET;
    }
    test_set
}

fn save_training_state(epoch: usize, lr: f64, patience: usize, best_error: f64, use_feature_abstractions: bool) {
    let file = File::create(TRAINING_STATE_FILE).expect("Could not create output file");
    let mut writer = BufWriter::new(file);

    write_u16(&mut writer, epoch as u16).expect("Could not write epoch");
    write_f32(&mut writer, lr as f32).expect("Could not write learning rate");
    write_u16(&mut writer, patience as u16).expect("Could not write learning rate");
    write_f32(&mut writer, best_error as f32).expect("Could not write learning rate");
    write_u8(&mut writer, u8::from(use_feature_abstractions)).expect("Could not write use_feature_abstractions");
}

fn load_training_state() -> Option<(usize, f64, usize, f64, bool)> {
    match File::open(TRAINING_STATE_FILE) {
        Ok(file) => {
            let mut reader = BufReader::new(file);

            let epoch = read_u16(&mut reader).expect("Could not read epoch");
            let lr = read_f32(&mut reader).expect("Could not read learning rate");
            let patience = read_u16(&mut reader).expect("Could not read patience");
            let best_error = read_f32(&mut reader).expect("Could not read best error");
            let use_feature_abstractions = read_u8(&mut reader).expect("Could not read use_feature_abstractions") == u8::from(true);

            Some((epoch as usize, lr as f64, patience as usize, best_error as f64, use_feature_abstractions))
        }

        Err(_) => {
            None
        }
    }
}

fn to_sparse_tensors(samples: &[DataSample], device: Device, input_feature_count: i64) -> (Tensor, Tensor, Tensor, Tensor) {
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

    let wpov_indices = Tensor::from_slice(&wpov_indices).to(device).view((2, wpov_value_count));
    let bpov_indices = Tensor::from_slice(&bpov_indices).to(device).view((2, bpov_value_count));
    let wpov_values = Tensor::ones(&[wpov_value_count], (Kind::Float, device));
    let bpov_values = Tensor::ones(&[bpov_value_count], (Kind::Float, device));


    let wpov_xs = Tensor::sparse_coo_tensor_indices_size(
        &wpov_indices,
        &wpov_values,
        &[samples.len() as i64, input_feature_count],
        (Kind::Float, device),
        true
    );

    let bpov_xs = Tensor::sparse_coo_tensor_indices_size(
        &bpov_indices,
        &bpov_values,
        &[samples.len() as i64, input_feature_count],
        (Kind::Float, device),
        true
    );

    (wpov_xs,
     bpov_xs,
     Tensor::from_slice(&stms).to(device).view((samples.len() as i64, 1)),
     Tensor::from_slice(&ys).to(device).view((samples.len() as i64, 1)))
}

fn write_raw(writer: &mut BufWriter<File>, weights: &Tensor) -> Result<(), Error> {
    for size in weights.size() {
        write_u16(writer, size as u16)?;
    }

    let ws: Vec<f32> = weights.flatten(0, -1).try_into().unwrap();
    for &weight in ws.iter() {
        write_f32(writer, weight)?;
    }

    Ok(())
}

fn spawn_data_reader_threads(
    threads: usize, device: Device, tx: &SyncSender<(Tensor, Tensor, Tensor, Tensor)>,
    id_source: &Arc<RwLock<IDSource>>,
    is_stopped: &Arc<AtomicBool>,
    use_feature_abstractions: bool,
    input_feature_count: i64,
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
                        read_samples( &mut training_samples, start, format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id).as_str(), use_feature_abstractions);
                        start += SAMPLES_PER_SET;
                    }

                    training_samples.0.shuffle(&mut rng);
                    count = start;

                    t_start = training_samples.0.len() - BATCH_SIZE as usize;
                    t_end = training_samples.0.len();
                }

                let (wpov_xs, bpov_xs, stms, ys) = to_sparse_tensors(&training_samples.0[t_start..t_end], device, input_feature_count);
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

