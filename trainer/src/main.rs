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

#[cfg(feature = "lr_finder")]
mod lr_finder;
mod lr_scheduler;

use crate::lr_scheduler::LrScheduler;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use itertools::Itertools;
use lz4_flex::frame::FrameEncoder;
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use std::cmp::{max, min};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::hash::Hasher;
use std::io::{stdout, BufRead, BufReader, BufWriter, Error, ErrorKind, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use tch::nn::{ModuleT, Optimizer, SequentialT, VarStore};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Reduction, Tensor};
use velvet::bitboard::{v_mirror, BitBoard};
use velvet::fen::{parse_fen, FenParseResult};
use velvet::nn::io::{BitWriter, CodeBook};
use velvet::nn::HL_NODES;
pub use velvet::nn::INPUTS;

const FEATURES_PER_BUCKET: i64 = 64 * 6 * 2;

const INPUT_FEATURES: i64 = FEATURES_PER_BUCKET * 5;

const DATA_WRITER_THREADS: usize = 4;

const TEST_SETS: usize = 4;

const BATCH_SIZE: i64 = 50000;

const K: f64 = 1.603;
const K_DIV: f64 = K / 400.0;

const MIN_TRAINING_SET_ID: usize = 1;
const FEN_TRAINING_SET_PATH: &str = "./data/train_fen/";
const LZ4_TRAINING_SET_PATH: &str = "./data/train_lz4";
const FEN_TEST_SET_PATH: &str = "./data/test_fen";
const LZ4_TEST_SET_PATH: &str = "./data/test_lz4";
const POS_PER_SET: usize = 200_000;

trait Mode {
    fn print_info(&self);
    fn net(&self, vs: &nn::Path) -> SequentialT;
    fn save_raw(&self, id: &str, vs: &nn::VarStore);
    fn save_quantized(&self, net_id: String);
}

struct EvalNet {}

impl Mode for EvalNet {
    fn print_info(&self) {
        println!("Training neural network for evaluation");
    }

    fn net(&self, vs: &nn::Path) -> SequentialT {
        nn::seq_t()
            .add(nn::linear(vs / "input", INPUT_FEATURES, HL_NODES as i64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs / "output", HL_NODES as i64, 1, Default::default()))
    }

    fn save_raw(&self, id: &str, vs: &nn::VarStore) {
        let vars = vs.variables();

        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8('V' as i8).unwrap();

        let hidden_layers = 0;
        writer.write_i8(hidden_layers).unwrap(); // Number of hidden layers

        write_layer(&mut writer, vars.get("input.weight").unwrap(), vars.get("input.bias").unwrap(), 1.0)
            .expect("Could not write layer");

        for i in 1..=hidden_layers {
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

    fn save_quantized(&self, net_id: String) {
        let in_file = &format!("./data/nets/velvet_{}.nn", net_id);
        let out_file = &format!("./data/nets/velvet_{}.qnn", net_id);

        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open NN file: {}", in_file));
        let mut reader = BufReader::new(file);

        read_header(&mut reader, 0).expect("could not read NN header data");

        let input = read_layer(&mut reader).expect("could not read NN input layer");
        let output = read_layer(&mut reader).expect("could not read NN output layer");

        // Regroup input weights for faster incremental updates
        let mut input_weights = Vec::with_capacity(input.weights.len());
        for i in 0..INPUTS {
            input.weights.iter().skip(i).step_by(INPUTS).for_each(|&w| input_weights.push(w));
        }

        let mut biases = input.biases.clone();
        biases.sort_by_key(|b| -(b * 1000.0) as i32);

        let max_bias = biases[0];
        let min_bias = biases[biases.len() - 1];

        let mut max_sum = f32::MIN;
        let mut min_sum = f32::MAX;

        for c in input.weights.chunks(768 * HL_NODES).into_iter() {
            let mut weights = c.to_vec();
            weights.sort_by_key(|w| -(w * 1000.0) as i32);
            let w_max_sum = weights.iter().take(32).sum::<f32>();

            if w_max_sum > max_sum {
                max_sum = w_max_sum;
            }

            weights.sort_by_key(|w| (w * 1000.0) as i32);
            let w_min_sum = weights.iter().take(32).sum::<f32>();
            if w_min_sum < min_sum {
                min_sum = w_min_sum;
            }
        }

        min_sum += min_bias;
        max_sum += max_bias;

        let bound = if -min_sum > max_sum { -min_sum } else { max_sum };

        let precision_bits = 15 - ((32767.0 / bound) as i16 | 1).leading_zeros();
        let scale_bits = 16 - precision_bits;
        println!("Required scale bits: {}", scale_bits);
        println!("Precision bits     : {}", precision_bits);

        let fp_one = 1 << precision_bits;
        println!("{}", max_sum * fp_one as f32);
        println!("{}", min_sum * fp_one as f32);

        let file = File::create(out_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8(precision_bits as i8).unwrap();

        print!("Input weights: ");
        write_quantized(&mut writer, fp_one, input_weights).expect("Could not write quantized input weights");
        print!("Input biases: ");
        write_quantized(&mut writer, fp_one, input.biases).expect("Could not write quantized input biases");

        print!("Output weights: ");
        write_quantized(&mut writer, fp_one, output.weights).expect("Could not write quantized output layer weights");

        writer
            .write_i16::<LittleEndian>((output.biases[0] * fp_one as f32) as i16)
            .expect("Could not write quantized output bias");
    }
}

struct PieceSquareTables {}

impl Mode for PieceSquareTables {
    fn print_info(&self) {
        println!("Training piece square tables");
    }

    fn net(&self, vs: &nn::Path) -> SequentialT {
        nn::seq_t().add(nn::linear(vs / "input", INPUT_FEATURES, 1, Default::default()))
    }

    fn save_raw(&self, id: &str, vs: &nn::VarStore) {
        let vars = vs.variables();

        let file = File::create(format!("./data/nets/velvet_psq_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8('V' as i8).unwrap();

        let hidden_layers = 0;
        writer.write_i8(hidden_layers).unwrap(); // Number of hidden layers

        write_layer(&mut writer, vars.get("input.weight").unwrap(), vars.get("input.bias").unwrap(), 2048.0)
            .expect("Could not write layer");
    }

    fn save_quantized(&self, net_id: String) {
        let in_file = &format!("./data/nets/velvet_psq_{}.nn", net_id);
        let out_file = &format!("./data/nets/velvet_psq_{}.qnn", net_id);

        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open PSQ file: {}", in_file));
        let mut reader = BufReader::new(file);

        read_header(&mut reader, 0).expect("could not read PSQ header data");

        let input = read_layer(&mut reader).expect("could not read PSQ input layer");

        let file = File::create(out_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        write_quantized(&mut writer, 1, input.weights).expect("Could not write quantized input weights");

        writer.write_i16::<LittleEndian>(input.biases[0] as i16).expect("Could not write quantized output bias");
    }
}

type DataSample = (Vec<i64>, f32);

fn config_optimizer(vs: &VarStore, initial_lr: f64) -> Optimizer {
    let mut opt = nn::AdamW::default().build(vs, initial_lr).unwrap();
    opt.set_weight_decay(0.35);

    opt
}

pub fn main() {
    if env::args().len() <= 2 {
        println!("Usage: trainer [psq|eval] [reader thread count]");
        return;
    }

    let mode: Box<dyn Mode> = match env::args().nth(1).unwrap().to_lowercase().as_str() {
        "psq" => Box::new(PieceSquareTables {}),
        "eval" => Box::new(EvalNet {}),
        _ => {
            println!("Usage: trainer [psq|eval] [reader thread count]");
            return;
        }
    };

    let data_reader_threads = match usize::from_str(env::args().nth(2).unwrap().as_str()) {
        Ok(count) => max(1, count),
        Err(_) => {
            println!("Usage: trainer [psq|eval] [reader thread count]");
            return;
        }
    };

    mode.print_info();

    println!("Scanning available training sets ...");
    let max_training_set_id =
        convert_sets("training", FEN_TRAINING_SET_PATH, LZ4_TRAINING_SET_PATH, MIN_TRAINING_SET_ID, 0.5);
    let training_set_count = (max_training_set_id - MIN_TRAINING_SET_ID) + 1;
    println!(
        "Using {} training sets with a total of {} positions",
        training_set_count,
        training_set_count * POS_PER_SET
    );

    println!("Scanning available test sets ...");
    let max_test_set_id = convert_sets("test", FEN_TEST_SET_PATH, LZ4_TEST_SET_PATH, 1, 0.0);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);

    let mut samples = Vec::with_capacity(POS_PER_SET * TEST_SETS);
    for i in 1..=min(TEST_SETS, max_test_set_id) {
        read_from_tensor_file(&mut samples, format!("{}/{}.lz4", LZ4_TEST_SET_PATH, i).as_str());
    }

    let (test_xs, mut test_ys) = to_sparse_tensors(&samples, device);

    println!("Using {} samples for validation", samples.len());

    test_ys = test_ys.multiply_scalar(2048.0 * K_DIV).sigmoid();

    let stop_readers = Arc::new(AtomicBool::default());

    let (tx, rx) = mpsc::sync_channel::<(usize, Tensor, Tensor)>(data_reader_threads * 2);
    spawn_data_reader_threads(data_reader_threads, max_training_set_id, device, &tx, &stop_readers);

    let net = mode.net(&vs.root());

    let mut best_loss = f64::MAX;

    // find_lr(&test_xs, &test_ys, &rx, &net, &mut opt);
    // return;

    let mut epoch = 1;

    let mut epoch_changed = false;
    let mut epoch_start = Instant::now();
    let mut epoch_sample_count = 0;
    let mut last_improved_epoch = 0;

    let max_epochs = 40;
    let lr_scheduler = LrScheduler::new(max_epochs, 0.005, 0.0005, 0.000005);
    let mut lr = lr_scheduler.calc_lr(1);
    let mut opt = config_optimizer(&vs, lr);

    let net_id: String = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_millis().to_string();

    let start_time = Instant::now();

    loop {
        opt.set_lr(lr);

        let mut train_loss = 0.0;
        let mut batch_count = 0;

        let mut train_count = 0;

        let train_start = Instant::now();
        while train_count < POS_PER_SET * 2 {
            let (batch_epoch, data_batch, label_batch) = rx.recv().expect("Could not receive test positions");
            train_count += label_batch.numel();

            if batch_epoch > epoch {
                epoch_changed = true;
                epoch = batch_epoch;
            }

            let loss = net
                .forward_t(&data_batch, true)
                .multiply_scalar(2048.0 * K_DIV)
                .sigmoid()
                .mse_loss(&(label_batch.multiply_scalar(2048.0 * K_DIV)).sigmoid(), Reduction::Mean);

            opt.backward_step(&loss);
            train_loss += f64::from(&loss);

            batch_count += 1;
        }

        let test_loss = tch::no_grad(|| {
            f64::from(
                &net.forward_t(&test_xs, false)
                    .multiply_scalar(2048.0 * K_DIV)
                    .sigmoid()
                    .mse_loss(&test_ys, Reduction::Mean),
            )
        });

        epoch_sample_count += train_count;

        train_loss /= batch_count as f64;

        if test_loss <= best_loss {
            last_improved_epoch = epoch;
            best_loss = test_loss;
            mode.save_raw(&net_id, &vs);
        }

        let samples_per_sec =
            (train_count as f64 * 1000.0) / Instant::now().duration_since(train_start).as_millis() as f64;

        println!(
            "epoch: {:3} lr: {:.8} batch size: {:5} train loss: {:8.8} test loss: {:8.8}, best loss: {:8.8} ({:.2} samples/s)",
            epoch,
            lr,
            BATCH_SIZE,
            train_loss,
            test_loss,
            best_loss,
            samples_per_sec
        );

        if epoch_changed {
            let epoch_duration = Instant::now().duration_since(epoch_start);
            let samples_per_sec = (epoch_sample_count as f64 * 1000.0) / epoch_duration.as_millis() as f64;
            println!(
                "- Epoch finished: {:.2} samples/s (epoch duration: {} seconds)",
                samples_per_sec,
                epoch_duration.as_secs()
            );

            if epoch > max_epochs {
                if (epoch - last_improved_epoch) > 2 {
                    break;
                }

                lr *= 0.8;
            } else {
                lr = lr_scheduler.calc_lr(min(epoch, max_epochs));
            }

            epoch_changed = false;
            epoch_sample_count = 0;
            epoch_start = Instant::now();
        }
    }

    println!("- Stopping reader threads ...");
    stop_readers.store(true, Ordering::Release);
    thread::sleep(Duration::from_millis(50));
    while !matches!(rx.try_recv(), Err(_)) {
        thread::sleep(Duration::from_millis(10));
    }

    println!("- Training finished in {:.1} minutes", Instant::now().duration_since(start_time).as_secs() as f64 / 60.0);
    mode.save_quantized(net_id);
}

fn to_sparse_tensors(samples: &[DataSample], device: Device) -> (Tensor, Tensor) {
    let mut ys = Vec::with_capacity(samples.len());

    let mut indices = Vec::with_capacity(2_097_152);

    for (j, sample) in samples.iter().enumerate() {
        ys.push(sample.1);
        indices.resize(indices.len() + sample.0.len(), j as i64);
    }

    for sample in samples.iter() {
        indices.extend_from_slice(&sample.0);
    }

    let value_count = indices.len() as i64 / 2;

    let indices = Tensor::of_slice(&indices).to(device).view((2, value_count));
    let values = Tensor::ones(&[value_count], (Kind::Float, device));

    let xs = Tensor::sparse_coo_tensor_indices_size(
        &indices,
        &values,
        &[samples.len() as i64, INPUT_FEATURES],
        (Kind::Float, device),
    )
    .coalesce();

    (xs, Tensor::of_slice(&ys).to(device).view((samples.len() as i64, 1)))
}

#[allow(dead_code)]
fn to_dense_tensors(samples: &[DataSample], device: Device) -> (Tensor, Tensor) {
    let mut ys = Vec::with_capacity(samples.len());
    let mut xs = Tensor::zeros(&[samples.len() as i64, INPUT_FEATURES], (Kind::Float, device));

    let mut columns = Vec::new();
    let mut rows = Vec::new();

    for (j, sample) in samples.iter().enumerate() {
        ys.push(sample.1);

        for i in sample.0.iter() {
            columns.push(*i);
            rows.push(j as i64);
        }
    }

    let row_select = Tensor::of_slice(&rows).to_device(device);
    let col_select = Tensor::of_slice(&columns).to_device(device);
    let values = Tensor::ones(&[columns.len() as i64], (Kind::Float, device));

    let _ = xs.index_put_(&[Some(row_select), Some(col_select)], &values, false);

    (xs, Tensor::of_slice(&ys).view((samples.len() as i64, 1)).to_device_(device, Kind::Float, false, true))
}

fn convert_sets(caption: &str, in_path: &str, out_path: &str, min_id: usize, game_result_pct: f32) -> usize {
    let mut max_set_id = 0;
    let mut min_unconverted_id = 25000;
    for id in min_id..25000 {
        if !Path::new(&format!("{}/test_pos_{}.fen", in_path, id)).exists() {
            break;
        }

        if !Path::new(&format!("{}/{}.lz4", out_path, id)).exists() {
            min_unconverted_id = min(id, min_unconverted_id);
        }

        max_set_id = id;
    }

    if min_unconverted_id < max_set_id {
        println!("Converting {} added {} sets ...", (max_set_id - min_unconverted_id + 1), caption);
        convert_test_pos(in_path.to_string(), out_path.to_string(), min_unconverted_id, max_set_id, game_result_pct)
            .expect("Could not convert test positions!");
    }

    max_set_id
}

fn convert_test_pos(
    in_path: String, out_path: String, min_unconverted_id: usize, max_training_set_id: usize, game_result_pct: f32,
) -> Result<(), Error> {
    let mut threads = Vec::new();
    for c in 1..=DATA_WRITER_THREADS {
        let in_path2 = in_path.clone();
        let out_path2 = out_path.clone();
        threads.push(thread::spawn(move || {
            for i in ((c + min_unconverted_id - 1)..=max_training_set_id).step_by(DATA_WRITER_THREADS) {
                print!("{} ", i);
                stdout().flush().unwrap();

                let file = File::create(format!("{}/{}.lz4", out_path2, i)).expect("Could not create tensor data file");
                let encoder = lz4_flex::frame::FrameEncoder::new(file);
                let mut writer = BufWriter::with_capacity(1024 * 1024, encoder);

                let ys = read_from_fen_file(
                    format!("{}/test_pos_{}.fen", in_path2, i).as_str(),
                    &mut writer,
                    game_result_pct,
                );
                writer.write_u16::<LittleEndian>(u16::MAX).unwrap();

                for y in ys {
                    writer.write_f32::<LittleEndian>(y).unwrap();
                }

                writer.flush().unwrap();
            }
        }));
    }

    for t in threads {
        t.join().unwrap();
    }

    println!("\nConversion finished");

    Ok(())
}

fn read_from_tensor_file(samples: &mut Vec<DataSample>, file_name: &str) {
    let file = File::open(file_name).unwrap_or_else(|_| panic!("Could not open test position file: {}", file_name));
    let decoder = lz4_flex::frame::FrameDecoder::new(file);
    let mut reader = BufReader::with_capacity(1024 * 1024, decoder);

    let start = samples.len();

    let mut total_samples = 0;

    loop {
        let bb_map = reader.read_u16::<LittleEndian>().unwrap();
        if bb_map == u16::MAX {
            break;
        }
        total_samples += 1;

        let mut xs = Vec::with_capacity(34);

        let kings = reader.read_u16::<LittleEndian>().unwrap();
        let white_king = kings & 0b111111;
        let black_king = kings >> 8;

        let bucket = if (bb_map & (1 << 4)) == 0 && (bb_map & (1 << 9)) == 0 {
            let own_rooks = if (bb_map & (1 << 3)) == 0 { 0b00 } else { 0b10 };
            let opp_rooks = if (bb_map & (1 << 8)) == 0 { 0b00 } else { 0b01 };

            own_rooks | opp_rooks
        } else {
            4
        };

        let offset = FEATURES_PER_BUCKET * bucket as i64;

        for i in 1i8..=5i8 {
            if bb_map & (1 << (i - 1)) != 0 {
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    xs.push(offset + (i as i64 - 1) * 2 * 64 + pos as i64);
                }
            }

            if bb_map & (1 << (i as usize + 4)) != 0 {
                let bb = reader.read_u64::<LittleEndian>().unwrap();
                for pos in BitBoard(bb) {
                    xs.push(offset + (i as i64 - 1) * 2 * 64 + 64 + pos as i64);
                }
            }
        }

        xs.push(offset + 5 * 2 * 64 + white_king as i64);
        xs.push(offset + 5 * 2 * 64 + 64 + black_king as i64);

        samples.push((xs, 0f32));
    }

    if total_samples != POS_PER_SET {
        panic!("Tensor file does not contain the expected 200_000 samples, but: {}", total_samples);
    }

    for i in 0..POS_PER_SET {
        let result = reader.read_f32::<LittleEndian>().unwrap();
        unsafe { samples.get_unchecked_mut(start + i).1 = result };
    }
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
    threads: usize, max_training_set_id: usize, device: Device, tx: &SyncSender<(usize, Tensor, Tensor)>,
    is_stopped: &Arc<AtomicBool>,
) {
    for start in MIN_TRAINING_SET_ID..(threads + MIN_TRAINING_SET_ID) {
        let tx2 = tx.clone();
        let is_stopped = is_stopped.clone();
        thread::spawn(move || {
            let mut epoch = 1;

            let mut r = rand::prelude::StdRng::from_entropy();

            let mut samples = Vec::with_capacity(2 * POS_PER_SET);

            loop {
                // Shuffling all data sets is too time consuming.
                // Instead, two random data sets will be merged and shuffled
                let mut ids = (start..(max_training_set_id - TEST_SETS)).step_by(threads).collect_vec();
                ids.shuffle(&mut r);

                while ids.len() >= 2 {
                    let id1 = ids.pop().unwrap();
                    let id2 = ids.pop().unwrap();

                    samples.clear();
                    read_from_tensor_file(&mut samples, format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id1).as_str());
                    read_from_tensor_file(&mut samples, format!("{}/{}.lz4", LZ4_TRAINING_SET_PATH, id2).as_str());

                    let mut remaining_samples: &mut [DataSample] = &mut samples;

                    while remaining_samples.len() >= BATCH_SIZE as usize {
                        let (batch, remaining_samples2) =
                            remaining_samples.partial_shuffle(&mut r, BATCH_SIZE as usize);
                        remaining_samples = remaining_samples2;

                        let (xs, ys) = to_sparse_tensors(batch, device);
                        tx2.send((epoch, xs, ys)).expect("Could not send training batch");
                        if is_stopped.load(Ordering::Acquire) {
                            return;
                        }
                    }
                }
                epoch += 1;
            }
        });
    }
}

fn read_from_fen_file(file_name: &str, writer: &mut BufWriter<FrameEncoder<File>>, game_result_pct: f32) -> Vec<f32> {
    let file = File::open(file_name).expect("Could not open test position file");
    let mut reader = BufReader::new(file);

    let mut ys = Vec::with_capacity(POS_PER_SET);

    loop {
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(read) => {
                if read == 0 {
                    return ys;
                }
            }

            Err(e) => panic!("Reading test position file failed: {}", e),
        };

        let parts = line.trim_end().split(' ').collect_vec();

        let mut result = if parts.len() == 10 {
            const SCORE_IDX: usize = 6;
            let score = i32::from_str(parts[SCORE_IDX]).expect("Could not parse score");

            const SCORE_PLY_IDX: usize = 7;
            let score_ply = i32::from_str(parts[SCORE_PLY_IDX]).expect("Could not parse score ply");

            const RESULT_IDX: usize = 8;
            let game_result = i32::from_str(parts[RESULT_IDX]).expect("Could not parse game result");
            const RESULT_PLY_IDX: usize = 9;
            let game_result_ply = i32::from_str(parts[RESULT_PLY_IDX]).expect("Could not parse game result ply");

            if game_result_pct > 0.0 {
                let result = game_result as f32 * 4000.0;
                let result_pct = 2.0 / (max(1, game_result_ply - score_ply) * 2) as f32;
                let score_pct = 1.0 - result_pct;

                ((score as f32) * score_pct + result * result_pct) / 2048.0
            } else {
                score as f32 / 2048.0
            }
        } else if parts.len() == 8 {
            const SCORE_IDX: usize = 7;
            let score = i32::from_str(parts[SCORE_IDX]).expect("Could not parse score");
            score as f32 / 2048.0
        } else {
            panic!("Invalid test position entry: {}", line);
        };

        let fen: String = (parts[0..=5].join(" ") as String).replace("~", "");

        let (mut pieces, active_player) = match parse_fen(fen.as_str()) {
            Ok(FenParseResult { pieces, active_player, .. }) => (pieces, active_player),
            Err(e) => panic!("could not parse FEN: {}", e),
        };

        if active_player.is_black() {
            mirror(&mut pieces);
            result = -result;
        }

        ys.push(result as f32);

        let mut black_king_pos = 0;
        let mut white_king_pos = 0;
        let mut bb: [u64; 13] = [0; 13];

        for (pos, piece) in pieces.iter().enumerate() {
            if *piece == 0 {
                continue;
            }

            if *piece == -6 {
                black_king_pos = pos;
            } else if *piece == 6 {
                white_king_pos = pos;
            } else {
                bb[(*piece + 6) as usize] |= 1 << pos;
            }
        }

        let mut bb_map = 0;
        for i in 1i8..=5i8 {
            if bb[(i + 6) as usize] != 0 {
                bb_map |= 1 << (i - 1);
            }
            if bb[(-i + 6) as usize] != 0 {
                bb_map |= 1 << (i + 4);
            }
        }

        writer.write_u16::<LittleEndian>(bb_map).unwrap();

        let kings = white_king_pos as u16 | ((black_king_pos as u16) << 8);
        writer.write_u16::<LittleEndian>(kings).unwrap();

        for i in 1i8..=5i8 {
            let bb_white = bb[(i + 6) as usize];
            if bb_white != 0 {
                writer.write_u64::<LittleEndian>(bb_white).unwrap();
            }

            let bb_black = bb[(-i + 6) as usize];
            if bb_black != 0 {
                writer.write_u64::<LittleEndian>(bb_black).unwrap();
            }
        }
    }
}

fn mirror(pieces: &mut Vec<i8>) {
    for i in 0..32 {
        let tmp = -pieces[i];
        pieces[i] = -pieces[v_mirror(i)];
        pieces[v_mirror(i)] = tmp;
    }
}

pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

fn read_header(reader: &mut BufReader<File>, expected_hl_count: usize) -> Result<(), Error> {
    let head = reader.read_i8()?;
    if head != 'V' as i8 {
        return Err(Error::new(ErrorKind::InvalidData, "missing header 'V'"));
    }

    let hidden_layer_count = reader.read_i8()? as usize;
    if hidden_layer_count != expected_hl_count {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("hidden layer count mismatch: expected {}, got {}", expected_hl_count, hidden_layer_count),
        ));
    }

    Ok(())
}

fn read_layer(reader: &mut BufReader<File>) -> Result<Layer, Error> {
    let out_count = reader.read_i32::<LittleEndian>()? as usize;
    let in_count = reader.read_i32::<LittleEndian>()? as usize;

    let mut weights = Vec::with_capacity(out_count * in_count);

    for _ in 0..(out_count * in_count) {
        let v = reader.read_f32::<LittleEndian>()?;
        weights.push(v);
    }

    let mut biases = Vec::with_capacity(out_count);
    for _ in 0..out_count {
        let v = reader.read_f32::<LittleEndian>()?;

        biases.push(v);
    }

    Ok(Layer { weights, biases })
}

fn write_quantized(writer: &mut dyn Write, fp_one: i16, values: Vec<f32>) -> Result<(), Error> {
    writer.write_u32::<LittleEndian>(values.len() as u32)?;

    let values = values.iter().map(|v| (v * fp_one as f32) as i16).collect_vec();

    let mut unused_values = HashSet::<i16>::from_iter(i16::MIN..=i16::MAX);
    for v in values.iter() {
        unused_values.remove(v);
    }

    let rep_zero_marker = unused_values.iter().copied().next().expect("No free value as marker available!");
    writer.write_i16::<LittleEndian>(rep_zero_marker)?;

    let mut outputs = Vec::with_capacity(values.len());
    let mut index = 0;
    let mut checksum = DefaultHasher::default();
    values.iter().for_each(|v| checksum.write_i16(*v));
    println!("{:016x}", checksum.finish());

    let mut total_repetitions = 0;
    while index < values.len() {
        let value = values[index];
        if let Some(repetitions) = find_zero_repetitions(value, &values[index..]) {
            total_repetitions += repetitions as usize;
            index += repetitions as usize;
            outputs.push(rep_zero_marker);
            outputs.push((repetitions as i32 - 32768) as i16);
            continue;
        }

        outputs.push(value);
        index += 1;
    }
    println!("Total repetitions: {}", total_repetitions);

    let codes = CodeBook::from_values(&outputs);
    codes.write(writer)?;

    let mut bit_writer = BitWriter::new();
    for v in outputs.iter() {
        let code = codes.get_code(*v);
        bit_writer.write(writer, code)?;
    }

    bit_writer.flush(writer)?;

    Ok(())
}

fn find_zero_repetitions(curr_value: i16, values: &[i16]) -> Option<u16> {
    if curr_value != 0 {
        return None;
    }

    let reps = values.iter().take(65535).take_while(|&v| *v == 0).count() as u16;

    if reps > 2 {
        Some(reps)
    } else {
        None
    }
}
