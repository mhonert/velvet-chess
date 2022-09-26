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

use crate::sets::DataSample;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use itertools::Itertools;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::ThreadRng;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Write};
use velvet::nn::io::{BitWriter, CodeBook};
use velvet::nn::{HL_HALF_NODES, HL_NODES, INPUTS, INPUT_WEIGHT_COUNT};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;

const WEIGHT_DECAY: f32 = 0.0001;

const K: f64 = 1.603;
const K_DIV: f64 = K / 400.0;

const MAX_WEIGHT: f32 = 127.0 / (1 << 6) as f32;
const MIN_WEIGHT: f32 = -MAX_WEIGHT;

const MIN_INPUT_WEIGHT: i32 = (MIN_WEIGHT * 1500000000.0) as i32;
const MAX_INPUT_WEIGHT: i32 = (MAX_WEIGHT * 1500000000.0) as i32;

const MIN_HIDDEN_WEIGHT: i32 = (MIN_WEIGHT * 1000000000.0) as i32;
const MAX_HIDDEN_WEIGHT: i32 = (MAX_WEIGHT * 1000000000.0) as i32;

#[derive(Copy, Clone)]
struct Gradients<const N: usize, const MIN_W: i32, const MAX_W: i32> {
    values: [f64; N],
    momentums: [(f64, f64); N],
}

impl<const N: usize, const MIN_W: i32, const MAX_W: i32> Default for Gradients<N, MIN_W, MAX_W> {
    fn default() -> Self {
        Self { values: [0f64; N], momentums: [(0f64, 0f64); N] }
    }
}

impl<const N: usize, const MIN_W: i32, const MAX_W: i32> Gradients<N, MIN_W, MAX_W> {
    pub fn copy_from(&mut self, other: &Gradients<N, MIN_W, MAX_W>) {
        self.values.copy_from_slice(&other.values);
        self.momentums.copy_from_slice(&other.momentums);
    }

    pub fn add(&mut self, idx: usize, gradient: f32) {
        *unsafe { self.values.get_unchecked_mut(idx) } += gradient as f64;
    }

    pub fn add_all(&mut self, other: &Gradients<N, MIN_W, MAX_W>) {
        for i in 0..self.values.len() {
            unsafe {
                *self.values.get_unchecked_mut(i) += *other.values.get_unchecked(i);
            }
        }
    }

    pub fn reset_values(&mut self) {
        self.values.fill(0f64);
    }

    pub fn reset_momentums(&mut self) {
        self.momentums.fill((0f64, 0f64));
    }

    pub fn update_weight(&mut self, idx: usize, weight: &mut f32, lr: f32, batch_size: usize, iteration: usize) {
        let value = unsafe { self.values.get_unchecked(idx) };

        let momentum = unsafe { self.momentums.get_unchecked_mut(idx) };

        let gradient = *value / batch_size as f64;
        momentum.0 = BETA1 * momentum.0 + (1.0 - BETA1) * gradient;
        momentum.1 = BETA2 * momentum.1 + (1.0 - BETA2) * gradient * gradient;

        let lr = (lr as f64 * (1.0 - momentum.1.powi(iteration as i32).sqrt())) as f32;
        let delta = (momentum.0 / (momentum.1.sqrt() + f64::EPSILON)) as f32;
        *weight -= lr * (delta + WEIGHT_DECAY * *weight);
        *weight = weight.clamp((MIN_W as f32) / 1000000000.0, (MAX_W as f32) / 1000000000.0);
    }
}

#[derive(Copy, Clone)]
#[repr(align(32))]
pub struct A32<T>(pub T); // Wrapper to ensure 32 Byte alignment of the wrapped type (e.g. for SIMD load/store instructions)

#[derive(Copy, Clone)]
pub struct Network {
    g: NetGradients,
    w: NetWeights,
    white_hidden_values: A32<[f32; HL_HALF_NODES]>,
    black_hidden_values: A32<[f32; HL_HALF_NODES]>,
}

#[derive(Copy, Clone)]
struct NetGradients {
    input_weight_gradients: Gradients<INPUT_WEIGHT_COUNT, MIN_INPUT_WEIGHT, MAX_INPUT_WEIGHT>,
    hidden_bias_gradients: Gradients<HL_NODES, MIN_INPUT_WEIGHT, MAX_INPUT_WEIGHT>,
    hidden_weight_gradients: Gradients<HL_NODES, MIN_HIDDEN_WEIGHT, MAX_HIDDEN_WEIGHT>,
}

#[derive(Copy, Clone)]
struct NetWeights {
    input_weights: A32<[f32; INPUT_WEIGHT_COUNT]>,
    hidden_biases: A32<[f32; HL_NODES]>,
    hidden_weights: A32<[f32; HL_NODES]>,
}

impl Network {
    pub fn new() -> Box<Network> {
        let mut nn = unsafe {
            let layout = std::alloc::Layout::new::<Network>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Network;
            Box::from_raw(ptr)
        };

        init_randomly(&mut nn.w.input_weights.0, INPUTS as usize);
        init_randomly(&mut nn.w.hidden_weights.0, HL_HALF_NODES as usize);

        nn
    }

    pub fn copy(&mut self, other: &Self) {
        self.g.input_weight_gradients.copy_from(&other.g.input_weight_gradients);
        self.w.input_weights.0.copy_from_slice(&other.w.input_weights.0);

        self.w.hidden_weights.0.copy_from_slice(&other.w.hidden_weights.0);
        self.w.hidden_biases.0.copy_from_slice(&other.w.hidden_biases.0);
        self.g.hidden_weight_gradients.copy_from(&other.g.hidden_weight_gradients);
    }

    pub fn copy_weights(&mut self, other: &Network) {
        self.w.input_weights.0.copy_from_slice(&other.w.input_weights.0);
        self.w.hidden_weights.0.copy_from_slice(&other.w.hidden_weights.0);
        self.w.hidden_biases.0.copy_from_slice(&other.w.hidden_biases.0);
    }

    pub fn reset_training_gradients(&mut self) {
        self.g.hidden_bias_gradients.reset_values();
        self.g.hidden_weight_gradients.reset_values();
        self.g.input_weight_gradients.reset_values();
    }

    pub fn reset_momentums(&mut self) {
        self.g.hidden_bias_gradients.reset_momentums();
        self.g.hidden_weight_gradients.reset_momentums();
        self.g.input_weight_gradients.reset_momentums();
    }

    pub fn add_gradients(&mut self, other: &Network) {
        self.g.input_weight_gradients.add_all(&other.g.input_weight_gradients);
        self.g.hidden_weight_gradients.add_all(&other.g.hidden_weight_gradients);
        self.g.hidden_bias_gradients.add_all(&other.g.hidden_bias_gradients);
    }

    pub fn train(&mut self, sample: &DataSample) {
        // Forward pass
        let out = self.forward(sample);

        // Gradient calculation
        let out_error = sigmoid(out, 2048.0 * K_DIV);
        let train_error = sigmoid(sample.result, 2048.0 * K_DIV);
        let out_delta = (out_error - train_error) * out_error * (1.0 - out_error) * 2048.0 * K_DIV as f32;

        // Backward pass
        let (own_hidden_values, opp_hidden_values) = if sample.wtm {
            (&self.white_hidden_values.0, &self.black_hidden_values.0)
        } else {
            (&self.black_hidden_values.0, &self.white_hidden_values.0)
        };

        for (i, hidden_value) in own_hidden_values.iter().chain(opp_hidden_values).enumerate() {
            let delta = out_delta * relu_deriv(*hidden_value);
            self.g.hidden_weight_gradients.add(i, delta);
        }

        for (i, (hidden_value, hidden_weight)) in
            own_hidden_values.iter().chain(opp_hidden_values).zip(self.w.hidden_weights.0.iter()).enumerate()
        {
            let delta = out_delta * relu_deriv(*hidden_value) * *hidden_weight;
            self.g.hidden_bias_gradients.add(i, delta);
        }

        // Input Layer weights
        let (white_hidden_weights, black_hidden_weights) = if sample.wtm {
            (&self.w.hidden_weights.0[0..HL_HALF_NODES], &self.w.hidden_weights.0[HL_HALF_NODES..])
        } else {
            (&self.w.hidden_weights.0[HL_HALF_NODES..], &self.w.hidden_weights.0[0..HL_HALF_NODES])
        };

        for i in sample.wpov_inputs.iter() {
            let weight_offset = *i as usize * HL_HALF_NODES;

            for (n, (hidden_value, hidden_weight)) in
                self.white_hidden_values.0.iter().zip(white_hidden_weights.iter()).enumerate()
            {
                let delta = out_delta * relu_deriv(*hidden_value) * *hidden_weight;
                self.g.input_weight_gradients.add(weight_offset + n, delta);
            }
        }
        for i in sample.bpov_inputs.iter() {
            let weight_offset = *i as usize * HL_HALF_NODES;

            for (n, (hidden_value, hidden_weight)) in
                self.black_hidden_values.0.iter().zip(black_hidden_weights.iter()).enumerate()
            {
                let delta = out_delta * relu_deriv(*hidden_value) * *hidden_weight;
                self.g.input_weight_gradients.add(weight_offset + n, delta);
            }
        }
    }

    pub fn update_weights(&mut self, lr: f32, batch_size: usize, iteration: usize) {
        for (i, weight) in self.w.input_weights.0.iter_mut().enumerate() {
            self.g.input_weight_gradients.update_weight(i, weight, lr, batch_size, iteration);
        }

        for (i, weight) in self.w.hidden_weights.0.iter_mut().enumerate() {
            self.g.hidden_weight_gradients.update_weight(i, weight, lr, batch_size, iteration);
        }

        for (i, bias) in self.w.hidden_biases.0.iter_mut().enumerate() {
            self.g.hidden_bias_gradients.update_weight(i, bias, lr, batch_size, iteration);
        }
    }

    pub fn test(&mut self, sample: &DataSample) -> f32 {
        // Error calculation
        let out = self.forward(sample);
        (sigmoid(sample.result, 2048.0 * K_DIV) - sigmoid(out, 2048.0 * K_DIV)).powi(2)
    }

    fn forward(&mut self, sample: &DataSample) -> f32 {
        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut self.white_hidden_values.0, &sample.bpov_inputs, &mut self.black_hidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut self.black_hidden_values.0, &sample.wpov_inputs, &mut self.white_hidden_values.0)
        };

        own_hidden_values.copy_from_slice(&self.w.hidden_biases.0[0..HL_HALF_NODES]);
        opp_hidden_values.copy_from_slice(&self.w.hidden_biases.0[HL_HALF_NODES..]);

        for i in own_inputs.iter() {
            let weight_offset = *i as usize * HL_HALF_NODES;
            for (n, hidden_node) in own_hidden_values.iter_mut().enumerate() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset + n) };
            }
        }
        for hidden_node in own_hidden_values.iter_mut() {
            *hidden_node = relu(*hidden_node);
        }

        for i in opp_inputs.iter() {
            let weight_offset = *i as usize * HL_HALF_NODES;
            for (n, hidden_node) in opp_hidden_values.iter_mut().enumerate() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset + n) };
            }
        }
        for hidden_node in opp_hidden_values.iter_mut() {
            *hidden_node = relu(*hidden_node);
        }

        own_hidden_values
            .iter()
            .chain(opp_hidden_values.iter())
            .zip(self.w.hidden_weights.0.iter())
            .map(|(v, w)| *v * *w)
            .sum::<f32>()
    }

    pub fn save_raw(&self, id: &str) {
        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8('V' as i8).unwrap();

        writer.write_i8(1).unwrap(); // Number of hidden layers

        write_layer(&mut writer, &self.w.input_weights.0, &self.w.hidden_biases.0).expect("Could not write layer");

        write_layer(&mut writer, &self.w.hidden_weights.0, &[0f32]).expect("Could not write layer");
    }

    pub fn save_quantized(&self, net_id: &str) {
        let in_file = &format!("./data/nets/velvet_{}.nn", net_id);
        let out_file = &format!("./data/nets/velvet_{}.qnn", net_id);

        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open NN file: {}", in_file));
        let mut reader = BufReader::new(file);

        read_header(&mut reader, 1).expect("could not read NN header data");

        let input = read_layer(&mut reader).expect("could not read NN input layer");
        let hidden = read_layer(&mut reader).expect("could not read NN hidden layer");

        // Weights grouped by hidden layer target node
        let mut input_weights = Vec::with_capacity(input.weights.len());
        for i in 0..INPUTS {
            input.weights.iter().skip(i).step_by(INPUTS).for_each(|&w| input_weights.push(w));
        }

        let mut max_sum = f32::MIN;
        let mut min_sum = f32::MAX;

        let min_bias = input.biases.iter().min_by(|a, b| b.partial_cmp(a).unwrap()).unwrap();
        let max_bias = input.biases.iter().max_by(|a, b| b.partial_cmp(a).unwrap()).unwrap();

        for weights in input_weights.chunks(HL_HALF_NODES) {
            let max_combo_sum =
                weights.iter().sorted_by(|a, b| b.partial_cmp(a).unwrap()).take(32).sum::<f32>() + max_bias;
            let min_combo_sum =
                weights.iter().sorted_by(|a, b| b.partial_cmp(a).unwrap()).rev().take(32).sum::<f32>() + min_bias;

            min_sum = min_sum.min(min_combo_sum);
            max_sum = max_sum.max(max_combo_sum);
        }

        println!("Min Combined Input Weight: {}, max combined input weight: {}", min_sum, max_sum);

        let bound = if -min_sum > max_sum { -min_sum } else { max_sum };

        let precision_bits = 15 - ((32767.0 / bound) as i16 | 1).leading_zeros();
        println!("Input weights precision bits: {}", precision_bits);

        let max_weight = *input_weights.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let min_weight = *input_weights.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        println!("Input Weight: min: {}, max: {}", min_weight, max_weight);
        println!("Input Biases: min: {}, max: {}", min_bias, max_bias);

        let hidden_weight_lb = hidden.weights.iter().sorted_by(|a, b| a.partial_cmp(b).unwrap()).take(8).sum::<f32>();
        let hidden_weight_ub = hidden.weights.iter().sorted_by(|a, b| b.partial_cmp(a).unwrap()).take(8).sum::<f32>();

        let hidden_bound = (hidden_weight_ub as f64
            * max_weight as f64
            * (HL_HALF_NODES as f64 * 2.0)
            * ((1 << precision_bits) as f64))
            .max(
                hidden_weight_lb as f64
                    * min_weight as f64
                    * (HL_HALF_NODES as f64 * 2.0)
                    * ((1 << precision_bits) as f64),
            );
        println!("Hidden bound: {}", hidden_bound);
        let hidden_multiplier = (i32::MAX as f64 / hidden_bound) as i16;
        println!("Hidden bound multiplier: {}", hidden_multiplier);

        let file = File::create(out_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        let input_multiplier = (32767.0 / bound) as i16;
        println!("Input weight multiplier: {}", input_multiplier);
        writer.write_i16::<LittleEndian>(input_multiplier).unwrap();

        println!("Hidden multiplier: {}", hidden_multiplier);
        writer.write_i16::<LittleEndian>(hidden_multiplier).unwrap();

        print!("Input weights: ");
        write_quantized(&mut writer, 1 << precision_bits, input.weights)
            .expect("Could not write quantized input weights");
        print!("Input biases: ");
        write_quantized(&mut writer, 1 << precision_bits, input.biases)
            .expect("Could not write quantized input biases");

        print!("Hidden weights: ");
        write_quantized(&mut writer, hidden_multiplier, hidden.weights)
            .expect("Could not write quantized hidden layer weights");

        writer.write_i16::<LittleEndian>(0).expect("Could not write quantized hidden bias");
    }
}

fn write_layer(writer: &mut BufWriter<File>, weights: &[f32], biases: &[f32]) -> Result<(), Error> {
    let size = weights.len();
    let in_count = size / biases.len();
    let out_count = biases.len();
    println!("In: {}, Out: {}", in_count, out_count);

    writer.write_i32::<LittleEndian>(out_count as i32)?;
    writer.write_i32::<LittleEndian>(in_count as i32)?;

    for &weight in weights.iter() {
        writer.write_f32::<LittleEndian>(weight)?;
    }

    for &bias in biases.iter() {
        writer.write_f32::<LittleEndian>(bias)?;
    }

    Ok(())
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

fn write_quantized(writer: &mut dyn Write, multiplier: i16, values: Vec<f32>) -> Result<(), Error> {
    writer.write_u32::<LittleEndian>(values.len() as u32)?;

    let values = values.iter().map(|v| (v * multiplier as f32) as i16).collect_vec();

    let mut unused_values = HashSet::<i16>::from_iter(i16::MIN..=i16::MAX);
    for v in values.iter() {
        unused_values.remove(v);
    }

    let rep_zero_marker = unused_values.iter().copied().next().expect("No free value as marker available!");
    writer.write_i16::<LittleEndian>(rep_zero_marker)?;

    let mut outputs = Vec::with_capacity(values.len());
    let mut index = 0;

    while index < values.len() {
        let value = values[index];
        if let Some(repetitions) = find_zero_repetitions(value, &values[index..]) {
            index += repetitions as usize;
            outputs.push(rep_zero_marker);
            outputs.push((repetitions as i32 - 32768) as i16);
            continue;
        }

        outputs.push(value);
        index += 1;
    }

    let codes = CodeBook::from_values(&mut outputs);
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

fn sigmoid(v: f32, scale: f64) -> f32 {
    1.0 / (1.0 + (-v as f64 * scale).exp()) as f32
}

fn relu(v: f32) -> f32 {
    v.max(0.0)
}

fn relu_deriv(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn init_randomly(weights: &mut [f32], nodes: usize) {
    let range = (1.0 / (nodes as f32).sqrt()).min(MAX_WEIGHT);
    let distribution = Uniform::from(-range..range);
    let mut rng = ThreadRng::default();

    for weight in weights.iter_mut() {
        *weight = distribution.sample(&mut rng);
    }
}
