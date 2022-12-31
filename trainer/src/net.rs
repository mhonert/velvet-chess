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
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Write};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use velvet::nn::io::{BitWriter, CodeBook};
use velvet::nn::{HL_HALF_NODES, HL_NODES, INPUTS, INPUT_WEIGHT_COUNT};

const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;

const WEIGHT_DECAY: f64 = 0.001;

const K: f64 = 1.603;
const K_DIV: f64 = K / 400.0;

const MAX_WEIGHT_INPUTS: f32 = 240.0 / (1 << 6) as f32;
const MIN_WEIGHT_INPUTS: f32 = -MAX_WEIGHT_INPUTS;

const MAX_WEIGHT_HIDDEN: f32 = 160.0 / (1 << 6) as f32;
const MIN_WEIGHT_HIDDEN: f32 = -MAX_WEIGHT_HIDDEN;

pub struct InputGradients {
    updated: [AtomicU32; INPUTS],
    values: SharedGradients<INPUT_WEIGHT_COUNT>,
}

impl Default for InputGradients {
    fn default() -> Self {
        unsafe { MaybeUninit::zeroed().assume_init() }
    }
}

impl InputGradients {
    pub fn mark_updated(&self, idx: usize) {
        unsafe { self.updated.get_unchecked(idx) }.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn add(&self, idx: usize, gradient: f32) {
        self.values.add(idx, gradient);
    }
}

#[derive(Default)]
pub struct HiddenGradients {
    iweights: Gradients<HL_NODES>,
    ibias: Gradients<HL_NODES>,
    output_bias: Gradients<1>,
}

impl HiddenGradients {
    pub fn reset(&mut self) {
        self.iweights.reset();
        self.ibias.reset();
        self.output_bias.reset();
    }

    pub fn add_all(&mut self, other: &HiddenGradients) {
        self.iweights.add_all(&other.iweights);
        self.ibias.add_all(&other.ibias);
        self.output_bias.add_all(&other.output_bias);
    }
}

struct SharedGradients<const N: usize>([AtomicU64; N]);

impl<const N: usize> Default for SharedGradients<N> {
    fn default() -> Self {
        SharedGradients(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl<const N: usize> SharedGradients<N> {
    #[inline(always)]
    pub fn add(&self, idx: usize, gradient: f32) {
        let v: &AtomicU64 = unsafe { self.0.get_unchecked(idx) };

        // Note: overall not an atomic operation!
        let curr = v.load(Ordering::Relaxed);
        let new = (f64::from_bits(curr) + gradient as f64).to_bits();
        v.store(new, Ordering::Relaxed);
    }
}

struct Gradients<const N: usize>([f64; N]);

impl<const N: usize> Default for Gradients<N> {
    fn default() -> Self {
        Gradients(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

impl<const N: usize> Gradients<N> {
    #[inline(always)]
    pub fn add(&mut self, idx: usize, gradient: f64) {
        unsafe { *self.0.get_unchecked_mut(idx) += gradient };
    }

    pub fn reset(&mut self) {
        self.0.fill(0f64);
    }

    pub fn add_all(&mut self, other: &Gradients<N>) {
        for (s, o) in self.0.iter_mut().zip(other.0.iter()) {
            *s += o;
        }
    }
}

#[inline(always)]
pub fn update_weight_hidden(
    weight: &mut f32, momentum: &mut (f64, f64), gradient: f64, lr: f64, iteration: usize, scale: f64,
) {
    let gradient = gradient * scale;

    momentum.0 = BETA1 * momentum.0 + (1.0 - BETA1) * gradient;
    momentum.1 = BETA2 * momentum.1 + (1.0 - BETA2) * gradient * gradient;

    let lr = lr * (1.0 - momentum.1.powi(iteration as i32).sqrt());
    let delta = momentum.0 / (momentum.1.sqrt() + f64::EPSILON);
    *weight -= (lr * (delta + WEIGHT_DECAY * *weight as f64)) as f32;
    *weight = weight.clamp(MIN_WEIGHT_HIDDEN, MAX_WEIGHT_HIDDEN);
}

#[inline(always)]
pub fn update_weight_input(
    weight: &mut f32, momentum: &mut (f64, f64), gradient: &AtomicU64, lr: f64, iteration: usize, scale: f64,
) {
    let gradient = f64::from_bits(gradient.load(Ordering::Relaxed)) * scale;

    momentum.0 = BETA1 * momentum.0 + (1.0 - BETA1) * gradient;
    momentum.1 = BETA2 * momentum.1 + (1.0 - BETA2) * gradient * gradient;

    let lr = lr * (1.0 - momentum.1.powi(iteration as i32).sqrt());
    let delta = momentum.0 / (momentum.1.sqrt() + f64::EPSILON);
    *weight -= (lr * (delta + WEIGHT_DECAY * *weight as f64)) as f32;
    *weight = weight.clamp(MIN_WEIGHT_INPUTS, MAX_WEIGHT_INPUTS);
}

pub struct NetMomentums {
    input_weights: [(f64, f64); INPUT_WEIGHT_COUNT],
    ihidden_bias: [(f64, f64); HL_NODES],
    ihidden_weights: [(f64, f64); HL_NODES],
    output_bias: (f64, f64),
}

impl NetMomentums {
    pub fn new() -> Box<Self> {
        Box::new(unsafe { MaybeUninit::zeroed().assume_init() })
    }
}

#[derive(Copy, Clone)]
#[repr(align(32))]
pub struct A32<T>(pub T); // Wrapper to ensure 32 Byte alignment of the wrapped type (e.g. for SIMD load/store instructions)

#[derive(Copy, Clone, Debug, Default)]
pub struct NetworkStats {
    input_max: f32,
    hidden_max: f32,
    output_max: f32,
}

impl NetworkStats {
    pub fn max(self, other: &NetworkStats) -> NetworkStats {
        NetworkStats {
            input_max: self.input_max.max(other.input_max),
            hidden_max: self.hidden_max.max(other.hidden_max),
            output_max: self.output_max.max(other.output_max),
        }
    }
}

#[derive(Clone)]
pub struct Network {
    w: NetWeights,
}

#[derive(Copy, Clone)]
struct NetWeights {
    input_weights: A32<[f32; INPUT_WEIGHT_COUNT]>,
    ihidden_biases: A32<[f32; HL_NODES]>,
    ihidden_weights: A32<[f32; HL_NODES]>,
    output_bias: f32,
}

impl Network {
    pub fn new() -> Box<Network> {
        let mut nn: Box<Network> = Box::new(unsafe { MaybeUninit::zeroed().assume_init() });

        init_randomly(&mut nn.w.input_weights.0, INPUTS);
        init_randomly(&mut nn.w.ihidden_weights.0, HL_HALF_NODES);

        nn
    }

    pub fn copy_weights(&mut self, other: &Network) {
        self.w.input_weights.0.copy_from_slice(&other.w.input_weights.0);
        self.w.ihidden_weights.0.copy_from_slice(&other.w.ihidden_weights.0);
        self.w.ihidden_biases.0.copy_from_slice(&other.w.ihidden_biases.0);
        self.w.output_bias = other.w.output_bias;
    }

    pub fn quantize(&mut self, input_scale: i16, hidden_scale: i16) {
        for w in self.w.input_weights.0.iter_mut() {
            *w = (((*w as f64 * input_scale as f64) as i32) as f64 / input_scale as f64) as f32;
        }

        for w in self.w.ihidden_weights.0.iter_mut() {
            *w = (((*w as f64 * hidden_scale as f64) as i32) as f64 / hidden_scale as f64) as f32;
        }

        for w in self.w.ihidden_biases.0.iter_mut() {
            *w = (((*w as f64 * input_scale as f64) as i32) as f64 / input_scale as f64) as f32;
        }

        self.w.output_bias =
            (((self.w.output_bias as f64 * hidden_scale as f64) as i32) as f64 / hidden_scale as f64) as f32;
    }

    pub fn zero_check(&self, input_scale: i16, hidden_scale: i16) {
        let mut zero_count = 0;
        for w in self.w.input_weights.0.iter() {
            if ((*w as f64 * input_scale as f64) as i32) == 0 {
                zero_count += 1;
            }
        }
        println!("Input weights: {:2.8}%", zero_count as f64 * 100.0 / self.w.input_weights.0.len() as f64);

        let mut zero_count = 0;
        for w in self.w.ihidden_weights.0.iter() {
            if ((*w as f64 * hidden_scale as f64) as i32) == 0 {
                zero_count += 1;
            }
        }
        println!("Hidden weights: {:2.8}%", zero_count as f64 * 100.0 / self.w.ihidden_weights.0.len() as f64);
    }

    pub fn train(
        &self, sample: &DataSample, input_gradients: &InputGradients, hidden_gradients: &mut HiddenGradients,
        white_ihidden_values: &mut A32<[f32; HL_HALF_NODES]>, black_ihidden_values: &mut A32<[f32; HL_HALF_NODES]>,
    ) -> f32 {
        // Forward pass
        let out = sigmoid(self.forward(sample, white_ihidden_values, black_ihidden_values), 2048.0 * K_DIV);

        // Gradient calculation
        let train = sigmoid(sample.result, 2048.0 * K_DIV);

        let out_error = (out - train).powi(2);
        let out_delta = (out - train) * out * (1.0 - out) * 2048.0 * K_DIV as f32; // Can * 2048 * K_DIV be removed?

        hidden_gradients.output_bias.add(0, out_delta as f64);

        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut white_ihidden_values.0, &sample.bpov_inputs, &mut black_ihidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut black_ihidden_values.0, &sample.wpov_inputs, &mut white_ihidden_values.0)
        };

        let prev_delta = out_delta;

        // STM
        let own_deltas = own_hidden_values
            .iter()
            .zip(hidden_gradients.iweights.0.iter_mut().take(HL_HALF_NODES))
            .zip(hidden_gradients.ibias.0.iter_mut().take(HL_HALF_NODES))
            .zip(self.w.ihidden_weights.0.iter().take(HL_HALF_NODES))
            .map(|(((v, g), b), w)| {
                *g += prev_delta as f64 * *v as f64;
                let delta = prev_delta * *w * relu_deriv(*v);
                *b += delta as f64;
                delta
            })
            .collect_vec();

        // non-STM
        let opp_deltas = opp_hidden_values
            .iter()
            .zip(hidden_gradients.iweights.0.iter_mut().skip(HL_HALF_NODES).take(HL_HALF_NODES))
            .zip(hidden_gradients.ibias.0.iter_mut().skip(HL_HALF_NODES).take(HL_HALF_NODES))
            .zip(self.w.ihidden_weights.0.iter().skip(HL_HALF_NODES).take(HL_HALF_NODES))
            .map(|(((v, g), b), w)| {
                *g += prev_delta as f64 * *v as f64;
                let delta = prev_delta * *w * relu_deriv(*v);
                *b += delta as f64;
                delta
            })
            .collect_vec();

        // Input Layer weights
        for &i in own_inputs.iter() {
            input_gradients.mark_updated(i as usize);
            let offset = i as usize * HL_HALF_NODES;
            for j in 0..HL_HALF_NODES {
                let delta = unsafe { *own_deltas.get_unchecked(j) };
                input_gradients.add(offset + j, delta);
            }
        }

        for &i in opp_inputs.iter() {
            input_gradients.mark_updated(i as usize);
            let offset = i as usize * HL_HALF_NODES;
            for j in 0..HL_HALF_NODES {
                let delta = unsafe { *opp_deltas.get_unchecked(j) };
                input_gradients.add(offset + j, delta);
            }
        }

        out_error
    }

    #[inline(always)]
    pub fn update_weights(
        &mut self, input_gradients: &InputGradients, hidden_gradients: &HiddenGradients, momentums: &mut NetMomentums,
        lr: f64, iteration: usize, samples: u32,
    ) {
        let scale = 1.0 / samples as f64;
        self.w
            .input_weights
            .0
            .par_chunks_exact_mut(HL_HALF_NODES)
            .zip(momentums.input_weights.par_chunks_exact_mut(HL_HALF_NODES))
            .zip(input_gradients.values.0.par_chunks_exact(HL_HALF_NODES))
            .zip(input_gradients.updated.par_iter())
            .filter(|((_, _), updated)| updated.load(Ordering::Relaxed) > 0)
            .for_each(|(((w, m), g), updated)| {
                let c = updated.load(Ordering::Relaxed);
                let scale = 1.0 / c as f64;
                updated.store(0, Ordering::Relaxed);
                for ((w, m), g) in w.iter_mut().zip(m.iter_mut()).zip(g.iter()) {
                    update_weight_input(w, m, g, lr, iteration, scale);
                    g.store(0, Ordering::Relaxed);
                }
            });

        self.w
            .ihidden_weights
            .0
            .iter_mut()
            .zip(momentums.ihidden_weights.iter_mut())
            .zip(hidden_gradients.iweights.0.iter())
            .for_each(|((w, m), &g)| {
                update_weight_hidden(w, m, g, lr, iteration, scale);
            });

        self.w
            .ihidden_biases
            .0
            .iter_mut()
            .zip(momentums.ihidden_bias.iter_mut())
            .zip(hidden_gradients.ibias.0.iter())
            .for_each(|((w, m), &g)| {
                update_weight_hidden(w, m, g, lr, iteration, scale);
            });

        update_weight_hidden(
            &mut self.w.output_bias,
            &mut momentums.output_bias,
            hidden_gradients.output_bias.0[0],
            lr,
            iteration,
            scale,
        );
    }

    pub fn test(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
    ) -> f64 {
        // Error calculation
        let out = self.forward(sample, white_hidden_values, black_hidden_values);
        ((sigmoid(sample.result, 2048.0 * K_DIV) - sigmoid(out, 2048.0 * K_DIV)) as f64).powi(2)
    }

    pub fn stats(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
    ) -> NetworkStats {
        let mut stats = NetworkStats::default();

        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut white_hidden_values.0, &sample.bpov_inputs, &mut black_hidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut black_hidden_values.0, &sample.wpov_inputs, &mut white_hidden_values.0)
        };

        own_hidden_values.copy_from_slice(&self.w.ihidden_biases.0[0..HL_HALF_NODES]);
        for i in own_inputs.iter() {
            let mut weight_offset = *i as usize * HL_HALF_NODES;
            for hidden_node in own_hidden_values.iter_mut() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset) };
                weight_offset += 1;
                stats.input_max = stats.input_max.max(hidden_node.abs());
            }
        }
        for hidden_node in own_hidden_values.iter_mut() {
            stats.input_max = stats.input_max.max(hidden_node.abs());
            *hidden_node = relu(*hidden_node);
        }

        opp_hidden_values.copy_from_slice(&self.w.ihidden_biases.0[HL_HALF_NODES..]);
        for i in opp_inputs.iter() {
            let mut weight_offset = *i as usize * HL_HALF_NODES;
            for hidden_node in opp_hidden_values.iter_mut() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset) };
                weight_offset += 1;
                stats.input_max = stats.input_max.max(hidden_node.abs());
            }
        }
        for hidden_node in opp_hidden_values.iter_mut() {
            stats.input_max = stats.input_max.max(hidden_node.abs());
            *hidden_node = relu(*hidden_node);
        }

        let mut own_acc: f32 = 0.;
        for (v, w) in own_hidden_values.iter().zip(self.w.ihidden_weights.0.iter().take(HL_HALF_NODES)) {
            own_acc += v * w;
            stats.hidden_max = stats.hidden_max.max(own_acc.abs());
        }
        stats.hidden_max = stats.hidden_max.max(own_acc.abs());

        let mut opp_acc: f32 = 0.;
        for (v, w) in
            opp_hidden_values.iter().zip(self.w.ihidden_weights.0.iter().skip(HL_HALF_NODES).take(HL_HALF_NODES))
        {
            opp_acc += v * w;
            stats.hidden_max = stats.hidden_max.max(opp_acc.abs());
        }
        stats.hidden_max = stats.hidden_max.max(opp_acc.abs());

        let output = own_acc + opp_acc + self.w.output_bias;
        stats.output_max = output;
        stats
    }

    fn forward(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL_HALF_NODES]>,
    ) -> f32 {
        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut white_hidden_values.0, &sample.bpov_inputs, &mut black_hidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut black_hidden_values.0, &sample.wpov_inputs, &mut white_hidden_values.0)
        };

        own_hidden_values.copy_from_slice(&self.w.ihidden_biases.0[0..HL_HALF_NODES]);
        for i in own_inputs.iter() {
            let mut weight_offset = *i as usize * HL_HALF_NODES;
            for hidden_node in own_hidden_values.iter_mut() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset) };
                weight_offset += 1;
            }
        }
        for hidden_node in own_hidden_values.iter_mut() {
            *hidden_node = relu(*hidden_node);
        }

        opp_hidden_values.copy_from_slice(&self.w.ihidden_biases.0[HL_HALF_NODES..]);
        for i in opp_inputs.iter() {
            let mut weight_offset = *i as usize * HL_HALF_NODES;
            for hidden_node in opp_hidden_values.iter_mut() {
                *hidden_node += unsafe { *self.w.input_weights.0.get_unchecked(weight_offset) };
                weight_offset += 1;
            }
        }
        for hidden_node in opp_hidden_values.iter_mut() {
            *hidden_node = relu(*hidden_node);
        }

        let own_acc = own_hidden_values
            .iter()
            .zip(self.w.ihidden_weights.0.iter().take(HL_HALF_NODES))
            .map(|(&v, &w)| v * w)
            .sum::<f32>();
        let opp_acc = opp_hidden_values
            .iter()
            .zip(self.w.ihidden_weights.0.iter().skip(HL_HALF_NODES).take(HL_HALF_NODES))
            .map(|(&v, &w)| v * w)
            .sum::<f32>();

        own_acc + opp_acc + self.w.output_bias
    }

    pub fn save_raw(&self, id: &str) {
        let file = File::create(format!("./data/nets/velvet_{}.nn", id)).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        writer.write_i8('V' as i8).unwrap();

        writer.write_i8(1).unwrap(); // Number of hidden layers

        write_layer(&mut writer, &self.w.input_weights.0, &self.w.ihidden_biases.0).expect("Could not write layer");
        write_layer(&mut writer, &self.w.ihidden_weights.0, &[self.w.output_bias]).expect("Could not write layer");
    }

    pub fn save_quantized(&self, stats: &NetworkStats) -> (i16, i16) {
        let out_file = &format!("./data/nets/velvet_{}.qnn", "final");

        let input_weights = &self.w.input_weights.0;
        let input_biases = &self.w.ihidden_biases.0;

        let max_input_weight = input_weights.iter().max_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap()).unwrap();
        let max_input_bias = input_biases.iter().max_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap()).unwrap();

        let input_bound = max_input_weight.max(*max_input_bias).max(stats.input_max);
        let input_precision_bits = 15 - ((32767.0 / input_bound) as i16 | 1).leading_zeros();
        println!("Input weights precision bits: {}", input_precision_bits);

        let hidden_weights = &self.w.ihidden_weights.0;
        let max_hidden_weight = hidden_weights.iter().max_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap()).unwrap();
        let max_hidden_bias = &self.w.output_bias;

        let hidden_bound = max_hidden_weight.max(*max_hidden_bias).max(stats.hidden_max);
        let hidden_precision_bits = 15 - ((32767.0 / hidden_bound) as i16 | 1).leading_zeros();
        println!("Hidden weights precision bits: {}", hidden_precision_bits);

        let file = File::create(out_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        let input_scale = 1 << input_precision_bits;
        let hidden_scale = 1 << hidden_precision_bits;
        // let input_multiplier = (32767.0 / bound) as i16;
        writer.write_i16::<LittleEndian>(input_scale).unwrap();
        writer.write_i16::<LittleEndian>(hidden_scale).unwrap();

        write_quantized(&mut writer, input_scale, input_weights).expect("Could not write quantized input weights");
        write_quantized(&mut writer, input_scale, input_biases).expect("Could not write quantized input biases");

        write_quantized(&mut writer, hidden_scale, hidden_weights)
            .expect("Could not write quantized hidden layer weights");
        write_quantized(&mut writer, hidden_scale, &[self.w.output_bias])
            .expect("Could not write quantized hidden layer biases");

        writer.write_i16::<LittleEndian>(0).expect("Could not write quantized hidden bias");

        (input_scale, hidden_scale)
    }

    pub fn init_from_base_file(&mut self, in_file: &String) {
        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open NN file: {}", in_file));
        let mut reader = BufReader::new(file);

        read_header(&mut reader, 1).expect("could not read NN header data");

        let input = read_layer(&mut reader).expect("could not read NN input layer");
        let hidden = read_layer(&mut reader).expect("could not read NN hidden layer");

        self.w.input_weights.0.copy_from_slice(&input.weights);
        self.w.ihidden_biases.0.copy_from_slice(&input.biases);
        self.w.ihidden_weights.0.copy_from_slice(&hidden.weights);
        self.w.output_bias = hidden.biases[0];
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

fn write_quantized(writer: &mut dyn Write, multiplier: i16, values: &[f32]) -> Result<(), Error> {
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

#[inline(always)]
fn sigmoid(v: f32, scale: f64) -> f32 {
    1.0 / (1.0 + (-v as f64 * scale).exp()) as f32
}

#[inline(always)]
fn relu(v: f32) -> f32 {
    v.max(0.0)
}

#[inline(always)]
fn relu_deriv(v: f32) -> f32 {
    if v > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn init_randomly(weights: &mut [f32], nodes: usize) {
    let range = (1.55 / nodes as f32).sqrt();
    let distribution = Uniform::from(-range..range);
    let mut rng = ThreadRng::default();

    for weight in weights.iter_mut() {
        *weight = distribution.sample(&mut rng);
    }
}
