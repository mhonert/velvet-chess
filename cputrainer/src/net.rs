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

use crate::sets::DataSample;
use itertools::Itertools;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Write};
use std::mem::MaybeUninit;
use traincommon::sets::K;
use velvet::nn::io::{BitWriter, CodeBook, read_f32, read_quantized, read_u16, read_u8, write_i16, write_u32, write_u8};
use velvet::nn::{HL1_HALF_NODES, HL1_NODES, INPUT_WEIGHT_COUNT, SCORE_SCALE, piece_idx, king_bucket, PIECE_BUCKETS, KING_BUCKETS, MAX_RELU, FP_IN_PRECISION_BITS, FP_OUT_PRECISION_BITS, FP_IN_MULTIPLIER, FP_OUT_MULTIPLIER};
use velvet::pieces::P;

const K_DIV: f64 = K / (400.0 / SCORE_SCALE as f64);

#[derive(Copy, Clone)]
#[repr(align(32))]
pub struct A32<T>(pub T); // Wrapper to ensure 32 Byte alignment of the wrapped type (e.g. for SIMD load/store instructions)

#[derive(Copy, Clone, Debug, Default)]
pub struct NetworkStats {
    i_max: f32,
    h_max: f32,
    // o_max: f32,
}

impl NetworkStats {
    pub fn max(self, other: &NetworkStats) -> NetworkStats {
        NetworkStats {
            i_max: self.i_max.max(other.i_max),
            h_max: self.h_max.max(other.h_max),
            // o_max: self.o_max.max(other.o_max),
        }
    }
}

#[derive(Clone)]
pub struct Network {
    w: NetWeights,
}

#[derive(Copy, Clone)]
struct NetWeights {
    in_to_h1_weights: A32<[f32; INPUT_WEIGHT_COUNT]>,
    h1_biases: A32<[f32; HL1_NODES]>,

    // h1_to_ml_own_weights: A32<[f32; HL1_ML_CHUNKS * ML_NODES]>,
    // h1_to_ml_opp_weights: A32<[f32; HL1_ML_CHUNKS * ML_NODES]>,
    // ml_biases: A32<[f32; ML_NODES]>,

    // ml_to_h2_weights: A32<[f32; ML_NODES * HL2_NODES]>,
    // h2_biases: A32<[f32; HL2_NODES]>,

    h1_to_out_weights: A32<[f32; HL1_NODES]>,
    out_bias: f32,
}

impl Network {
    pub fn new() -> Box<Network> {
        Box::new(unsafe { MaybeUninit::zeroed().assume_init() })
    }

    pub fn test(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
    ) -> f64 {
        // Error calculation
        let out = self.forward(sample, white_hidden_values, black_hidden_values);
        ((sigmoid(sample.result, K_DIV) - sigmoid(out, K_DIV)) as f64).powi(2)
    }

    pub fn stats(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
    ) -> NetworkStats {
        let mut stats = NetworkStats::default();

        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut white_hidden_values.0, &sample.bpov_inputs, &mut black_hidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut black_hidden_values.0, &sample.wpov_inputs, &mut white_hidden_values.0)
        };

        // Inputs to 1st hidden layer
        own_hidden_values.copy_from_slice(&self.w.h1_biases.0[0..HL1_HALF_NODES]);
        for i in own_inputs.iter() {
            let mut weight_offset = *i as usize * HL1_HALF_NODES;
            for hidden_node in own_hidden_values.iter_mut() {
                let w = unsafe { *self.w.in_to_h1_weights.0.get_unchecked(weight_offset) };
                stats.i_max = stats.i_max.max(w);
                stats.i_max = stats.i_max.max(*hidden_node);
                *hidden_node += w;
                stats.i_max = stats.i_max.max(*hidden_node);
                weight_offset += 1;
            }
        }

        opp_hidden_values.copy_from_slice(&self.w.h1_biases.0[HL1_HALF_NODES..]);
        for i in opp_inputs.iter() {
            let mut weight_offset = *i as usize * HL1_HALF_NODES;
            for hidden_node in opp_hidden_values.iter_mut() {
                let w = unsafe { *self.w.in_to_h1_weights.0.get_unchecked(weight_offset) };
                stats.i_max = stats.i_max.max(w);
                stats.i_max = stats.i_max.max(*hidden_node);
                *hidden_node += w;
                stats.i_max = stats.i_max.max(*hidden_node);
                weight_offset += 1;
            }
        }

        // 1st hidden layer to output layer
        let mut acc = 0.0;

        for (&n, &w) in own_hidden_values.iter().zip(self.w.h1_to_out_weights.0.iter().take(HL1_HALF_NODES)) {
            let requ_n = requ(n);
            stats.h_max = stats.h_max.max(requ_n);
            acc += w * requ_n;
            stats.h_max = stats.h_max.max(acc);
        }

        for (&n, &w) in opp_hidden_values.iter().zip(self.w.h1_to_out_weights.0.iter().skip(HL1_HALF_NODES)) {
            let requ_n = requ(n);
            stats.h_max = stats.h_max.max(requ_n);
            acc += w * requ_n;
            stats.h_max = stats.h_max.max(acc);
        }
        // stats.o_max = stats.o_max.max(out);

        stats
    }

    fn forward(
        &self, sample: &DataSample, white_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A32<[f32; HL1_HALF_NODES]>,
    ) -> f32 {
        let (own_inputs, own_hidden_values, opp_inputs, opp_hidden_values) = if sample.wtm {
            (&sample.wpov_inputs, &mut white_hidden_values.0, &sample.bpov_inputs, &mut black_hidden_values.0)
        } else {
            (&sample.bpov_inputs, &mut black_hidden_values.0, &sample.wpov_inputs, &mut white_hidden_values.0)
        };

        // Inputs to 1st hidden layer
        own_hidden_values.copy_from_slice(&self.w.h1_biases.0[0..HL1_HALF_NODES]);
        for i in own_inputs.iter() {
            let mut weight_offset = *i as usize * HL1_HALF_NODES;
            for hidden_node in own_hidden_values.iter_mut() {
                let w = unsafe { *self.w.in_to_h1_weights.0.get_unchecked(weight_offset) };
                *hidden_node += w;
                weight_offset += 1;
            }
        }

        opp_hidden_values.copy_from_slice(&self.w.h1_biases.0[HL1_HALF_NODES..]);
        for i in opp_inputs.iter() {
            let mut weight_offset = *i as usize * HL1_HALF_NODES;
            for hidden_node in opp_hidden_values.iter_mut() {
                let w = unsafe { *self.w.in_to_h1_weights.0.get_unchecked(weight_offset) };
                *hidden_node += w;
                weight_offset += 1;
            }
        }

        // 1st hidden layer to output layer
        let mut acc = 0.0;

        for (&n, &w) in own_hidden_values.iter().zip(self.w.h1_to_out_weights.0.iter().take(HL1_HALF_NODES)) {
            let requ_n = requ(n);
            acc += w * requ_n;
        }

        for (&n, &w) in opp_hidden_values.iter().zip(self.w.h1_to_out_weights.0.iter().skip(HL1_HALF_NODES)) {
            let requ_n = requ(n);
            acc += w * requ_n;
        }

        acc + self.w.out_bias
    }

    pub fn save_quantized(&self, stats: &NetworkStats, out_file: &str) {
        let i_precision_bits = 15 - ((32767.0 / stats.i_max) as i16 | 1).leading_zeros();
        println!("IN precision bits: {} ({})", i_precision_bits, stats.i_max);

        if i_precision_bits < FP_IN_PRECISION_BITS as u32 {
            panic!("Requires input precision of at least {} bits!", FP_IN_PRECISION_BITS)
        }

        let out_precision_bits = 15 - ((32767.0 / stats.h_max) as i16 | 1).leading_zeros();
        println!("HI precision bits: {} ({})", out_precision_bits, stats.h_max);

        if out_precision_bits < FP_OUT_PRECISION_BITS as u32 {
            panic!("Requires output precision of at least {} bits!", FP_OUT_PRECISION_BITS)
        }

        let file = File::create(out_file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        write_u8(&mut writer, FP_IN_PRECISION_BITS).unwrap();
        write_u8(&mut writer, FP_OUT_PRECISION_BITS).unwrap();

        write_quantized(&mut writer, FP_IN_MULTIPLIER as i16, &self.w.in_to_h1_weights.0).expect("Could not write quantized input to h1 weights");
        write_quantized(&mut writer, FP_IN_MULTIPLIER as i16, &self.w.h1_biases.0).expect("Could not write quantized h1 biases");

        write_quantized(&mut writer, FP_OUT_MULTIPLIER as i16, &self.w.h1_to_out_weights.0)
            .expect("Could not write quantized ml to output weights");
        write_quantized(&mut writer, FP_OUT_MULTIPLIER as i16, &[self.w.out_bias])
            .expect("Could not write quantized output bias");
    }

    pub fn init_from_raw_file(&mut self, in_file: &str) {
        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open NN file: {}", in_file));
        let mut reader = BufReader::new(file);

        read_header(&mut reader, 1).expect("could not read NN header data");

        let input = read_input_layer(&mut reader).expect("could not read NN input layer");
        let out = read_layer(&mut reader).expect("could not read NN output layer");

        self.w.in_to_h1_weights.0.copy_from_slice(&input.weights);
        self.w.h1_biases.0.copy_from_slice(&input.biases);

        self.w.h1_to_out_weights.0.copy_from_slice(&out.weights);
        self.w.out_bias = out.biases[0];
    }

    pub fn init_from_qnn_file(&mut self, in_file: &str) {
        let file = File::open(in_file).unwrap_or_else(|_| panic!("Could not open NN file: {}", in_file));
        let mut reader = BufReader::new(file);

        let in_precision_bits = read_u8(&mut reader).expect("Could not read fixed point precision bits");
        let in_precision = (1 << in_precision_bits) as f32;

        let out_precision_bits = read_u8(&mut reader).expect("Could not read fixed point precision bits");
        let out_precision = (1 << out_precision_bits) as f32;

        let mut in_to_h1_weights = Vec::with_capacity(INPUT_WEIGHT_COUNT);
        fill_to_capacity(&mut in_to_h1_weights);
        let mut h1_biases = Vec::with_capacity(HL1_NODES);
        fill_to_capacity(&mut h1_biases);

        // let mut h1_to_ml_own_weights= Vec::with_capacity(HL1_ML_CHUNKS * ML_NODES);
        // fill_to_capacity(&mut h1_to_ml_own_weights);
        // let mut h1_to_ml_opp_weights= Vec::with_capacity(HL1_ML_CHUNKS * ML_NODES);
        // fill_to_capacity(&mut h1_to_ml_opp_weights);
        // let mut ml_biases= Vec::with_capacity(ML_NODES);
        // fill_to_capacity(&mut ml_biases);

        // let mut ml_to_h2_weights= Vec::with_capacity(ML_NODES * HL2_NODES);
        // fill_to_capacity(&mut ml_to_h2_weights);
        // let mut h2_biases = Vec::with_capacity(HL2_NODES);
        // fill_to_capacity(&mut h2_biases);

        let mut h1_to_out_weights= Vec::with_capacity(HL1_NODES);
        fill_to_capacity(&mut h1_to_out_weights);
        let mut out_biases = Vec::with_capacity(1);
        fill_to_capacity(&mut out_biases);

        read_quantized(&mut reader, &mut in_to_h1_weights).expect("Could not read weights");
        read_quantized(&mut reader, &mut h1_biases).expect("Could not read biases");

        read_quantized(&mut reader, &mut h1_to_out_weights).expect("Could not read weights");
        read_quantized(&mut reader, &mut out_biases).expect("Could not read biases");

        decode_quantized(&mut self.w.in_to_h1_weights.0, &in_to_h1_weights, in_precision);
        decode_quantized(&mut self.w.h1_biases.0, &h1_biases, in_precision);

        decode_quantized(&mut self.w.h1_to_out_weights.0, &h1_to_out_weights, out_precision);
        self.w.out_bias = out_biases[0] as f32 / out_precision;
    }

    pub fn zero_unused_weights(&mut self) {
        const PIECE_SKIP_MASK: [[bool; 6]; PIECE_BUCKETS] = [
            // 0 -> all pieces presents
            [false, false, false, false, false, false],
            // 1 -> no queens
            [false, false, false, false, true, false],
            // 2 -> no queens, no rooks
            [false, false, false, true, true, false],
        ];

        const BUCKET_SIZE: usize = 6 * 64 * 2;

        for pb in 0..PIECE_BUCKETS {
            for kb in 0..KING_BUCKETS {
                let bucket: usize = pb * KING_BUCKETS + kb;
                let offset = BUCKET_SIZE * bucket;

                for piece in 1..=5 {
                    if piece != 1  && !PIECE_SKIP_MASK[pb][piece_idx(piece) as usize] {
                        continue;
                    }
                    for pos in 0..64 {
                        if piece == P && (pos >= 8 || pos <= 55) {
                            continue;
                        }
                        let base_index = piece_idx(piece) as usize * 64 * 2;

                        let idx = (offset + base_index + pos) * HL1_HALF_NODES;
                        for i in idx..(idx + HL1_HALF_NODES) {
                            self.w.in_to_h1_weights.0[i] = 0.0;
                        }
                        const OPP_OFFSET: usize = 64;
                        let idx = (offset + base_index + pos + OPP_OFFSET) * HL1_HALF_NODES;
                        for i in idx..(idx + HL1_HALF_NODES) {
                            self.w.in_to_h1_weights.0[i] = 0.0;
                        }
                    }
                }

                for pos in 0..64 {
                    if king_bucket(pos) == kb as u16 {
                        continue;
                    }
                    let base_index = piece_idx(6) as usize * 64 * 2;

                    let idx = (offset + base_index + pos as usize) * HL1_HALF_NODES;
                    for i in idx..(idx + HL1_HALF_NODES) {
                        self.w.in_to_h1_weights.0[i] = 0.0;
                    }
                }

            }
        }
    }
}

fn decode_quantized(targets: &mut [f32], sources: &[i16], scale: f32) {
    for (target, &source) in targets.iter_mut().zip(sources.iter()) {
        *target = source as f32 / scale;
    }
}

fn fill_to_capacity(v: &mut Vec<i16>) {
    for _ in 0..v.capacity() {
        v.push(0);
    }
}

pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

fn read_header(reader: &mut BufReader<File>, expected_hl_count: usize) -> Result<(), Error> {
    let head = read_u8(reader)?;
    if head != b'V' {
        return Err(Error::new(ErrorKind::InvalidData, "missing header 'V'"));
    }

    let hidden_layer_count = read_u8(reader)? as usize;
    if hidden_layer_count != expected_hl_count {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("hidden layer count mismatch: expected {}, got {}", expected_hl_count, hidden_layer_count),
        ));
    }

    Ok(())
}

fn read_input_layer(reader: &mut BufReader<File>) -> Result<Layer, Error> {
    let out_count = read_u16(reader)? as usize;
    let in_count = read_u16(reader)? as usize;

    let mut weights = Vec::with_capacity(out_count * in_count);
    for _ in 0..(out_count * in_count) {
        let v = read_f32(reader)?;
        weights.push(v);
    }

    let mut input_weights = Vec::with_capacity(weights.len());
    for i in 0..in_count {
        weights.iter().skip(i).step_by(in_count).for_each(|&w| input_weights.push(w));
    }

    let bias_count = read_u16(reader)? as usize;
    assert_eq!(out_count * 2, bias_count, "bias count mismatch");

    let mut biases = Vec::with_capacity(out_count * 2);
    for _ in 0..out_count * 2 {
        let v = read_f32(reader)?;

        biases.push(v);
    }

    Ok(Layer { weights: input_weights, biases })
}

fn read_layer(reader: &mut BufReader<File>) -> Result<Layer, Error> {
    let out_count = read_u16(reader)? as usize;
    let in_count = read_u16(reader)? as usize;

    let mut unoriented_weights = Vec::with_capacity(out_count * in_count);

    for _ in 0..(out_count * in_count) {
        let v = read_f32(reader)?;
        unoriented_weights.push(v);
    }

    let mut weights = Vec::with_capacity(unoriented_weights.len());
    for i in 0..in_count {
        unoriented_weights.iter().skip(i).step_by(in_count).for_each(|&w| weights.push(w));
    }

    let bias_count = read_u16(reader)? as usize;
    assert_eq!(out_count, bias_count, "bias count mismatch");

    let mut biases = Vec::with_capacity(out_count);
    for _ in 0..out_count {
        let v = read_f32(reader)?;

        biases.push(v);
    }

    Ok(Layer { weights, biases })
}

fn write_quantized(writer: &mut dyn Write, multiplier: i16, values: &[f32]) -> Result<(), Error> {
    write_u32(writer, values.len() as u32)?;

    let ivalues = values.iter().map(|v| (v * multiplier as f32) as i32).collect_vec();
    let mut values = Vec::with_capacity(ivalues.len());

    let mut unused_values = HashSet::<i16>::from_iter(i16::MIN..=i16::MAX);
    for &v in ivalues.iter() {
        assert!(v >= i16::MIN as i32, "quantized value below i16 min bound");
        assert!(v <= i16::MAX as i32, "quantized value above i16 max bound");

        values.push(v as i16);
        unused_values.remove(&(v as i16));
    }

    let rep_zero_marker = unused_values.iter().copied().next().expect("No free value as marker available!");
    write_i16(writer, rep_zero_marker)?;

    let mut outputs = Vec::with_capacity(values.len());
    let mut index = 0;

    while index < values.len() {
        let value = values[index] as i16;
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

// Rectified Quadratic Units (ReQU)
#[inline(always)]
fn requ(v: f32) -> f32 {
    v.clamp(0.0, MAX_RELU).powi(2)
}
