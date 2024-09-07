/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Write};
use std::mem::MaybeUninit;
use traincommon::sets::K;
use velvet::nn::io::{BitWriter, read_f32, read_quantized, read_u16, read_u8, write_u32, write_u8};
use velvet::nn::{HL1_HALF_NODES, HL1_NODES, INPUT_WEIGHT_COUNT, SCORE_SCALE, piece_idx, king_bucket, MAX_RELU, FP_IN_PRECISION_BITS, FP_OUT_PRECISION_BITS, FP_IN_MULTIPLIER, FP_OUT_MULTIPLIER, BUCKETS};

const K_DIV: f64 = K / (400.0 / SCORE_SCALE as f64);

#[derive(Copy, Clone)]
#[repr(align(32))]
pub struct A64<T>(pub T); // Wrapper to ensure 32 Byte alignment of the wrapped type (e.g. for SIMD load/store instructions)

#[derive(Copy, Clone, Debug, Default)]
pub struct NetworkStats {
    i_max: f32,
    h_max: f32,
}

impl NetworkStats {
    pub fn max(self, other: &NetworkStats) -> NetworkStats {
        NetworkStats {
            i_max: self.i_max.max(other.i_max),
            h_max: self.h_max.max(other.h_max),
        }
    }
}

#[derive(Clone)]
pub struct Network {
    w: NetWeights,
}

#[derive(Copy, Clone)]
struct NetWeights {
    in_to_h1_weights: A64<[f32; INPUT_WEIGHT_COUNT]>,
    h1_biases: A64<[f32; HL1_NODES]>,

    h1_to_out_weights: A64<[f32; HL1_NODES]>,
    out_bias: f32,
}

impl Network {
    pub fn new() -> Box<Network> {
        Box::new(unsafe { MaybeUninit::zeroed().assume_init() })
    }

    pub fn test(
        &self, sample: &DataSample, white_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
    ) -> f64 {
        // Error calculation
        let out = self.forward(sample, white_hidden_values, black_hidden_values);
        ((sigmoid(sample.result, K_DIV) - sigmoid(out, K_DIV)) as f64).abs().powf(2.6)
    }

    pub fn stats(
        &self, sample: &DataSample, white_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
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

        stats
    }

    fn forward(
        &self, sample: &DataSample, white_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
        black_hidden_values: &mut A64<[f32; HL1_HALF_NODES]>,
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
        const BUCKET_SIZE: usize = 6 * 64 * 2;
        const BASE_OFFSET: usize = 0;

        for bucket in 0..BUCKETS {
            let offset = BASE_OFFSET + BUCKET_SIZE * bucket;

            for pos in 0..64 {
                if king_bucket(pos) == bucket as u16 {
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

    let quant_values: Vec<i32> = values.iter().map(|v| ((*v as f64) * multiplier as f64) as i32).collect();
    let mut values = Vec::with_capacity(quant_values.len());

    let mut counts = [0usize; 256];
    for &v in quant_values.iter() {
        assert!(v >= i16::MIN as i32, "quantized value below i16 min bound");
        assert!(v <= i16::MAX as i32, "quantized value above i16 max bound");

        values.push(v as i16);

        let idx = (v as u16 >> 8) as usize;
        counts[idx] += 1;
    }

    let mut entries: Vec<(usize, usize)> = counts.iter().copied().enumerate().filter(|(_, count)| *count > 0).collect();

    entries.sort_by_key(|e| e.1);
    entries.reverse();

    let mut code_book = [0; 256];
    let mut max_code: u8 = 0;

    write_u8(writer, entries.len() as u8)?;
    for (next_code_idx, &(idx, _)) in (0_u8..).zip(entries.iter()) {
        max_code = next_code_idx;
        write_u8(writer, idx as u8)?;
        write_u8(writer, next_code_idx)?;
        code_book[idx] = next_code_idx;
    }

    max_code = (max_code as i8 - (1 + 3)).max(3) as u8;
    let max_bits = 8 - (max_code | 1).leading_zeros();
    println!("Max code: {}, max bits: {}", max_code, max_bits);

    let mut index = 0;

    let mut bit_writer = BitWriter::default();
    while index < values.len() {
        let value = values[index];
        if let Some(repetitions) = find_zero_repetitions(value, &values[index..]) {
            index += repetitions as usize;

            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 6, repetitions - 1)?;

            continue;
        }

        let hi_code = code_book[(value as u16 >> 8) as usize];
        if hi_code <= 1 {
            bit_writer.write(writer, 1, 0)?;
            bit_writer.write(writer, 1, hi_code as u32)?;
        } else if (hi_code - 1) <= 3 {
            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 1, 0)?;
            bit_writer.write(writer, 2, hi_code as u32 - 1)?;
        } else {
            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 1, 1)?;
            bit_writer.write(writer, 1, 0)?;
            bit_writer.write(writer, max_bits as usize, hi_code as u32 - (1 + 3))?;
        }

        let lo_value = value as u16 & 0xFF;
        bit_writer.write(writer, 8, lo_value as u32)?;

        index += 1;
    }

    bit_writer.flush(writer)?;

    Ok(())
}

fn find_zero_repetitions(curr_value: i16, values: &[i16]) -> Option<u32> {
    if curr_value != 0 {
        return None;
    }

    let reps = values.iter().take(64).take_while(|&v| *v == 0).count() as u32;
    Some(reps)
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
