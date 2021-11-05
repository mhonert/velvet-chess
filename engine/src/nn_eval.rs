/*
 * Velvet Chess Engine
 * Copyright (C) 2021 mhonert (https://github.com/mhonert)
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

use std::io::BufReader;
use crate::bitboard::{BitBoard, mirror_pos};
use byteorder::{ReadBytesExt, LittleEndian};
use std::cmp::max;
use std::sync::{Arc, Once};
use crate::colors::{Color, WHITE};
use crate::pieces::{Q, R};

// Fixed point number precision
const FP_PRECISION_BITS: i16 = 12;

// NN layer size
const FEATURES_PER_BUCKET: usize = 64 * 6 * 2;
pub const INPUTS: usize = FEATURES_PER_BUCKET * 4;
pub const HL_INPUTS: usize = 64;
pub const HL_COUNT: i8 = 2;
const HL_NODES: usize = 64;

const HL_OUT_NODES: usize = 16;

static INIT_NN_PARAMS: Once = Once::new();
static mut NN_PARAMS: Option<Arc<NeuralNetParams>> = None;

#[derive(Clone)]
#[repr(align(32))]
struct A32<T>(T); // Wrapper to ensure 32 Byte alignment of the wrapped type for SIMD load/store instructions

struct NeuralNetParams {
    psq_weights: A32<[i16; INPUTS]>,
    psq_bias: i16,

    input_weights: A32<[i16; INPUTS * HL_INPUTS]>,
    input_biases: A32<[i16; HL_INPUTS]>,

    hidden1_weights: A32<[i16; HL_INPUTS * HL_NODES]>,
    hidden1_biases: A32<[i16; HL_NODES]>,

    hidden2_weights: A32<[i16; HL_NODES * HL_OUT_NODES]>,
    hidden2_biases: A32<[i16; HL_OUT_NODES]>,

    output_weights: A32<[i16; HL_OUT_NODES]>,
    output_bias: i16,
}

impl NeuralNetParams {
    pub fn new() -> Arc<Self> {
        let mut reader = BufReader::new(&include_bytes!("../nets/velvet.qnn")[..]);

        let precision_bits = reader.read_i8().unwrap() as i16;
        assert_eq!(precision_bits, FP_PRECISION_BITS, "NN has been quantized with a different fixed point precision, expected: {}, got: {}", FP_PRECISION_BITS, precision_bits);

        let mut params = NeuralNetParams::default();
        read_quantized(&mut reader, &mut params.input_weights.0);
        read_quantized(&mut reader, &mut params.input_biases.0);

        read_quantized(&mut reader, &mut params.hidden1_weights.0);
        read_quantized(&mut reader, &mut params.hidden1_biases.0);

        read_quantized(&mut reader, &mut params.hidden2_weights.0);
        read_quantized(&mut reader, &mut params.hidden2_biases.0);

        read_quantized(&mut reader, &mut params.output_weights.0);

        params.output_bias = reader.read_i16::<LittleEndian>().expect("Could not read output bias");

        let mut psq_reader = BufReader::new(&include_bytes!("../nets/velvet_psq.qnn")[..]);
        read_quantized(&mut psq_reader, &mut params.psq_weights.0);

        params.psq_bias = psq_reader.read_i16::<LittleEndian>().expect("Could not read psq bias");

        Arc::new(params)
    }
}

impl Default for NeuralNetParams {
    fn default() -> Self {
        NeuralNetParams{
            psq_weights: A32([0; INPUTS]),
            psq_bias: 0,

            input_weights: A32([0; INPUTS * HL_INPUTS]),
            input_biases: A32([0; HL_INPUTS]),

            hidden1_weights: A32([0; HL_INPUTS * HL_NODES]),
            hidden1_biases: A32([0; HL_NODES]),

            hidden2_weights: A32([0; HL_NODES * HL_OUT_NODES]),
            hidden2_biases: A32([0; HL_OUT_NODES]),

            output_weights: A32([0; HL_OUT_NODES]),
            output_bias: 0,
        }
    }
}

#[derive(Clone)]
pub struct NeuralNetEval {
    params: Arc<NeuralNetParams>,

    hidden1_nodes_wtm: A32<[i16; HL_INPUTS]>, // wtm - white to move
    hidden1_nodes_btm: A32<[i16; HL_INPUTS]>, // btm - black to move
    hidden2_nodes: A32<[i16; HL_NODES]>,
    output_nodes: A32<[i16; HL_OUT_NODES]>,

    active_player: Color,
    bucket: u8,
    offset: usize,

    psq_wtm_score: i16,
    psq_btm_score: i16,

    base_psq_wtm_score: i16,
    base_psq_btm_score: i16,
}

impl NeuralNetEval {

    pub fn new() -> Box<Self> {
        INIT_NN_PARAMS.call_once(|| unsafe { NN_PARAMS = Some(NeuralNetParams::new()) });
        Box::new(NeuralNetEval {
            params: unsafe { NN_PARAMS.clone().unwrap() },

            hidden1_nodes_wtm: A32([0; HL_INPUTS]),
            hidden1_nodes_btm: A32([0; HL_INPUTS]),
            hidden2_nodes: A32([0; HL_NODES]),
            output_nodes: A32([0; HL_OUT_NODES]),

            active_player: WHITE,
            bucket: 0b11,
            offset: 0,

            psq_wtm_score: 0,
            psq_btm_score: 0,

            base_psq_wtm_score: 0,
            base_psq_btm_score: 0,
        })
    }

    pub fn init_pos(&mut self, active_player: Color, bitboards: &[u64; 13]) {
        self.active_player = active_player;

        self.hidden1_nodes_wtm.0.copy_from_slice(&self.params.input_biases.0);
        self.hidden1_nodes_btm.0.copy_from_slice(&self.params.input_biases.0);

        self.bucket = calc_bucket(bitboards);

        self.offset = self.bucket as usize * 768;

        self.psq_wtm_score = self.params.psq_bias;
        self.psq_btm_score = self.params.psq_bias;

        for piece in 1..=6 {
            for pos in BitBoard(bitboards[(piece + 6) as usize]) {
                self.add_piece(pos as usize, piece);
            }

            for pos in BitBoard(bitboards[(-piece + 6) as usize]) {
                self.add_piece(pos as usize, -piece);
            }
        }
    }

    pub fn save_base_scores(&mut self) {
        self.base_psq_wtm_score = self.psq_wtm_score;
        self.base_psq_btm_score = self.psq_btm_score;
    }

    pub fn set_stm(&mut self, active_player: Color) {
        self.active_player = active_player;
    }

    pub fn check_refresh(&mut self, active_player: Color, bitboards: &[u64; 13]) {
        let bucket = calc_bucket(bitboards);
        if bucket != self.bucket {
            self.init_pos(active_player, bitboards);
        }
    }

    pub fn add_piece(&mut self, pos: usize, piece: i8) {
        let base_index = self.offset + ((piece.abs() as usize - 1) * 2) as usize * 64;

        let mut idx = base_index + pos;
        if piece < 0 {
            idx += 64;
        }

        self.psq_wtm_score += unsafe { self.params.psq_weights.0.get_unchecked(idx) };
        for (nodes, weights) in self.hidden1_nodes_wtm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_INPUTS).nth(idx).unwrap()) {
            *nodes += *weights;
        }

        let mut idx = base_index + mirror_pos(pos);
        if piece > 0 {
            idx += 64;
        }

        self.psq_btm_score -= unsafe { self.params.psq_weights.0.get_unchecked(idx) };
        for (nodes, weights) in self.hidden1_nodes_btm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_INPUTS).nth(idx).unwrap()) {
            *nodes += *weights;
        }
    }

    pub fn remove_piece(&mut self, pos: usize, piece: i8) {
        let base_index = self.offset + ((piece.abs() as usize - 1) * 2) as usize * 64;

        let mut idx = base_index + pos;
        if piece < 0 {
            idx += 64;
        }

        self.psq_wtm_score -= unsafe { self.params.psq_weights.0.get_unchecked(idx) };

        for (nodes, weights) in self.hidden1_nodes_wtm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_INPUTS).nth(idx).unwrap()) {
            *nodes -= *weights;
        }

        let mut idx = base_index + mirror_pos(pos);
        if piece > 0 {
            idx += 64;
        }

        self.psq_btm_score += unsafe { self.params.psq_weights.0.get_unchecked(idx) };
        for (nodes, weights) in self.hidden1_nodes_btm.0.iter_mut().zip(self.params.input_weights.0.chunks_exact(HL_INPUTS).nth(idx).unwrap()) {
            *nodes -= *weights;
        }
    }

    pub fn fast_eval(&self) -> i32 {
        if self.active_player == WHITE {
            self.psq_wtm_score as i32
        } else {
            self.psq_btm_score as i32
        }
    }

    pub fn eval(&mut self, half_move_clock: u8) -> i32 {
        if self.active_player == WHITE {
            if self.psq_wtm_score.abs() > 500 && (self.base_psq_wtm_score - self.psq_wtm_score).abs() > 500 {
                return adjust_eval(self.psq_wtm_score as i32, half_move_clock);
            }
            for ((node, &bias), weights) in self.hidden2_nodes.0.iter_mut().zip(&self.params.hidden1_biases.0).zip(self.params.hidden1_weights.0.chunks_exact(HL_INPUTS)) {
                *node = relu(dot_product::<HL_INPUTS>(&self.hidden1_nodes_wtm.0, weights) + bias);
            }

        } else {
            if self.psq_btm_score.abs() > 500 && (self.base_psq_btm_score - self.psq_btm_score).abs() > 500 {
                return adjust_eval(self.psq_btm_score as i32, half_move_clock);
            }
            for ((node, &bias), weights) in self.hidden2_nodes.0.iter_mut().zip(&self.params.hidden1_biases.0).zip(self.params.hidden1_weights.0.chunks_exact(HL_INPUTS)) {
                *node = relu(dot_product::<HL_INPUTS>(&self.hidden1_nodes_btm.0, weights) + bias);
            }
        }

        for ((node, &bias), weights) in self.output_nodes.0.iter_mut().zip(&self.params.hidden2_biases.0).zip(self.params.hidden2_weights.0.chunks_exact(HL_NODES)) {
            *node = relu(dot_product::<HL_NODES>(&self.hidden2_nodes.0, weights) + bias);
        }

        let out = (dot_product::<HL_OUT_NODES>(&self.output_nodes.0, &self.params.output_weights.0) + self.params.output_bias) as i32;
        let score = out * 2048 / (1 << FP_PRECISION_BITS);

        if self.active_player == WHITE {
            adjust_eval(score, half_move_clock)
        } else {
            adjust_eval(-score, half_move_clock)
        }
    }
}

// Scale eval score towards 0 for decreasing number of remaining half moves till the 50-move (draw) rule triggers
fn adjust_eval(score: i32, half_move_clock: u8) -> i32 {
    let remaining_half_moves = max(0, 100 - half_move_clock as i32);
    if remaining_half_moves >= 95 {
        score
    } else {
        score * remaining_half_moves / 95
    }
}

fn read_quantized(reader: &mut BufReader<&[u8]>, target: &mut [i16]) {
    let size = reader.read_i32::<LittleEndian>().expect("Could not read size") as usize;
    assert_eq!(size, target.len(), "Size mismatch: expected {}, but got {}", target.len(), size);

    reader.read_i16_into::<LittleEndian>(target).expect("Could not fill target");
}

fn calc_bucket(bitboards: &[u64; 13]) -> u8 {
    unsafe {
        let queens = if *bitboards.get_unchecked((Q + 6) as usize) == 0 && *bitboards.get_unchecked((-Q + 6) as usize) == 0 { 0 } else { 0b10 };
        let rooks = if *bitboards.get_unchecked((R + 6) as usize) == 0 && *bitboards.get_unchecked((-R + 6) as usize) == 0 { 0 } else { 0b01 };

        queens | rooks
    }
}

#[inline(always)]
fn dot_product<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
    #[cfg(target_feature = "avx2")]
        {
            avx2::dot_product::<N>(nodes, weights)
        }

    #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        {
            sse2::dot_product::<N>(nodes, weights)
        }

    #[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
        {
            fallback::dot_product::<N>(nodes, weights)
        }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod avx2 {
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;
    use crate::nn_eval::FP_PRECISION_BITS;

    #[inline(always)]
    pub fn dot_product<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        unsafe {
            let mut acc = _mm256_setzero_si256();

            for i in 0..(N / 16) {
                let n = _mm256_load_si256(transmute(nodes.as_ptr().add(i * 16)));
                let w = _mm256_load_si256(transmute(weights.as_ptr().add(i * 16)));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(n, w));
            }

            // Final horizontal sum of the lanes for the accumulator
            let sum128 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(hi64, sum128);
            let hi32 = _mm_shuffle_epi32::<0b10110001>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            (_mm_cvtsi128_si32(sum32) >> FP_PRECISION_BITS) as i16
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", not(target_feature = "avx2")))]
mod sse2 {
    use core::arch::x86_64::*;
    use std::intrinsics::transmute;
    use crate::nn_eval::FP_PRECISION_BITS;

    #[inline(always)]
    pub fn dot_product<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        unsafe {
            let mut acc = _mm_setzero_si128();

            for i in 0..(N / 8) {
                let n = _mm_load_si128(transmute(nodes.as_ptr().add(i * 8)));
                let w = _mm_load_si128(transmute(weights.as_ptr().add(i * 8)));
                acc = _mm_add_epi32(acc, _mm_madd_epi16(n, w));
            }

            // Final horizontal sum of the lanes for the accumulator
            let hi64 = _mm_shuffle_epi32::<0b01001110>(acc);
            let sum64 = _mm_add_epi32(hi64, acc);
            let hi32 = _mm_shufflelo_epi16::<0b01001110>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);

            (_mm_cvtsi128_si32(sum32) >> FP_PRECISION_BITS) as i16
        }
    }
}

#[cfg(not(any(target_feature = "sse2", target_feature = "avx2")))]
mod fallback {
    use crate::nn_eval::FP_PRECISION_BITS;

    #[inline(always)]
    pub fn dot_product<const N: usize>(nodes: &[i16], weights: &[i16]) -> i16 {
        (nodes.iter().zip(weights).map(|(&n, &w)| (n as i32 * w as i32)).sum::<i32>() >> FP_PRECISION_BITS) as i16
    }
}

#[inline(always)]
fn relu(v: i16) -> i16 {
    max(0, v)
}
