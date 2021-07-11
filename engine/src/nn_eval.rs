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

use std::io::{BufReader};
use crate::bitboard::BitBoard;

use byteorder::{ReadBytesExt, LittleEndian};
use std::cmp::{max};
use crate::colors::{Color, WHITE};
use crate::pieces::{Q, R};
use packed_simd_2::{i16x16, i32x16};

// Fixed point number precision
const FP_PRECISION_BITS: i16 = 12;

// NN layer size
pub const IL_COUNT: usize = INPUTS * 2 * 4;
pub const HL_INPUTS: usize = 64;
const INPUTS: usize = 64 * 6 * 2;
pub const HL_COUNT: i8 = 2;
const HL_NODES: usize = 64;

pub struct NeuralNetEval {
    input_weights: [i16x16; IL_COUNT * HL_INPUTS / 16],
    input_biases: [i16x16; HL_INPUTS / 16],

    hidden1_nodes_wtm: [i16x16; HL_INPUTS / 16], // wtm - white to move
    hidden1_nodes_btm: [i16x16; HL_INPUTS / 16], // btm - black to move
    hidden1_weights: [i16x16; HL_INPUTS * HL_NODES / 16],
    hidden1_biases: [i16; HL_NODES],

    hidden2_nodes: [i16; HL_NODES],
    hidden2_weights: [i16x16; HL_NODES * HL_NODES / 16],
    hidden2_biases: [i16; HL_NODES],

    output_nodes: [i16; HL_NODES],
    output_weights: [i16x16; HL_NODES / 16],
    output_bias: i16,

    active_player: Color,
    bucket: u8,
    offset: usize,
}

impl Default for NeuralNetEval {
    fn default() -> Self {
        NeuralNetEval{
            input_weights: [i16x16::splat(0); IL_COUNT * HL_INPUTS / 16],
            input_biases: [i16x16::splat(0); HL_INPUTS / 16],

            hidden1_nodes_wtm: [i16x16::splat(0); HL_INPUTS / 16],
            hidden1_nodes_btm: [i16x16::splat(0); HL_INPUTS / 16],
            hidden1_weights: [i16x16::splat(0); HL_INPUTS * HL_NODES / 16],
            hidden1_biases: [0; HL_NODES],

            hidden2_nodes: [0; HL_NODES],
            hidden2_weights: [i16x16::splat(0); HL_NODES * HL_NODES / 16],
            hidden2_biases: [0; HL_NODES],

            output_nodes: [0; HL_NODES],
            output_weights: [i16x16::splat(0); HL_NODES / 16],
            output_bias: 0,

            active_player: WHITE,
            bucket: 0b11,
            offset: 0,
        }
    }
}

impl NeuralNetEval {

    pub fn new() -> Box<Self> {
        let mut reader = BufReader::new(&include_bytes!("../nets/velvet.qnn")[..]);

        let precision_bits = reader.read_i8().unwrap() as i16;
        if precision_bits != FP_PRECISION_BITS {
            panic!("NN has been quantized with a different fixed point precision, expected: {}, got: {}", FP_PRECISION_BITS, precision_bits);
        }

        let mut nn = Box::new(NeuralNetEval::default());

        read_quantized_simd(&mut reader, &mut nn.input_weights);
        read_quantized_simd(&mut reader, &mut nn.input_biases);

        read_quantized_simd(&mut reader, &mut nn.hidden1_weights);
        read_quantized(&mut reader, &mut nn.hidden1_biases);

        read_quantized_simd(&mut reader, &mut nn.hidden2_weights);
        read_quantized(&mut reader, &mut nn.hidden2_biases);

        read_quantized_simd(&mut reader, &mut nn.output_weights);

        nn.output_bias = reader.read_i16::<LittleEndian>().expect("Could not read output bias");

        nn
    }

    pub fn init_pos(&mut self, active_player: Color, bitboards: &[u64; 13]) {
        self.active_player = active_player;

        self.hidden1_nodes_wtm.copy_from_slice(&self.input_biases);
        self.hidden1_nodes_btm.copy_from_slice(&self.input_biases);

        self.bucket = calc_bucket(bitboards);

        self.offset = self.bucket as usize * 768 * 2;

        for piece in 1..=6 {
            for pos in BitBoard(bitboards[(piece + 6) as usize]) {
                self.add_piece(pos as usize, piece);
            }

            for pos in BitBoard(bitboards[(-piece + 6) as usize]) {
                self.add_piece(pos as usize, -piece);
            }
        }
    }

    pub fn set_stm(&mut self, active_player: Color) {
        self.active_player = active_player;
    }

    pub fn check_refresh(&mut self, active_player: Color, bitboards: &[u64; 13]) {
        let piece_combo = calc_bucket(bitboards);
        if piece_combo != self.bucket {
            self.init_pos(active_player, bitboards);
        }
    }

    pub fn add_piece(&mut self, pos: usize, piece: i8) {
        let mut idx = self.offset + ((piece.abs() as usize - 1) * 2) as usize * 64 + pos;
        if piece < 0 {
            idx += 64;
        }

        for (nodes, weights) in self.hidden1_nodes_wtm.iter_mut().zip(self.input_weights[(idx * HL_INPUTS / 16)..((idx * HL_INPUTS / 16) + HL_INPUTS / 16)].iter()) {
            *nodes += *weights;
        }

        idx += 768;

        for (nodes, weights) in self.hidden1_nodes_btm.iter_mut().zip(self.input_weights[(idx * HL_INPUTS / 16)..((idx * HL_INPUTS / 16) + HL_INPUTS / 16)].iter()) {
            *nodes += *weights;
        }
    }

    pub fn remove_piece(&mut self, pos: usize, piece: i8) {
        let mut idx = self.offset + ((piece.abs() as usize - 1) * 2) as usize * 64 + pos;
        if piece < 0 {
            idx += 64;
        }

        for (nodes, weights) in self.hidden1_nodes_wtm.iter_mut().zip(self.input_weights[(idx * HL_INPUTS / 16)..((idx * HL_INPUTS / 16) + HL_INPUTS / 16)].iter()) {
            *nodes -= *weights;
        }

        idx += 768;

        for (nodes, weights) in self.hidden1_nodes_btm.iter_mut().zip(self.input_weights[(idx * HL_INPUTS / 16)..((idx * HL_INPUTS / 16) + HL_INPUTS / 16)].iter()) {
            *nodes -= *weights;
        }
    }

    pub fn eval(&mut self) -> i32 {
        if self.active_player == WHITE {
            for ((node, &bias), weights) in self.hidden2_nodes.iter_mut().zip(&self.hidden1_biases).zip(self.hidden1_weights.chunks_exact(HL_INPUTS / 16)) {
                *node = relu(self.hidden1_nodes_wtm.dot_product(weights) + bias);
            }
        } else {
            for ((node, &bias), weights) in self.hidden2_nodes.iter_mut().zip(&self.hidden1_biases).zip(self.hidden1_weights.chunks_exact(HL_INPUTS / 16)) {
                *node = relu(self.hidden1_nodes_btm.dot_product(weights) + bias);
            }
        }

        for ((node, &bias), weights) in self.output_nodes.iter_mut().zip(&self.hidden2_biases).zip(self.hidden2_weights.chunks_exact(HL_NODES / 16)) {
            *node = relu(self.hidden2_nodes.dot_product(weights) + bias);
        }

        let out = (self.output_nodes.dot_product(&self.output_weights) + self.output_bias) as i32;

        out * 2048 / (1 << FP_PRECISION_BITS)
    }
}

fn read_quantized(reader: &mut BufReader<&[u8]>, target: &mut [i16]) {
    let size = reader.read_i32::<LittleEndian>().expect("Could not read size") as usize;
    if size != target.len() {
        panic!("Size mismatch: expected {}, but got {}", target.len(), size);
    }

    reader.read_i16_into::<LittleEndian>(target).expect("Could not fill target");
}

fn read_quantized_simd(reader: &mut BufReader<&[u8]>, target: &mut [i16x16]) {
    let size = reader.read_i32::<LittleEndian>().expect("Could not read size") as usize;
    if size != target.len() * 16 {
        panic!("Size mismatch: expected {}, but got {}", target.len() * 16, size);
    }

    for t in target.iter_mut() {
        let mut entry = [0i16; 16];
        reader.read_i16_into::<LittleEndian>(&mut entry).expect("Could not fill target");
        *t = i16x16::from_slice_unaligned(&entry);
    }
}

fn calc_bucket(bitboards: &[u64; 13]) -> u8 {
    unsafe {
        let queens = if *bitboards.get_unchecked((Q + 6) as usize) == 0 && *bitboards.get_unchecked((-Q + 6) as usize) == 0 { 0 } else { 0b10 };
        let rooks = if *bitboards.get_unchecked((R + 6) as usize) == 0 && *bitboards.get_unchecked((-R + 6) as usize) == 0 { 0 } else { 0b01 };

        queens | rooks
    }
}

trait DotProduct16<T> {
    fn dot_product(self, other: &[i16x16]) -> i16;
}

impl DotProduct16<&[i16]> for &[i16] {
    #[inline(always)]
    fn dot_product(self, weights: &[i16x16]) -> i16 {
        (self.chunks_exact(16)
            .map(|v| i16x16::from_slice_unaligned(v))
            .zip(weights)
            .map(|(n, w)| (i32x16::from(n) * i32x16::from(*w)))
            .sum::<i32x16>()
            .wrapping_sum() >> FP_PRECISION_BITS) as i16
    }
}

impl DotProduct16<&[i16x16]> for &[i16x16] {
    #[inline(always)]
    fn dot_product(self, weights: &[i16x16]) -> i16 {
        (self.iter().zip(weights)
            .map(|(n, w)| (i32x16::from(*n) * i32x16::from(*w)))
            .sum::<i32x16>()
            .wrapping_sum() >> FP_PRECISION_BITS) as i16
    }
}

#[inline(always)]
fn relu(v: i16) -> i16 {
    max(0, v)
}
