/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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
use crate::align::A64;

pub mod eval;
pub mod io;

// NN layer size
pub const BUCKETS: usize = 32;
pub const BUCKET_SIZE: usize = 6 * 64 * 2;
pub const INPUTS: usize = BUCKET_SIZE * BUCKETS;

pub const HL1_NODES: usize = 2 * HL1_HALF_NODES;
pub const HL1_HALF_NODES: usize = 1536;

pub const MAX_RELU: f32 = 2.499;
pub const FP_MAX_RELU: i16 = (MAX_RELU * FP_IN_MULTIPLIER as f32) as i16;

pub const INPUT_WEIGHT_COUNT: usize = INPUTS * HL1_HALF_NODES;

// Fixed point number precision
pub const FP_IN_PRECISION_BITS: u8 = 6;
pub const FP_IN_MULTIPLIER: i64 = 1 << FP_IN_PRECISION_BITS;

pub const FP_OUT_PRECISION_BITS: u8 = 10; // must be an even number
pub const FP_OUT_MULTIPLIER: i64 = 1 << FP_OUT_PRECISION_BITS;

pub const SCORE_SCALE: i16 = 1024;

macro_rules! include_layer {
    ($file:expr, $T:ty, $S:expr) => {{
        let layer_bytes = include_bytes!($file);
        let layer: A64<[$T; $S]> = A64(unsafe { std::mem::transmute_copy(layer_bytes) });
        layer
    }};
}

pub static IN_TO_H1_WEIGHTS: A64<[i8; INPUT_WEIGHT_COUNT]> = include_layer!("../nets/velvet_layer_in_weights.qnn", i8, INPUT_WEIGHT_COUNT);

pub static H1_BIASES: A64<[i8; HL1_NODES]> = include_layer!("../nets/velvet_layer_in_bias.qnn", i8, HL1_NODES);

pub static H1_TO_OUT_WEIGHTS: A64<[i16; HL1_NODES]> = include_layer!("../nets/velvet_layer_out_weights.qnn", i16, HL1_NODES);

pub static OUT_BIASES: A64<[i16; 1]> = include_layer!("../nets/velvet_layer_out_bias.qnn", i16, 1);

pub const fn piece_idx(piece_id: i8) -> u16 {
    (piece_id - 1) as u16
}

#[inline(always)]
pub fn king_bucket(pos: u16) -> u16 {
    let row = pos / 8;
    let col = pos & 3;

    row * 4 + col
}
