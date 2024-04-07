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

use std::sync::Once;

use crate::align::A32;
use crate::nn::io::{read_quantized, read_u8};

pub mod eval;
pub mod io;

// NN layer size
pub const BUCKETS: usize = 32;
pub const BUCKET_SIZE: usize = 6 * 64 * 2;
pub const INPUTS: usize = BUCKET_SIZE * BUCKETS;

pub const HL1_NODES: usize = 2 * HL1_HALF_NODES;
pub const HL1_HALF_NODES: usize = 768;

pub const MAX_RELU: f32 = 1.999;
pub const FP_MAX_RELU: i16 = (MAX_RELU * FP_IN_MULTIPLIER as f32) as i16;

pub const INPUT_WEIGHT_COUNT: usize = INPUTS * HL1_HALF_NODES;

// Fixed point number precision
pub const FP_IN_PRECISION_BITS: u8 = 10;
pub const FP_IN_MULTIPLIER: i64 = 1 << FP_IN_PRECISION_BITS;

pub const FP_OUT_PRECISION_BITS: u8 = 10; // must be an even number
pub const FP_OUT_MULTIPLIER: i64 = 1 << FP_OUT_PRECISION_BITS;

pub const SCORE_SCALE: i16 = 1280;

pub static mut IN_TO_H1_WEIGHTS: A32<[i16; INPUT_WEIGHT_COUNT]> = A32([0; INPUT_WEIGHT_COUNT]);
pub static mut H1_BIASES: A32<[i16; HL1_NODES]> = A32([0; HL1_NODES]);

pub static mut H1_TO_OUT_WEIGHTS: A32<[i16; HL1_NODES]> = A32([0; HL1_NODES]);

pub static mut OUT_BIASES: A32<[i16; 1]> = A32([0; 1]);

static INIT_NN_PARAMS: Once = Once::new();

pub const fn piece_idx(piece_id: i8) -> u16 {
    (piece_id - 1) as u16
}

pub fn init_nn_params() {
    INIT_NN_PARAMS.call_once(|| {
        let mut reader = &include_bytes!("../nets/velvet.qnn")[..];

        let in_precision_bits = read_u8(&mut reader).expect("Could not read input fixed point precision bits");
        assert_eq!(
            in_precision_bits, FP_IN_PRECISION_BITS,
            "NN hidden layer has been quantized with a different (input) fixed point precision, expected: {}, got: {}",
            FP_IN_PRECISION_BITS, in_precision_bits
        );

        let out_precision_bits = read_u8(&mut reader).expect("Could not read output fixed point precision bits");
        assert_eq!(
            out_precision_bits, FP_OUT_PRECISION_BITS,
            "NN hidden layer has been quantized with a different (input) fixed point precision, expected: {}, got: {}",
            FP_OUT_PRECISION_BITS, out_precision_bits
        );

        read_quantized(&mut reader, unsafe { &mut IN_TO_H1_WEIGHTS.0 }).expect("Could not read weights");
        read_quantized(&mut reader, unsafe { &mut H1_BIASES.0 }).expect("Could not read biases");

        read_quantized(&mut reader, unsafe { &mut H1_TO_OUT_WEIGHTS.0 }).expect("Could not read weights");
        read_quantized(&mut reader, unsafe { &mut OUT_BIASES.0 }).expect("Could not read biases");
    });
}

#[inline(always)]
pub fn king_bucket(pos: u16) -> u16 {
    let row = pos / 8;
    let col = pos & 3;

    row * 4 + col
}
