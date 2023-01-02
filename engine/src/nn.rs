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

use std::sync::Once;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::align::A32;
use crate::nn::io::read_quantized;
use crate::pieces::{B, P, Q, R};

pub mod eval;
pub mod io;

pub const fn bucket_size(max_piece_id: i8) -> usize {
    (piece_idx(max_piece_id) + 1) as usize * 64
}

// NN layer size
pub const KING_BUCKETS: usize = 8;
pub const INPUTS: usize =
    ((bucket_size(Q) + bucket_size(R) + bucket_size(B) + bucket_size(P)) * KING_BUCKETS + 64 * 4) * 2;

pub const HL_NODES: usize = 2 * HL_HALF_NODES;
pub const HL_HALF_NODES: usize = 256;

pub const INPUT_WEIGHT_COUNT: usize = INPUTS * HL_HALF_NODES;

// Fixed point number precision
pub const FP_HIDDEN_MULTIPLIER: i16 = 1 << 12;
pub const FP_INPUT_MULTIPLIER: i16 = 1 << 10;

pub static mut INPUT_WEIGHTS: A32<[i16; INPUT_WEIGHT_COUNT]> = A32([0; INPUT_WEIGHT_COUNT]);
pub static mut INPUT_BIASES: A32<[i16; HL_NODES]> = A32([0; HL_NODES]);
pub static mut OUTPUT_WEIGHTS: A32<[i16; HL_NODES]> = A32([0; HL_NODES]);
pub static mut OUTPUT_BIASES: A32<[i16; 1]> = A32([0; 1]);

static INIT_NN_PARAMS: Once = Once::new();

const PIECE_INDEXES: [u16; 7] = [0, 0, 1, 2, 3, 4, 0];

pub const fn piece_idx(piece_id: i8) -> u16 {
    PIECE_INDEXES[piece_id as usize]
}

pub fn init_nn_params() {
    INIT_NN_PARAMS.call_once(|| {
        let mut reader = &include_bytes!("../nets/velvet.qnn")[..];

        let input_multiplier = reader.read_i16::<LittleEndian>().expect("Could not read input multiplier");
        assert_eq!(
            input_multiplier, FP_INPUT_MULTIPLIER,
            "NN hidden layer has been quantized with a different (input) fixed point multiplier, expected: {}, got: {}",
            FP_INPUT_MULTIPLIER, input_multiplier
        );

        let hidden_multiplier = reader.read_i16::<LittleEndian>().expect("Could not read hidden multiplier");
        assert_eq!(
            hidden_multiplier, FP_HIDDEN_MULTIPLIER,
            "NN hidden layer has been quantized with a different (hidden) fixed point multiplier, expected: {}, got: {}",
            FP_HIDDEN_MULTIPLIER, hidden_multiplier
        );

        read_quantized(&mut reader, unsafe { &mut INPUT_WEIGHTS.0 }).expect("Could not read input weights");
        read_quantized(&mut reader, unsafe { &mut INPUT_BIASES.0 }).expect("Could not read input biases");
        read_quantized(&mut reader, unsafe { &mut OUTPUT_WEIGHTS.0 }).expect("Could not read output weights");
        read_quantized(&mut reader, unsafe { &mut OUTPUT_BIASES.0 }).expect("Could not read output biases");
    });
}
