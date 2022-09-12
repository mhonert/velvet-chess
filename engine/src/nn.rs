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

use std::io::BufReader;
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::align::A32;
use crate::nn::io::read_quantized;

pub mod eval;
pub mod io;

// NN layer size
pub const FULL_BUCKETS: usize = 8 + 8 + 8;
pub const INPUTS: usize = FULL_BUCKETS * FULL_BUCKET_SIZE;
pub const HL_NODES: usize = 2 * HL_HALF_NODES;
pub const HL_HALF_NODES: usize = 288;

pub const INPUT_WEIGHT_COUNT: usize = INPUTS * HL_HALF_NODES;

pub const FULL_BUCKET_SIZE: usize = (64 * 6) * 2;

// Fixed point number precision
pub const FP_HIDDEN_MULTIPLIER: i16 = 3379;
pub const FP_INPUT_MULTIPLIER: i16 = 683;

pub struct NeuralNetParams {
    pub input_weights: Box<A32<[i16; INPUT_WEIGHT_COUNT]>>,
    pub input_biases: Box<A32<[i16; HL_NODES]>>,

    pub output_weights: Box<A32<[i16; HL_NODES]>>,
}

impl NeuralNetParams {
    pub fn new() -> Arc<Self> {
        let mut reader = BufReader::new(&include_bytes!("../nets/velvet.qnn")[..]);

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

        let mut params = NeuralNetParams::default();

        read_quantized(&mut reader, &mut params.input_weights.as_mut().0).expect("Could not read input weights");
        read_quantized(&mut reader, &mut params.input_biases.as_mut().0).expect("Could not read input biases");

        read_quantized(&mut reader, &mut params.output_weights.as_mut().0)
            .expect("Could not read output weights biases");

        Arc::new(params)
    }
}

impl Default for NeuralNetParams {
    fn default() -> Self {
        NeuralNetParams {
            input_weights: Box::new(A32([0; INPUT_WEIGHT_COUNT])),
            input_biases: Box::new(A32([0; HL_NODES])),

            output_weights: Box::new(A32([0; HL_NODES])),
        }
    }
}
