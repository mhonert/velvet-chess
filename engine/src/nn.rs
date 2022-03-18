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
pub const FEATURES_PER_BUCKET: usize = 64 * 6 * 2;
pub const INPUTS: usize = FEATURES_PER_BUCKET * 5;
pub const HL_NODES: usize = 512;

// Fixed point number precision
pub const FP_PRECISION_BITS: i16 = 11;

pub struct NeuralNetParams {
    pub input_weights: A32<[i16; INPUTS * HL_NODES]>,
    pub input_biases: A32<[i16; HL_NODES]>,

    pub output_weights: A32<[i16; HL_NODES]>,
    pub output_bias: i16,
}

impl NeuralNetParams {
    pub fn new() -> Arc<Self> {
        let mut reader = BufReader::new(&include_bytes!("../nets/velvet.qnn")[..]);

        let precision_bits = reader.read_i8().unwrap() as i16;
        assert_eq!(
            precision_bits, FP_PRECISION_BITS,
            "NN has been quantized with a different fixed point precision, expected: {}, got: {}",
            FP_PRECISION_BITS, precision_bits
        );

        let mut params = Box::new(NeuralNetParams::default());

        read_quantized(&mut reader, &mut params.input_weights.0).expect("Could not read input weights");
        read_quantized(&mut reader, &mut params.input_biases.0).expect("Could not read input biases");
        read_quantized(&mut reader, &mut params.output_weights.0).expect("Could not read output weights biases");

        params.output_bias = reader.read_i16::<LittleEndian>().expect("Could not read output bias");

        Arc::new(*params)
    }
}

impl Default for NeuralNetParams {
    fn default() -> Self {
        NeuralNetParams {
            input_weights: A32([0; INPUTS * HL_NODES]),
            input_biases: A32([0; HL_NODES]),

            output_weights: A32([0; HL_NODES]),
            output_bias: 0,
        }
    }
}
