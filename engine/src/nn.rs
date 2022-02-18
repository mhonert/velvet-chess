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

use std::io::{BufReader, Error, ErrorKind, Read};
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt};

use crate::align::A32;
use crate::compression::{BitReader, CodeBook, read_value};

pub mod eval;

// NN layer size
pub const FEATURES_PER_BUCKET: usize = 64 * 6 * 2;
pub const INPUTS: usize = FEATURES_PER_BUCKET * 5;
pub const HL_NODES: usize = 256;

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
        assert_eq!(precision_bits, FP_PRECISION_BITS, "NN has been quantized with a different fixed point precision, expected: {}, got: {}", FP_PRECISION_BITS, precision_bits);

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
        NeuralNetParams{
            input_weights: A32([0; INPUTS * HL_NODES]),
            input_biases: A32([0; HL_NODES]),

            output_weights: A32([0; HL_NODES]),
            output_bias: 0,
        }
    }
}

fn read_quantized(reader: &mut dyn Read, target: &mut [i16]) -> Result<(), Error> {
    let size = reader.read_u32::<LittleEndian>()? as usize;
    if size != target.len() {
        return Result::Err(Error::new(ErrorKind::InvalidData, format!("Size mismatch: expected {}, but got {}", target.len(), size)));
    }

    let zero_rep_marker = reader.read_i16::<LittleEndian>()?;

    let codebook = CodeBook::from_reader(reader).expect("Could not read codebook for unpacking the NN data");
    let mut bitreader = BitReader::new();

    let mut index = 0;

    while index < size {
        let mut value = read_value(&codebook, reader, &mut bitreader)?;
        let repetitions = if value == zero_rep_marker {
            value = 0;
            read_value(&codebook, reader, &mut bitreader)? as i32 + 32768
        } else {
            1
        };

        for _ in 0..repetitions {
            target[index] = value;
            index += 1;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use byteorder::WriteBytesExt;
    use crate::compression::{BitWriter, CodeBook};
    use super::*;

    #[test]
    fn compression_roundtrip() {
        let values: Vec<i16> = Vec::from_iter(-100..100);
        let output = write_values(&values);

        let mut read_values = vec![0; values.len()];
        let mut reader = Cursor::new(output);
        read_quantized(&mut reader, &mut read_values).expect("Could not read values");

        assert_eq!(values, read_values);
    }

    fn write_values(values: &[i16]) -> Vec<u8> {
        let codebook = CodeBook::from_values(values);
        let mut writer = Cursor::new(Vec::<u8>::new());
        let mut bitwriter = BitWriter::new();

        writer.write_u32::<LittleEndian>(values.len() as u32).expect("Could not write size");
        writer.write_i16::<LittleEndian>(2000).expect("Could not zero-rep marker");
        codebook.write(&mut writer).expect("Could not write codebook");
        for &value in values.iter() {
            bitwriter.write(&mut writer, codebook.get_code(value)).expect("Could not write code");
        }
        bitwriter.flush(&mut writer).expect("Could not flush");

        writer.into_inner()
    }

}
