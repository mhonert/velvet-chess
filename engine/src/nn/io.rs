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

use std::hash::{Hasher};
use std::io::{Error, ErrorKind, Read, Write};

pub fn read_quantized(reader: &mut dyn Read, target: &mut [i16]) -> Result<(), Error> {
    let size = read_u32(reader)? as usize;
    if size != target.len() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("Size mismatch: expected {}, but got {}", target.len(), size),
        ));
    }

    let code_len = read_u8(reader)? as usize;
    let mut code_book = [0; 256];
    let mut max_code: u8 = 0;
    for _ in 0..code_len {
        let hi_value = read_u8(reader)?;
        let code = read_u8(reader)?;
        max_code = max_code.max(code);
        code_book[code as usize] = hi_value;
    }

    max_code = (max_code as i8 - (1 + 3)).max(3) as u8;
    let max_bits = 8 - (max_code | 1).leading_zeros() as usize;

    let mut index = 0;

    let mut bit_reader = BitReader::default();
    while index < target.len() {
        let bit1 = bit_reader.read(reader, 1)?;
        let hi_value;
        if bit1 == 0 {
            let hi_code = bit_reader.read(reader, 1)?;
            hi_value = code_book[hi_code as usize];
        } else {
            let bit2 = bit_reader.read(reader, 1)?;
            if bit2 == 0 {
                let hi_code = bit_reader.read(reader, 2).expect("Could not read 2 hi code bits") + 1;
                hi_value = code_book[hi_code as usize];
            } else {
                let bit3 = bit_reader.read(reader, 1)?;
                if bit3 == 0 {
                    let hi_code = bit_reader.read(reader, max_bits).expect("Could not read 2 hi code bits") + 1 + 3;
                    hi_value = code_book[hi_code as usize];
                } else {
                    // 0-value repetitions
                    let reps = bit_reader.read(reader, 6).expect("Could not read 2 hi code bits") as usize + 1;
                    target[index..(index + reps)].fill(0);
                    index += reps;
                    continue;
                }
            }
        }

        let lo_value = bit_reader.read(reader, 8).expect("Could not read lo value bits");
        let value = ((hi_value as u16) << 8 | lo_value as u16) as i16;
        target[index] = value;
        index += 1;
    }

    Ok(())
}

#[derive(Default)]
struct BitReader {
    remaining_bits: usize,
    buffer: u64,
}

impl BitReader {
    pub fn read(&mut self, reader: &mut dyn Read, count: usize) -> Result<u8, Error> {
        if self.remaining_bits < count {
            let add = read_u32(reader)? as u64;
            self.buffer |= add << self.remaining_bits;
            self.remaining_bits += 32;
        }
        self.remaining_bits -= count;
        let bits = self.buffer % (1 << count);
        self.buffer >>= count;

        Ok(bits as u8)
    }
}

#[derive(Default)]
pub struct BitWriter {
    bit_index: usize,
    value: u32,
}

impl BitWriter {
    pub fn write(&mut self, writer: &mut dyn Write, bit_count: usize, bits: u32) -> Result<(), Error> {
        if (self.bit_index + bit_count) < 32 {
            self.add(bit_count, bits);
            return Ok(());
        }

        let part1_count = 32 - self.bit_index;
        let part2_count = bit_count - part1_count;

        let part1 = bits  % (1 << part1_count);
        let part2 = bits  >> part1_count;

        self.add(part1_count, part1);
        self.flush(writer)?;
        self.add(part2_count, part2);

        Ok(())
    }

    pub fn flush(&mut self, writer: &mut dyn Write) -> Result<(), Error> {
        if self.bit_index > 0 {
            write_u32(writer, self.value)?;
            self.value = 0;
            self.bit_index = 0;
        }

        Ok(())
    }

    fn add(&mut self, bit_count: usize, bit_pattern: u32) {
        self.value |= bit_pattern << self.bit_index;
        self.bit_index += bit_count;
    }
}

#[derive(Default)]
pub struct FastHasher(u64);

const GOLDEN_RATIO: u64 = 0x9E3779B97F4A7C15;

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, _: &[u8]) {
        panic!("write for &[u8] not implemented")
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.0 = (self.0 ^ value as u64).wrapping_mul(GOLDEN_RATIO);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.0 = (self.0 ^ value as u64).wrapping_mul(GOLDEN_RATIO);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0 ^ value).wrapping_mul(GOLDEN_RATIO);
    }

    #[inline]
    fn write_i16(&mut self, value: i16) {
        self.0 = (self.0 ^ value as u64).wrapping_mul(GOLDEN_RATIO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn bitreader_read_bit() {
        let input = vec![0b1010101u8, 0, 0, 0];
        let mut reader = Cursor::new(input);
        let mut bitreader = BitReader::default();

        let bit1 = bitreader.read(&mut reader, 1).expect("Could not read value");
        let bit2 = bitreader.read(&mut reader, 1).expect("Could not read value");
        let bit3 = bitreader.read(&mut reader, 1).expect("Could not read value");

        assert_eq!([1, 0, 1], [bit1, bit2, bit3]);
    }

    #[test]
    fn bitreader_read_bit_across_buffer_size() {
        let input = vec![0, 0, 0, 0, 0xFF, 0, 0, 0];
        let mut reader = Cursor::new(input);
        let mut bitreader = BitReader::default();

        for _ in 0..32 {
            bitreader.read(&mut reader, 1).expect("Could not read value");
        }

        let bit33 = bitreader.read(&mut reader, 1).expect("Could not read value");

        assert_eq!(bit33, 1);
    }

    #[test]
    fn write_bits() {
        let mut writer = Cursor::new(Vec::<u8>::new());
        let mut bit_writer = BitWriter::default();

        bit_writer.write(&mut writer, 3, 0b101).unwrap();
        bit_writer.flush(&mut writer).unwrap();

        let output = writer.into_inner();

        assert!(!output.is_empty());
        assert_eq!(output[0], 0b101);
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut bit_writer = BitWriter::default();
        let mut writer = Cursor::new(Vec::<u8>::new());

        let count = 7;
        let written_bits = 0b1110001;
        for _ in 0..10 {
            bit_writer.write(&mut writer, count, written_bits).expect("Could not write bits");
        }
        bit_writer.flush(&mut writer).expect("Could not flush");
        let output = writer.into_inner();

        let mut reader = Cursor::new(output);
        let mut bit_reader = BitReader::default();

        for _ in 0..10 {
            let read_bits = bit_reader.read(&mut reader, count).expect("Could not read bits");
            assert_eq!(read_bits as u32, written_bits);
        }
    }
}

pub fn read_u8(reader: &mut dyn Read) -> Result<u8, Error> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn read_u16(reader: &mut dyn Read) -> Result<u16, Error> {
    let mut buf = [0; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

pub fn read_u32(reader: &mut dyn Read) -> Result<u32, Error> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

pub fn read_u64(reader: &mut dyn Read) -> Result<u64, Error> {
    let mut buf = [0; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

pub fn read_f32(reader: &mut dyn Read) -> Result<f32, Error> {
    let mut buf = [0; 4];
    reader.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

pub fn read_i16(reader: &mut dyn Read) -> Result<i16, Error> {
    let mut buf = [0; 2];
    reader.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

pub fn write_u8(writer: &mut dyn Write, v: u8) -> Result<(), Error> {
    writer.write_all(&[v])
}

pub fn write_u16(writer: &mut dyn Write, v: u16) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
}

pub fn write_u64(writer: &mut dyn Write, v: u64) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
}

pub fn write_u32(writer: &mut dyn Write, v: u32) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
}

pub fn write_f32(writer: &mut dyn Write, v: f32) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
}

pub fn write_i16(writer: &mut dyn Write, v: i16) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
}
