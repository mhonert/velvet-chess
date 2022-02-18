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

use std::cmp::{max};
use std::collections::HashMap;
use std::io::{Error, ErrorKind, Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use itertools::Itertools;

pub type BitCount = u8;
pub type BitPattern = u32;

pub struct CodeBook {
    code_by_value: CodeByValue,
    value_by_code: ValueByCode,
}

enum Tree {
    Leaf{count: usize, value: i16},
    Node{count: usize, left: Box<Tree>, right: Box<Tree>}
}

type CodeByValue = HashMap<i16, (BitCount, BitPattern)>;
type ValueByCode = HashMap<(BitCount, BitPattern), i16>;

impl CodeBook {
    const END: u8 = 0;

    pub fn from_values(input: &[i16]) -> Self {
        let mut entries: Vec<Tree> = input.iter().sorted().group_by(|v| *v).into_iter()
            .map(|group| Tree::Leaf{value: *group.0, count: group.1.count()})
            .collect_vec();

        while entries.len() > 1 {
            entries.sort_by_key(Self::count);
            entries.reverse();

            let left = entries.pop().unwrap();
            let right = entries.pop().unwrap();

            entries.push(CodeBook::node(left, right));
        }

        let root = entries.pop().unwrap();
        let mut value_bitcounts = Vec::new();
        Self::gen_value_bitcounts(root, &mut value_bitcounts, 0);
        value_bitcounts.sort_unstable_by_key(|v| (v.1, v.0));

        let (code_by_value, value_by_code) = Self::gen_codes(&value_bitcounts);
        assert_eq!(code_by_value.len(), value_by_code.len(), "Inconsistent coding");
        CodeBook { code_by_value, value_by_code }
    }

    pub fn from_reader(reader: &mut dyn Read) -> Result<Self, Error> {
        let mut value_bitcounts = Vec::new();
        loop {
            let bitcount = reader.read_u8()?;
            if bitcount == 0 {
                break;
            }
            let value_count = reader.read_u16::<LittleEndian>()?;
            for _ in 0..value_count {
                let value = reader.read_i16::<LittleEndian>()?;
                value_bitcounts.push((value, bitcount));
            }
        }

        value_bitcounts.sort_unstable_by_key(|v| (v.1, v.0));

        let (code_by_value, value_by_code) = Self::gen_codes(&value_bitcounts);
        Ok(CodeBook { code_by_value, value_by_code })
    }

    pub fn write(&self, writer: &mut dyn Write) -> Result<(), Error> {
        for (bitcount, group) in self.code_by_value.iter().map(|(value, entry)| (entry.0, value))
            .sorted_unstable()
            .group_by(|entry| entry.0).into_iter() {

            writer.write_u8(bitcount)?;

            let values = group.map(|e| *e.1).collect_vec();
            writer.write_u16::<LittleEndian>(values.len() as u16)?;
            for v in values.iter() {
                writer.write_i16::<LittleEndian>(*v)?;
            }
        }

        writer.write_u8(CodeBook::END)?;

        Ok(())
    }

    pub fn get_code(&self, value: i16) -> (BitCount, BitPattern) {
        *self.code_by_value.get(&value).unwrap()
    }

    pub fn get_value(&self, bit_count: BitCount, code: BitPattern) -> Option<i16> {
        self.value_by_code.get(&(bit_count, code)).copied()
    }

    fn gen_codes(value_bitcounts: &[(i16, BitCount)]) -> (CodeByValue, ValueByCode) {
        let mut code_by_value = HashMap::new();
        let mut value_by_code = HashMap::new();

        let mut code = 0;
        let mut curr_bitcount = 0;
        for &(value, bitcount) in value_bitcounts.iter() {
            if bitcount > curr_bitcount {
                code <<= bitcount - curr_bitcount;
                curr_bitcount = bitcount;
            }

            code_by_value.insert(value, (bitcount, code));
            value_by_code.insert((bitcount, code), value);
            code += 1;
        }

        (code_by_value, value_by_code)
    }

    fn gen_value_bitcounts(node: Tree, value_bitcounts: &mut Vec<(i16, BitCount)>, bit_count: BitCount) {
        match node {
            Tree::Leaf {value, .. } => {
                value_bitcounts.push((value, max(1, bit_count)));
            }

            Tree::Node {left, right, .. } => {
                Self::gen_value_bitcounts(*left, value_bitcounts, bit_count + 1);
                Self::gen_value_bitcounts(*right, value_bitcounts, bit_count + 1);
            }
        }
    }

    fn count(node: &Tree) -> usize {
        match *node {
            Tree::Leaf {count, .. } => count,
            Tree::Node {count, .. } => count
        }
    }

    fn node(left: Tree, right: Tree) -> Tree {
        Tree::Node {
            count: Self::count(&left) + Self::count(&right),
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

type Bit = u8;

pub struct BitReader {
    bit_index: usize,
    buffer: Option<u32>,
}

impl BitReader {
    pub fn new() -> Self {
        BitReader{
            bit_index: 0,
            buffer: None,
        }
    }

    pub fn read(&mut self, reader: &mut dyn Read) -> Result<Option<Bit>, Error> {
        if self.bit_index == 32 || self.buffer.is_none() {
            self.buffer = Some(reader.read_u32::<LittleEndian>()?);
            self.bit_index = 0;
        }

        let value = self.buffer.unwrap();
        let bit = (value & 1) as Bit;
        self.bit_index += 1;
        self.buffer = Some(value >> 1);

        Ok(Some(bit))
    }
}

pub fn read_value(codebook: &CodeBook, reader: &mut dyn Read, bitreader: &mut BitReader) -> Result<i16, Error> {
    let mut code: BitPattern = 0;
    let mut bit_count = 0;

    while let Some(bit) = bitreader.read(reader)? {
        code <<= 1;
        code |= bit as BitPattern;
        bit_count += 1;

        if let Some(value) = codebook.get_value(bit_count, code) {
            return Ok(value);
        }
    }

    Result::Err(Error::new(ErrorKind::UnexpectedEof, "Could not read additional bits"))
}

pub struct BitWriter {
    bit_index: usize,
    value: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self{ bit_index: 0, value: 0 }
    }

    pub fn write(&mut self, writer: &mut dyn Write, value: (BitCount, BitPattern)) -> Result<(), Error> {
        let bit_count = value.0;
        let bits = reverse(bit_count, value.1);
        if (self.bit_index + bit_count as usize) < 32 {
            self.add(bit_count, bits);
            return Ok(());
        }

        let part1_count = 32 - self.bit_index;
        let part2_count = bit_count as usize - part1_count;

        let part1 = bits & ((1 << part1_count) - 1);
        let part2 = bits >> part1_count;

        self.add(part1_count as u8, part1);
        self.flush(writer)?;
        self.add(part2_count as u8, part2);

        Ok(())
    }

    pub fn flush(&mut self, writer: &mut dyn Write) -> Result<(), Error> {
        if self.bit_index > 0 {
            writer.write_u32::<LittleEndian>(self.value)?;
            self.value = 0;
            self.bit_index = 0;
        }

        Ok(())
    }

    fn add(&mut self, bit_count: u8, bit_pattern: u32) {
        self.value |= bit_pattern << self.bit_index as u32;
        self.bit_index += bit_count as usize;
    }
}

fn reverse(bit_count: u8, bits: u32) -> u32 {
    bits.reverse_bits() >> (32 - bit_count)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use super::*;

    #[test]
    fn code_book_from_values() {
        let input = Vec::from([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]);

        let codes = CodeBook::from_values(&input);

        assert_eq!(codes.get_code(1), (1, 0b0));
        assert_eq!(codes.get_code(2), (2, 0b10));
        assert_eq!(codes.get_code(3), (3, 0b110));
        assert_eq!(codes.get_code(4), (3, 0b111));
    }

    #[test]
    fn code_book_from_single_value() {
        let input = Vec::from([1, 1, 1]);

        let codes = CodeBook::from_values(&input);

        assert_eq!(codes.get_code(1), (1, 0b0));
    }

    #[test]
    fn code_book_write() {
        let input = Vec::from([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4]);
        let codes = CodeBook::from_values(&input);

        let mut output = Vec::<u8>::new();
        codes.write(&mut output).expect("Writing the code book failed");

        let expected: &[u8] = &[1, 1, 0, 1, 0, 2, 1, 0, 2, 0, 3, 2, 0, 3, 0, 4, 0, 0];
        assert_eq!(expected, &output);
    }

    #[test]
    fn tree_node() {
        let entry1 = Tree::Leaf{value: 1, count: 10};
        let entry2 = Tree::Leaf{value: 2, count: 5};

        let entry = CodeBook::node(entry1, entry2);

        assert_eq!(CodeBook::count(&entry), 15);
    }

    #[test]
    fn reverse_bits() {
        assert_eq!(0b001, reverse(3, 0b100));
        assert_eq!(0b100, reverse(3, 0b001));
        assert_eq!(0b1010, reverse(4, 0b0101));
        assert_eq!(0b01011, reverse(5, 0b11010));
    }

    #[test]
    fn bitreader_read_bit() {
        let input = vec![0b1010101u8, 0, 0, 0];
        let mut reader = Cursor::new(input);
        let mut bitreader = BitReader::new();

        let bit1 = bitreader.read(&mut reader).expect("Could not read value").unwrap();
        let bit2 = bitreader.read(&mut reader).expect("Could not read value").unwrap();
        let bit3 = bitreader.read(&mut reader).expect("Could not read value").unwrap();

        assert_eq!([1, 0, 1], [bit1, bit2, bit3]);
    }

    #[test]
    fn bitreader_read_bit_across_buffer_size() {
        let input = vec![0, 0, 0, 0, 0xFF, 0, 0, 0];
        let mut reader = Cursor::new(input);
        let mut bitreader = BitReader::new();

        for _ in 0..32 {
            bitreader.read(&mut reader).expect("Could not read value").unwrap();
        }

        let bit33 = bitreader.read(&mut reader).expect("Could not read value").unwrap();

        assert_eq!(bit33, 1);
    }

    #[test]
    fn write_bits() {
        let mut writer = Cursor::new(Vec::<u8>::new());
        let mut bit_writer = BitWriter::new();

        bit_writer.write(&mut writer, (3, 0b101)).unwrap();
        bit_writer.flush(&mut writer).unwrap();

        let output = writer.into_inner();

        assert!(!output.is_empty());
        assert_eq!(output[0], 0b101);
    }

    #[test]
    fn write_multiple_bits() {
        let mut writer = Cursor::new(Vec::<u8>::new());
        let mut bit_writer = BitWriter::new();

        bit_writer.write(&mut writer, (30, 0b111111111111111111111111111111)).unwrap();
        bit_writer.write(&mut writer, (3, 0b111)).unwrap();
        bit_writer.flush(&mut writer).unwrap();

        let output = writer.into_inner();

        assert!(!output.is_empty());
        assert_eq!(output[0], 0xFF);
        assert_eq!(output[1], 0xFF);
        assert_eq!(output[2], 0xFF);
        assert_eq!(output[3], 0xFF);
        assert_eq!(output[4], 0b1);
    }

    #[test]
    fn bit_writer_reader_roundtrip() {
        let mut bit_writer = BitWriter::new();
        let mut writer = Cursor::new(Vec::<u8>::new());
        for _ in 0..10 {
            bit_writer.write(&mut writer, (7, 0b1110001)).expect("Could nod write bits");
        }
        bit_writer.flush(&mut writer).expect("Could not flush");
        let output = writer.into_inner();

        let mut reader = Cursor::new(output);
        let mut bit_reader = BitReader::new();

        for _ in 0..10 {
            let bit1 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit2 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit3 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit4 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit5 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit6 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");
            let bit7 = bit_reader.read(&mut reader).expect("Could not read bit").expect("Unexpected EOF");

            assert_eq!([1, 1, 1, 0, 0, 0, 1], [bit1, bit2, bit3, bit4, bit5, bit6, bit7]);
        }
    }
}
