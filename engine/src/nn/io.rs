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

use std::hash::{Hasher};
use std::io::{Error, Read, Write};

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

pub fn read_u8(reader: &mut dyn Read) -> Result<u8, Error> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

pub fn read_i8(reader: &mut dyn Read) -> Result<i8, Error> {
    let mut buf = [0; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0] as i8)
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

pub fn write_i8(writer: &mut dyn Write, v: i8) -> Result<(), Error> {
    writer.write_all(&v.to_le_bytes())
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
