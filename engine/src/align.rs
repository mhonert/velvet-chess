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

#[derive(Clone)]
#[repr(align(32))]
pub struct A32<T>(pub T); // Wrapper to ensure 32 Byte alignment of the wrapped type (e.g. for SIMD load/store instructions)

#[derive(Clone)]
#[repr(align(64))]
pub struct A64<T>(pub T); // Wrapper to ensure 64 Byte alignment (e.g. for cache line alignment)