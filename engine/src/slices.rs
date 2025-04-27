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

/// Trait for accessing elements of slices.
/// Bounds checking can be enabled via the `checked_slice_access` feature.
pub trait SliceElementAccess<T> {
    fn el(&self, idx: usize) -> &T;
    fn el_mut(&mut self, idx: usize) -> &mut T;
}

impl<T> SliceElementAccess<T> for [T] {
    fn el(&self, idx: usize) -> &T {
        #[cfg(not(feature = "checked_slice_access"))]
        {
            unsafe { self.get_unchecked(idx) }
        }
        #[cfg(feature = "checked_slice_access")]
        {
            &self[idx]
        }
    }

    fn el_mut(&mut self, idx: usize) -> &mut T {
        #[cfg(not(feature = "checked_slice_access"))]
        {
            unsafe { self.get_unchecked_mut(idx) }
        }
        #[cfg(feature = "checked_slice_access")]
        {
            &mut self[idx]
        }
    }
}