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

use traincommon::sets::DataSamples;
use velvet::colors::Color;

#[derive(Clone, Debug)]
pub struct DataSample {
    pub wpov_inputs: Vec<u16>,
    pub bpov_inputs: Vec<u16>,
    pub result: f32,
    pub wtm: bool,
}

impl Default for DataSample {
    fn default() -> Self {
        DataSample { wpov_inputs: Vec::with_capacity(32), bpov_inputs: Vec::with_capacity(32), result: 0.0, wtm: true }
    }
}

pub struct CpuDataSamples(pub Vec<DataSample>);

impl DataSamples for CpuDataSamples {
    fn init(&mut self, idx: usize, result: f32, stm: u8) {
        let sample = &mut self.0[idx];
        sample.wpov_inputs.clear();
        sample.bpov_inputs.clear();
        sample.result = result;
        sample.wtm = Color(stm).is_white();
    }

    fn add_wpov(&mut self, idx: usize, pos: u16) {
        self.0[idx].wpov_inputs.push(pos);
    }

    fn add_bpov(&mut self, idx: usize, pos: u16) {
        self.0[idx].bpov_inputs.push(pos);
    }

    fn finalize(&mut self, _idx: usize) {
    }
}
