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

use itertools::Itertools;
use rand::prelude::SliceRandom;
use rand::RngCore;

pub struct IDSource {
    ids: Vec<usize>,
    min_id: usize,
    max_id: usize,
    batch_id_count: usize,
    epoch: usize,
}

impl IDSource {
    pub fn new(rng: &mut dyn RngCore, min_id: usize, max_id: usize, batch_id_count: usize) -> Self {
        IDSource { ids: shuffled_ids(rng, min_id, max_id), min_id, max_id, batch_id_count, epoch: 1 }
    }

    pub fn next_batch(&mut self, rng: &mut dyn RngCore) -> (usize, Vec<usize>) {
        if self.ids.len() < self.batch_id_count {
            self.ids.append(&mut shuffled_ids(rng, self.min_id, self.max_id));
            self.epoch += 1;
        }
        (self.epoch, self.ids.drain(self.ids.len() - self.batch_id_count..).collect_vec())
    }

    pub fn per_batch_count(&self) -> usize {
        self.batch_id_count
    }
}

fn shuffled_ids(rng: &mut dyn RngCore, min: usize, max: usize) -> Vec<usize> {
    let mut ids = (min..=max).collect_vec();
    ids.shuffle(rng);
    ids
}
