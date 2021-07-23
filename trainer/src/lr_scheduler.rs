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

pub struct LrScheduler {
    max_epochs: usize,
    init_lr: f64,
    mid_lr: f64,
    final_lr: f64,
}

impl LrScheduler {
    pub fn new(max_epochs: usize, init_lr: f64, mid_lr: f64, final_lr: f64) -> Self {
        LrScheduler{max_epochs, init_lr, mid_lr, final_lr}
    }

    pub fn calc_lr(&self, epoch: usize) -> f64 {
        if epoch <= self.max_epochs / 3 {
            let steps = (self.max_epochs / 3 - 1) as f64;
            return self.init_lr - (self.init_lr - self.mid_lr) / steps * (epoch as f64 - 1.0);
        }

        let steps = (self.max_epochs - (self.max_epochs / 3)) as f64;
        self.mid_lr - (self.mid_lr - self.final_lr) / steps * ((epoch - (self.max_epochs / 3)) as f64)
    }
}
#[cfg(test)]
mod tests {
    use crate::lr_scheduler::LrScheduler;

    #[test]
    fn test() {
        let lr_scheduler = LrScheduler::new(30, 0.005, 0.0005, 0.000005);
        for i in 1..=30 {
            println!("{:2}: {:1.9}", i, lr_scheduler.calc_lr(i))
        }
    }
}