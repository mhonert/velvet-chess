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

use crate::{Tensor};
use std::borrow::Borrow;
use tch::{IndexOp};
use tch::nn::{Path};
use tch::nn::init::DEFAULT_KAIMING_UNIFORM;
use velvet::nn::{INPUTS};

pub struct InputLayer {
    pub weights: Tensor,
    pub own_biases: Tensor,
    pub opp_biases: Tensor,
}

pub fn input_layer<'a, T: Borrow<Path<'a>>>(
    vs: T,
    input_count: i64,
    output_count: i64,
) -> InputLayer {
    let vs = vs.borrow();
    let own_biases = vs.zeros("own_bias", &[output_count]);
    let opp_biases = vs.zeros("opp_bias", &[output_count]);
    let weights = vs.var("weight", &[output_count, input_count], DEFAULT_KAIMING_UNIFORM);

    tch::no_grad(|| {
        let _ = weights.i((.., INPUTS as i64..)).zero_();
    });

    InputLayer { weights, own_biases, opp_biases }
}

impl InputLayer {
    pub fn forward(&self, white_xs: &Tensor, black_xs: &Tensor, stms: &Tensor) -> Tensor {
        let white = white_xs.matmul(&self.weights.tr());
        let black = black_xs.matmul(&self.weights.tr());

        stms * Tensor::cat(&[&white + &self.own_biases, &black + &self.opp_biases], 1)
            + (1i32 - stms) * Tensor::cat(&[&black + &self.own_biases, &white + &self.opp_biases], 1)
    }

    pub fn copy_from(&mut self, weights: &Tensor, own_biases: &Tensor, opp_biases: &Tensor) {
        tch::no_grad(|| {
            self.weights.copy_(weights);
            self.own_biases.copy_(own_biases);
            self.opp_biases.copy_(opp_biases);
        });
    }
}

pub struct OutputLayer {
    weights: Tensor,
    biases: Tensor,
}

/// Creates a new output layer.
pub fn output_layer<'a, T: Borrow<Path<'a>>>(
    vs: T,
    input_count: i64,
) -> OutputLayer {
    let vs = vs.borrow();

    let weights = vs.var("weight", &[1, input_count], DEFAULT_KAIMING_UNIFORM);
    let biases = vs.zeros("bias", &[1]);

    OutputLayer { weights, biases }
}

impl OutputLayer {
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        xs.linear(&self.weights, Some(&self.biases))
    }

    pub fn copy_from(&mut self, weights: &Tensor, biases: &Tensor) {
        tch::no_grad(|| {
            self.weights.copy_(weights);
            self.biases.copy_(biases);
        });
    }
}
