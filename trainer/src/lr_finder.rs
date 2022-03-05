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

use crate::{BATCH_SIZE, K_DIV, POS_PER_SET};
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::prelude::BitMapBackend;
use plotters::series::LineSeries;
use plotters::style::{RED, WHITE};
use std::sync::mpsc::Receiver;
use tch::nn::{ModuleT, Optimizer};
use tch::{nn, Reduction, Tensor};

// find_lr performs a short training run with decreasing learning rates (lr) and plots a diagram for the lr and the calculated loss
// This diagram can then be used to (visually) determine an appropriate initial learning rate.
pub fn find_lr(
    test_xs: &Tensor, test_ys: &Tensor, rx: &Receiver<(usize, Tensor, Tensor)>, net: &impl ModuleT, opt: &mut Optimizer,
) {
    let mut lr = 0.0001;

    let beta = 0.98;
    let mut batch_num = 0;
    let multiplier = (10.0f64 / lr).powf(1.0 / (POS_PER_SET as f64 * 100.0 / BATCH_SIZE as f64));

    let mut loss_by_lr = Vec::new();
    let mut best_loss = 1000.0;
    let mut worst_loss = 0.0;

    let mut avg_loss = 0.0;

    loop {
        let (_, data_batch, label_batch) = rx.recv().expect("Could not receive test positions");
        opt.set_lr(lr);

        let train_loss = net
            .forward_t(&data_batch, true)
            .multiply_scalar(2048.0 * K_DIV)
            .sigmoid()
            .mse_loss(&(label_batch * 2048.0 * K_DIV).sigmoid(), Reduction::Mean);

        opt.backward_step(&train_loss);

        let loss = f64::from(
            &net.forward_t(test_xs, false).multiply_scalar(2048.0 * K_DIV).sigmoid().mse_loss(test_ys, Reduction::Mean),
        );

        avg_loss = if batch_num == 0 { loss } else { beta * avg_loss + (1.0 - beta) * loss };
        let avg_loss_smoothed = if batch_num == 0 { avg_loss } else { avg_loss / (1.0 - beta.powf(batch_num as f64)) };

        if avg_loss < best_loss {
            best_loss = avg_loss_smoothed;
        }

        if avg_loss > best_loss * 3.0 {
            plot_lr_vs_loss(&mut loss_by_lr, worst_loss);
            return;
        }

        println!("{}; {}; {}", lr, lr.log10(), avg_loss_smoothed);

        if avg_loss_smoothed.is_nan() {
            println!("--------------------------------");
            println!("{}", avg_loss);
            println!("{}", (1.0 - beta.powf(batch_num as f64)));
            plot_lr_vs_loss(&mut loss_by_lr, worst_loss);
            return;
        }

        loss_by_lr.push((lr.log10() as f32, avg_loss as f32));

        if avg_loss > worst_loss {
            worst_loss = avg_loss_smoothed;
        }

        lr *= multiplier;
        batch_num += 1;
    }
}

fn plot_lr_vs_loss(loss_by_lr: &mut Vec<(f32, f32)>, worst_loss: f64) {
    let root = BitMapBackend::new("data/lr_loss.png", (4000, 4000)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(-6f32..1f32, 0f32..(worst_loss as f32))
        .unwrap();

    chart.configure_mesh().x_labels(20).y_labels(5).draw().unwrap();
    chart.draw_series(LineSeries::new(loss_by_lr.clone(), &RED)).unwrap();
}
