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

use crate::engine::Engine;
use crate::eval_search::EvalSearch;
use crate::moves::NO_MOVE;
use crate::colors::{WHITE, Color};
use crate::fen::write_fen;
use crate::uci_move::UCIMove;
use crate::random::Random;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn play_match(opening: &str, engines: &mut [&mut Engine]) -> Option<usize> {
    let mut player_color = WHITE;

    for engine in engines.iter_mut() {
        engine.reset();
        engine.set_position(opening, Vec::new());
    }

    loop {
        let idx = idx_by_color(player_color);

        let best_move = engines[idx].find_best_move_by_eval();
        if best_move == NO_MOVE {
            // println!("FEN: {}", write_fen(&engines[idx].board));

            if engines[idx].board.is_in_check(player_color) {
                return Some(idx_by_color(-player_color));
            } else {
                return None;
            }
        }

        if engines[idx].board.is_draw() {
            // println!("FEN: {}", write_fen(&engines[idx].board));
            return None;
        }

        engines[idx].board.perform_move(best_move);
        player_color = -player_color;

        engines[idx_by_color(player_color)].board.perform_move(best_move);
    }
}

fn idx_by_color(color: Color) -> usize {
    if color == WHITE {
        0
    } else {
        1
    }
}