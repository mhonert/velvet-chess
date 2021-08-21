/*
 * Velvet Chess Engine
 * Copyright (C) 2020 mhonert (https://github.com/mhonert)
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

pub mod bitboard;
pub mod board;
pub mod colors;
pub mod engine;
pub mod fen;
pub mod history_heuristics;
pub mod magics;
pub mod move_gen;
pub mod moves;
pub mod nn_eval;
pub mod perft;
pub mod pieces;
pub mod random;
pub mod scores;
pub mod search;
pub mod transposition_table;
pub mod uci;

mod pos_history;
mod time_management;
mod uci_move;
mod zobrist;