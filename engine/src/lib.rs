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
pub mod gen_quiet_pos;
pub mod history_heuristics;
pub mod magics;
pub mod move_gen;
pub mod moves;
pub mod nn_eval;
pub mod perft;
pub mod random;
pub mod search;
pub mod uci;

mod boardpos;
mod castling;
mod options;
mod pieces;
mod pos_history;
mod score_util;
mod time_management;
mod transposition_table;
mod uci_move;
mod zobrist;
pub mod eval;