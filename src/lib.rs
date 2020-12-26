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

pub mod board;
pub mod engine;
pub mod eval;
pub mod fen;
pub mod perft;
pub mod uci;

mod bitboard;
mod boardpos;
mod castling;
mod colors;
mod options;
mod history_heuristics;
mod moves;
mod move_gen;
mod move_sort;
mod pieces;
mod pos_history;
mod random;
mod score_util;
mod search;
mod transposition_table;
mod uci_move;
mod zobrist;