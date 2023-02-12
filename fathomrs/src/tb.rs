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

use std::cmp::Ordering;
use std::ffi::{c_uint, CString};
use std::mem::transmute;
use crate::bindings::{TB_BLESSED_LOSS, TB_CURSED_WIN, TB_DRAW, tb_free, tb_init, TB_LARGEST, TB_LOSS, TB_MAX_MOVES, tb_probe_root_impl, tb_probe_wdl_impl, TB_PROMOTES_BISHOP, TB_PROMOTES_KNIGHT, TB_PROMOTES_NONE, TB_PROMOTES_QUEEN, TB_PROMOTES_ROOK, TB_RESULT_FAILED, TB_RESULT_FROM_MASK, TB_RESULT_FROM_SHIFT, TB_RESULT_PROMOTES_MASK, TB_RESULT_PROMOTES_SHIFT, TB_RESULT_TO_MASK, TB_RESULT_TO_SHIFT, TB_RESULT_WDL_MASK, TB_RESULT_WDL_SHIFT, TB_WIN};


#[repr(u32)]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum TBResult {
    Loss = TB_LOSS,
    BlessedLoss = TB_BLESSED_LOSS,
    Draw = TB_DRAW,
    CursedWin = TB_CURSED_WIN,
    Win = TB_WIN,
}

impl TBResult {
    pub fn from_result(result: u32) -> TBResult {
        unsafe { transmute((result & TB_RESULT_WDL_MASK) >> TB_RESULT_WDL_SHIFT) }
    }
}

pub fn extract_move(result: u32) -> (i8, i8, Promotion) {
    let from = (result & TB_RESULT_FROM_MASK) >> TB_RESULT_FROM_SHIFT;
    let to = (result & TB_RESULT_TO_MASK) >> TB_RESULT_TO_SHIFT;
    let promotion = Promotion::from((result & TB_RESULT_PROMOTES_MASK) >> TB_RESULT_PROMOTES_SHIFT);
    (from as i8, to as i8, promotion)
}

pub fn is_failed_result(result: u32) -> bool {
    result == TB_RESULT_FAILED
}

#[repr(u32)]
pub enum Promotion {
    Queen = TB_PROMOTES_QUEEN,
    Rook = TB_PROMOTES_ROOK,
    Bishop = TB_PROMOTES_BISHOP,
    Knight = TB_PROMOTES_KNIGHT,
    None = TB_PROMOTES_NONE,
}

impl From<u32> for Promotion {
    fn from(value: u32) -> Self {
        unsafe { transmute(value) }
    }
}

impl TBResult {
    pub fn invert(&self) -> TBResult {
        match self {
            TBResult::Loss => TBResult::Win,
            TBResult::BlessedLoss => TBResult::CursedWin,
            TBResult::Draw => TBResult::Draw,
            TBResult::CursedWin => TBResult::BlessedLoss,
            TBResult::Win => TBResult::Loss,
        }
    }

    fn score(&self) -> i32 {
        match self {
            TBResult::Loss => -1000,
            TBResult::BlessedLoss => -1,
            TBResult::Draw => 0,
            TBResult::CursedWin => 1,
            TBResult::Win => 1000,
        }
    }
}

impl PartialOrd for TBResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.score().partial_cmp(&other.score())
    }
}

pub fn init(path: String) -> bool {
    unsafe {
        let c_path = CString::new(path).unwrap_or_default();
        tb_init(c_path.as_ptr())
    }
}

pub fn free() {
    unsafe {
        tb_free()
    }
}

pub fn probe_wdl(white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64, ep: u16, turn: bool) -> TBResult {
    unsafe {
        let result = tb_probe_wdl_impl(white, black, kings, queens, rooks, bishops, knights, pawns, ep as c_uint, turn);
        transmute(result)
    }
}

pub fn probe_root_wdl(white: u64, black: u64, kings: u64, queens: u64, rooks: u64, bishops: u64, knights: u64, pawns: u64, rule50: u8, ep: u16, turn: bool) -> (u32, Vec<u32>) {
    unsafe {
        let mut moves: Vec<u32> = Vec::with_capacity(TB_MAX_MOVES as usize);

        let result = tb_probe_root_impl(white, black, kings, queens, rooks, bishops, knights, pawns,
                          rule50 as c_uint, ep as c_uint, turn, moves.spare_capacity_mut().as_mut_ptr().cast());

        if !is_failed_result(result) {
            moves.set_len(TB_MAX_MOVES as usize);
            let count = moves.iter().position(|&m| m == TB_RESULT_FAILED).unwrap_or_default();
            moves.set_len(count);
        }

        (result, moves)
    }
}

pub fn max_piece_count() -> u32 {
    unsafe { TB_LARGEST }
}