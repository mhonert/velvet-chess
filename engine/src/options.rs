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

// Auto-generated file (see tools/tuning/gencode.py)

use std::sync::mpsc::Sender;
use crate::engine::Message;

const QS_SEE_THRESHOLD: i32 = 104;
const QS_PRUNE_MARGIN: i32 = 650;
const TIMEEXT_SCORE_DROP_THRESHOLD: i32 = 20;
const TIMEEXT_HISTORY_SIZE: i32 = 6;
const RAZOR_MARGIN: i32 = 130;
    
pub struct Options {}
    
impl Options {
    pub fn new() -> Self {
        Options{}
    }
    
    pub fn set_option(&mut self, name: String, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    
    pub fn set_array_option(&mut self, name: String, _: usize, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    

    #[inline]
    pub fn get_qs_see_threshold(&self) -> i32 {
        QS_SEE_THRESHOLD
    }


    #[inline]
    pub fn get_qs_prune_margin(&self) -> i32 {
        QS_PRUNE_MARGIN
    }


    #[inline]
    pub fn get_timeext_score_drop_threshold(&self) -> i32 {
        TIMEEXT_SCORE_DROP_THRESHOLD
    }


    #[inline]
    pub fn get_timeext_history_size(&self) -> i32 {
        TIMEEXT_HISTORY_SIZE
    }


    #[inline]
    pub fn get_razor_margin(&self) -> i32 {
        RAZOR_MARGIN
    }

}

pub fn parse_set_option(_: &Sender<Message>, _: &str, _: &str) {}
    