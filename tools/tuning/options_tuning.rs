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

// Auto-generated file (see tools/tuning/gencode.py)

use std::cmp::max;
use std::sync::mpsc::Sender;
use crate::engine::Message;
use std::str::FromStr;
    
pub struct Options {
    center_control_bonus: i32,
    passed_pawn_bonus: [i32; 4],
    eg_passed_pawn_bonus: [i32; 4],
    passed_pawn_king_defense_bonus: [i32; 8],
    passed_pawn_king_attacked_penalty: [i32; 8],
}
    
impl Options {
    pub fn new() -> Self {
        Options{
            center_control_bonus: 0,
            passed_pawn_bonus: [127, 10, 0, 0],
            eg_passed_pawn_bonus: [265, 193, 83, 26],
            passed_pawn_king_defense_bonus: [0, 82, 59, 21, 10, 7, 0, 0],
            passed_pawn_king_attacked_penalty: [0, 128, 91, 48, 21, 0, 0, 0],
        }
    }
    
    pub fn set_option(&mut self, name: String, value: i32) {
        match name.as_str() {
            "centercontrolbonus" => self.set_center_control_bonus(value),
            _ => println!("Unknown option {}", name)
        }
    }
    
    pub fn set_array_option(&mut self, name: String, index: usize, value: i32) {
        match name.as_str() {
            "passedpawnbonus" => self.set_passed_pawn_bonus(index, value),
            "egpassedpawnbonus" => self.set_eg_passed_pawn_bonus(index, value),
            "passedpawnkingdefensebonus" => self.set_passed_pawn_king_defense_bonus(index, value),
            "passedpawnkingattackedpenalty" => self.set_passed_pawn_king_attacked_penalty(index, value),
            _ => println!("Unknown option {}", name)
        }
    }
    
    fn set_center_control_bonus(&mut self, value: i32) {
        self.center_control_bonus = value;
    }

    #[inline]
    pub fn get_center_control_bonus(&self) -> i32 {
        self.center_control_bonus
    }

    fn set_passed_pawn_bonus(&mut self, index: usize, value: i32) {
        self.passed_pawn_bonus[index] = value;
    }

    #[inline]
    pub fn get_passed_pawn_bonus(&self, index: usize) -> i32 {
        self.passed_pawn_bonus[index]
    }
    
    fn set_eg_passed_pawn_bonus(&mut self, index: usize, value: i32) {
        self.eg_passed_pawn_bonus[index] = value;
    }

    #[inline]
    pub fn get_eg_passed_pawn_bonus(&self, index: usize) -> i32 {
        self.eg_passed_pawn_bonus[index]
    }
    
    fn set_passed_pawn_king_defense_bonus(&mut self, index: usize, value: i32) {
        self.passed_pawn_king_defense_bonus[index] = value;
    }

    #[inline]
    pub fn get_passed_pawn_king_defense_bonus(&self, index: usize) -> i32 {
        self.passed_pawn_king_defense_bonus[index]
    }
    
    fn set_passed_pawn_king_attacked_penalty(&mut self, index: usize, value: i32) {
        self.passed_pawn_king_attacked_penalty[index] = value;
    }

    #[inline]
    pub fn get_passed_pawn_king_attacked_penalty(&self, index: usize) -> i32 {
        self.passed_pawn_king_attacked_penalty[index]
    }
    
}

const SINGLE_VALUE_OPTION_NAMES: [&'static str; 1] = ["centercontrolbonus"];
const MULTI_VALUE_OPTION_NAMES: [&'static str; 4] = ["passedpawnbonus", "egpassedpawnbonus", "passedpawnkingdefensebonus", "passedpawnkingattackedpenalty"];

pub fn parse_set_option(tx: &Sender<Message>, name: &str, value_str: &str) {
    if SINGLE_VALUE_OPTION_NAMES.contains(&name) {
        set_option_value(tx, name, value_str);
        return;
    }

    let name_without_index = name.replace(&['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][..], "");
    if MULTI_VALUE_OPTION_NAMES.contains(&name_without_index.as_str()) {
        set_array_option_value(tx, name_without_index.as_str(), name, value_str);
        return;
    }
}

fn set_option_value(tx: &Sender<Message>, name: &str, value_str: &str) {
    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetOption(String::from(name), value));
}

fn set_array_option_value(tx: &Sender<Message>, name: &str, name_with_index: &str, value_str: &str) {
    let index = match i32::from_str(&name_with_index[name.len()..]) {
        Ok(index) => index,
        Err(_) => {
            println!("Invalid index: {}", name_with_index);
            return;
        }
    };

    let value = match i32::from_str(value_str) {
        Ok(value) => value,
        Err(_) => {
            println!("Invalid int value: {}", value_str);
            return;
        }
    };

    send_message(tx, Message::SetArrayOption(String::from(name), index, value));
}

fn send_message(tx: &Sender<Message>, msg: Message) {
    match tx.send(msg) {
        Ok(_) => return,
        Err(err) => {
            eprintln!("could not send message to engine thread: {}", err);
        }
    }
}
