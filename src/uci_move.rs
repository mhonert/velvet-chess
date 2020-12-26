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

use crate::board::Board;
use crate::move_gen::{Move};
use crate::pieces::{B, EMPTY, N, Q, R};

pub struct UCIMove {
    pub start: i8,
    pub end: i8,
    pub promotion: i8,
}

impl UCIMove {
    pub fn new(start: i8, end: i8, promotion: i8) -> Self {
        UCIMove {
            start,
            end,
            promotion,
        }
    }

    pub fn from_encoded_move(board: &Board, m: Move) -> Self {
        let current_piece = board.get_item(m.start() as i32).abs();
        let promotion = if m.piece_id() != current_piece {
            m.piece_id()
        } else {
            EMPTY
        };

        UCIMove {
            start: m.start() as i8,
            end: m.end() as i8,
            promotion,
        }
    }

    pub fn from_uci(uci: &str) -> Option<Self> {
        let bytes = uci.as_bytes();
        if bytes.len() < 4 {
            eprintln!("Invalid uci move notation: {}", uci);
            return None;
        }

        let start_col = bytes[0] - b'a';
        let start_row = b'8' - bytes[1];
        let start = (start_row * 8 + start_col) as i8;

        let end_col = bytes[2] - b'a';
        let end_row = b'8' - bytes[3];
        let end = (end_row * 8 + end_col) as i8;

        let promotion = if bytes.len() == 5 {
            match bytes[4] {
                b'q' => Q,
                b'r' => R,
                b'b' => B,
                b'n' => N,
                _ => {
                    eprintln!("Invalid promotion piece in UCI notation: {}", uci);
                    return None;
                }
            }
        } else {
            EMPTY
        };

        Some(UCIMove {
            start,
            end,
            promotion,
        })
    }

    pub fn to_uci(&self) -> String {
        let start_col = self.start & 7;
        let start_row = self.start / 8;
        let end_col = self.end & 7;
        let end_row = self.end / 8;

        let mut result = String::with_capacity(5);
        result.push(uci_col(start_col));
        result.push(uci_row(start_row));
        result.push(uci_col(end_col));
        result.push(uci_row(end_row));

        if self.promotion != EMPTY {
            result.push(uci_promotion(self.promotion));
        }

        result
    }

    pub fn to_move(&self, board: &Board) -> Move {
        let target_piece_id = if self.promotion != EMPTY {
            self.promotion
        } else {
            board.get_item(self.start as i32)
        };

        Move::new(target_piece_id, self.start as i32, self.end as i32)
    }
}

fn uci_col(col: i8) -> char {
    match col {
        0 => 'a',
        1 => 'b',
        2 => 'c',
        3 => 'd',
        4 => 'e',
        5 => 'f',
        6 => 'g',
        7 => 'h',
        _ => ' ',
    }
}

fn uci_row(row: i8) -> char {
    match row {
        0 => '8',
        1 => '7',
        2 => '6',
        3 => '5',
        4 => '4',
        5 => '3',
        6 => '2',
        7 => '1',
        _ => ' ',
    }
}

fn uci_promotion(promotion: i8) -> char {
    match promotion {
        Q => 'q',
        R => 'r',
        B => 'b',
        N => 'n',
        _ => ' ',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_standard_move() {
        let m = UCIMove::new(52, 36, EMPTY);
        assert_eq!("e2e4", m.to_uci());
    }

    #[test]
    fn write_promotion_move() {
        let m = UCIMove::new(8, 0, Q);
        assert_eq!("a7a8q", m.to_uci());
    }

    #[test]
    fn read_standard_move() {
        let m = UCIMove::from_uci("e2e4").unwrap();
        assert_eq!(52, m.start);
        assert_eq!(36, m.end);
        assert_eq!(EMPTY, m.promotion);
    }

    #[test]
    fn read_promotion_move() {
        let m = UCIMove::from_uci("a7a8q").unwrap();
        assert_eq!(8, m.start);
        assert_eq!(0, m.end);
        assert_eq!(Q, m.promotion);
    }
}
