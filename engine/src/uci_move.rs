/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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

use crate::board::castling::CastlingRules;
use crate::board::Board;
use crate::moves::{Move, MoveType};
use crate::pieces::{B, EMPTY, K, N, P, Q, R};

pub struct UCIMove {
    pub start: i8,
    pub end: i8,
    pub promotion: i8,
}

impl UCIMove {
    pub fn new(start: i8, end: i8, promotion: i8) -> Self {
        UCIMove { start, end, promotion }
    }

    pub fn from_move(board: &Board, m: Move) -> String {
        let mut end = m.end();
        let color = board.active_player();

        if !board.castling_rules.is_chess960() {
            if matches!(m.move_type(), MoveType::KingKSCastling) {
                end = CastlingRules::ks_king_end(color);
            } else if matches!(m.move_type(), MoveType::KingQSCastling) {
                end = CastlingRules::qs_king_end(color);
            }
        }

        let mut result = String::with_capacity(5);
        result.push(uci_col(m.start() & 7));
        result.push(uci_row(m.start() / 8));
        result.push(uci_col(end & 7));
        result.push(uci_row(end / 8));

        if m.move_type().is_promotion() {
            result.push(uci_promotion(m.move_type().piece_id()));
        };

        result
    }

    pub fn from_uci(uci: &str) -> Option<Self> {
        let bytes = uci.as_bytes();
        if bytes.len() < 4 {
            eprintln!("Invalid uci move notation: {}", uci);
            return None;
        }

        let start_col = bytes[0] - b'a';
        let start_row = b'8' - bytes[1];
        let start = ((7 - start_row) * 8 + start_col) as i8;

        let end_col = bytes[2] - b'a';
        let end_row = b'8' - bytes[3];
        let end = ((7 - end_row) * 8 + end_col) as i8;

        let promotion = if bytes.len() == 5 {
            match bytes[4] {
                b'q' => Q,
                b'r' => R,
                b'b' => B,
                b'n' => N,
                _ => {
                    eprintln!("Invalid promotion piece in UCI notation: '{uci}'");
                    return None;
                }
            }
        } else {
            EMPTY
        };

        Some(UCIMove { start, end, promotion })
    }

    pub fn to_move(&self, board: &Board) -> Move {
        let start = self.start;
        let end = self.end;
        let is_capture = board.get_item(end as usize) != EMPTY;

        let start_piece_id = board.get_item(start as usize).abs();
        match start_piece_id {
            P => {
                if (start - end).abs() == 16 {
                    Move::new(MoveType::PawnDoubleQuiet, start, end)
                } else if self.promotion != EMPTY && self.promotion != start_piece_id {
                    if is_capture {
                        Move::new(MoveType::new_capture_promotion(self.promotion), start, end)
                    } else {
                        Move::new(MoveType::new_quiet_promotion(self.promotion), start, end)
                    }
                } else if (start - end).abs() == 8 {
                    Move::new(MoveType::PawnQuiet, start, end)
                } else if !is_capture {
                    Move::new(MoveType::PawnEnPassant, start, end)
                } else {
                    Move::new(MoveType::PawnCapture, start, end)
                }
            }

            K => {
                let color = board.active_player();
                if board.castling_rules.is_king_start(color, start) {
                    if board.castling_rules.is_chess960() {
                        if board.can_castle_king_side(color) && board.castling_rules.is_ks_castling(color, end) {
                            return Move::new(MoveType::KingKSCastling, start, board.castling_rules.ks_rook_start(color));
                        } else if board.can_castle_queen_side(color)
                            && board.castling_rules.is_qs_castling(color, end)
                        {
                            return Move::new(MoveType::KingQSCastling, start, board.castling_rules.qs_rook_start(color));
                        }
                    } else {
                        if board.can_castle_king_side(color) && end == CastlingRules::ks_king_end(color) {
                            return Move::new(MoveType::KingKSCastling, start, board.castling_rules.ks_rook_start(color));
                        } else if board.can_castle_queen_side(color) && end == CastlingRules::qs_king_end(color) {
                            return Move::new(MoveType::KingQSCastling, start, board.castling_rules.qs_rook_start(color));
                        }
                    }
                }

                if is_capture {
                    Move::new(MoveType::KingCapture, start, end)
                } else {
                    Move::new(MoveType::KingQuiet, start, end)
                }
            }

            _ => {
                if is_capture {
                    Move::new(MoveType::new_capture(start_piece_id), start, end)
                } else {
                    Move::new(MoveType::new_quiet(start_piece_id), start, end)
                }
            }
        }
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
        0 => '1',
        1 => '2',
        2 => '3',
        3 => '4',
        4 => '5',
        5 => '6',
        6 => '7',
        7 => '8',
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
    use crate::board::castling::{CastlingRules, CastlingState};
    use crate::colors::WHITE;
    use crate::fen::{create_from_fen, START_POS};
    use crate::pieces::{K, P};

    #[test]
    fn write_standard_move() {
        let board = create_from_fen(START_POS);
        let m = UCIMove::new(12, 28, EMPTY);

        assert_eq!("e2e4", UCIMove::from_move(&board, m.to_move(&board)));
    }

    #[test]
    fn write_promotion_move() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  K,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            P,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0, -K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());

        let m = UCIMove::new(48, 56, Q);
        assert_eq!("a7a8q", UCIMove::from_move(&board, m.to_move(&board)));
    }

    #[test]
    fn write_castling_move() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  K,  0,  0,  R,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0, -K,  0,  0,  0,  0,
        ];

        let board = Board::new(&items, WHITE, CastlingState::ALL, None, 0, 1, CastlingRules::default());

        let uci_move = UCIMove::from_move(&board, Move::new(MoveType::KingKSCastling, 4, 7));
        assert_eq!("e1g1", uci_move);
    }

    #[test]
    fn write_castling_move_chess960() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  K,  0,  0,  R,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0, -K,  0,  0,  0,  0,
        ];

        let board =
            Board::new(&items, WHITE, CastlingState::ALL, None, 0, 1, CastlingRules::new(true, 4, 7, 0, 4, 7, 0));

        let uci_move = UCIMove::from_move(&board, Move::new(MoveType::KingKSCastling, 4, 7));
        assert_eq!("e1h1", uci_move);
    }

    #[test]
    fn read_castling_move_chess960() {
        #[rustfmt::skip]
        let items: [i8; 64] = [
            0,  0,  0,  0,  K,  0,  0,  R,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  P,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0, -P,  0,  0,  0,  0,
            0,  0,  0, -K,  0,  0,  0,  0,
        ];

        let board =
            Board::new(&items, WHITE, CastlingState::ALL, None, 0, 1, CastlingRules::new(true, 4, 7, 0, 4, 7, 0));

        let m = UCIMove::from_uci("e1h1").unwrap().to_move(&board);
        assert_eq!(m.end(), board.castling_rules.ks_rook_start(WHITE));
        assert!(matches!(m.move_type(), MoveType::KingKSCastling));
    }

    #[test]
    fn read_standard_move() {
        let m = UCIMove::from_uci("e2e4").unwrap();
        assert_eq!(12, m.start);
        assert_eq!(28, m.end);
        assert_eq!(EMPTY, m.promotion);
    }

    #[test]
    fn read_promotion_move() {
        let m = UCIMove::from_uci("a7a8q").unwrap();
        assert_eq!(48, m.start);
        assert_eq!(56, m.end);
        assert_eq!(Q, m.promotion);
    }
}
