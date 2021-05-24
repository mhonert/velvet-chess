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
use crate::boardpos::{BlackBoardPos, WhiteBoardPos};
use crate::castling::Castling;
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces;
use std::error::Error;
use std::fmt;
use std::process::exit;

pub const START_POS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Debug)]
pub struct FenError {
    msg: String,
}

impl Error for FenError {}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FEN error: {}", self.msg)
    }
}

pub fn read_fen(board: &mut Board, fen: &str) -> Result<(), FenError> {
    let mut fen_parts = fen.split(' ');

    let pieces = match fen_parts.next().and_then(read_pieces) {
        Some(pieces) => pieces,
        None => {
            return Result::Err(FenError {
                msg: format!("Error in piece part: {}", fen),
            })
        }
    };

    let active_player = match fen_parts.next().and_then(read_color) {
        Some(color) => color,
        None => {
            return Result::Err(FenError {
                msg: format!("Error in active player part: {}", fen),
            })
        }
    };

    let castling_state = match fen_parts.next().and_then(read_castling) {
        Some(castling) => castling,
        None => {
            return Result::Err(FenError {
                msg: format!("Error in castling part: {}", fen),
            })
        }
    };

    let enpassant_target = fen_parts.next().and_then(read_enpassant);

    let halfmove_clock: u8 = match fen_parts.next() {
        Some(halfmoves) => halfmoves.parse().unwrap(),
        None => 0,
    };

    let fullmove_num: u16 = match fen_parts.next() {
        Some(fullmoves) => fullmoves.parse().unwrap(),
        None => 0,
    };

    board.set_position(
        &pieces,
        active_player,
        castling_state,
        enpassant_target,
        halfmove_clock,
        fullmove_num
    );

    Result::Ok(())
}

pub fn create_from_fen(fen: &str) -> Board {
    let items: [i8; 64] = [0; 64];
    let mut board = Board::new(&items, WHITE, 0, None, 0, 1);
    match read_fen(&mut board, fen) {
        Ok(_) => board,
        Err(_) => {
            eprintln!("Could not create board from FEN: {}", fen);
            exit(-1)
        }
    }
}

// Black piece IDs go from -6 to -1, white piece IDs from 1 to 6
// add 6 to get the index to the FEN character for the piece:
const PIECE_FEN_CHARS: &str = "kqrbnp/PNBRQK";

fn read_pieces(piece_placements: &str) -> Option<Vec<i8>> {
    let mut pieces: Vec<i8> = Vec::new();

    for piece_row in piece_placements.split('/') {
        for piece in piece_row.chars() {
            if piece >= '1' && piece <= '8' {
                let empty_squares = match piece.to_digit(10) {
                    Some(chars) => chars,
                    None => return None,
                };

                for _ in 1..=empty_squares {
                    pieces.push(0)
                }
                continue;
            }

            let piece_id = match PIECE_FEN_CHARS.find(piece) {
                Some(piece) => piece as i8 - 6,
                None => return None,
            };
            pieces.push(piece_id);
        }
    }

    Some(pieces)
}

fn read_color(color: &str) -> Option<Color> {
    match color {
        "w" => Some(WHITE),
        "b" => Some(BLACK),
        _ => None,
    }
}

fn read_castling(castling: &str) -> Option<u8> {
    let mut state: u8 = 0;
    for ch in castling.bytes() {
        match ch {
            b'K' => state |= Castling::WhiteKingSide as u8,
            b'Q' => state |= Castling::WhiteQueenSide as u8,
            b'k' => state |= Castling::BlackKingSide as u8,
            b'q' => state |= Castling::BlackQueenSide as u8,
            b'-' => (),
            _ => return None,
        }
    }
    Some(state)
}

fn read_enpassant(en_passant: &str) -> Option<i8> {
    if en_passant == "-" {
        return None;
    }

    if en_passant.len() != 2 {
        return None;
    }

    let mut bytes = en_passant.bytes();
    let (col_char, row_char) = (bytes.next().unwrap(), bytes.next().unwrap());

    let col_offset = (col_char.wrapping_sub(b'a')) as i8;

    Some(match row_char {
        b'3' => WhiteBoardPos::PawnLineStart as i8 + col_offset,
        b'6' => BlackBoardPos::PawnLineStart as i8 + col_offset,
        _ => return None,
    })
}

pub fn write_fen(board: &Board) -> String {
    write_pieces(board)
        + " "
        + write_color(board.active_player())
        + " "
        + write_castling(board).as_str()
        + " "
        + write_enpassant(board).as_str()
        + " "
        + board.halfmove_clock().to_string().as_str()
        + " "
        + board.fullmove_count().to_string().as_str()
}

fn write_pieces(board: &Board) -> String {
    let mut result = String::new();

    let mut empty_count = 0;
    for pos in 0..64 {
        let item = board.get_item(pos);
        if item == pieces::EMPTY {
            empty_count += 1;
            if pos % 8 == 7 {
                result += empty_count.to_string().as_str();
                if pos != 63 {
                    result.push('/');
                }
                empty_count = 0;
            }
            continue;
        }

        if empty_count > 0 {
            result += empty_count.to_string().as_str();
            empty_count = 0;
        }

        let piece = match item.abs() {
            pieces::P => "P",
            pieces::N => "N",
            pieces::B => "B",
            pieces::R => "R",
            pieces::Q => "Q",
            pieces::K => "K",
            _ => panic!("Unexpected piece ID {}", item),
        };

        if item < 0 {
            result += piece.to_lowercase().as_str();
        } else {
            result += piece;
        }

        if pos != 63 && pos % 8 == 7 {
            result.push('/');
        }
    }

    result
}

fn write_color(color: Color) -> &'static str {
    match color {
        WHITE => "w",
        BLACK => "b",
        _ => panic!("Unexpected color {}", color),
    }
}

fn write_castling(board: &Board) -> String {
    let mut result = String::new();

    if board.can_castle(Castling::WhiteKingSide) {
        result.push('K');
    }

    if board.can_castle(Castling::WhiteQueenSide) {
        result.push('Q');
    }

    if board.can_castle(Castling::BlackKingSide) {
        result.push('k');
    }

    if board.can_castle(Castling::BlackQueenSide) {
        result.push('q');
    }

    if result.is_empty() {
        String::from("-")
    } else {
        result
    }
}

fn write_enpassant(board: &Board) -> String {
    for pos in WhiteBoardPos::EnPassantLineStart as u8..=WhiteBoardPos::EnPassantLineEnd as u8 {
        if board.can_enpassant(WHITE, pos) {
            let col = pos % 8;
            let col_letter = b'a' + col;
            let col_str =
                String::from_utf8(vec![col_letter]).expect("Could not convert columm letter");
            return col_str + "6";
        }
    }

    for pos in BlackBoardPos::EnPassantLineStart as u8..=BlackBoardPos::EnPassantLineEnd as u8 {
        if board.can_enpassant(BLACK, pos) {
            let col = pos % 8;
            let col_letter = b'a' + col;
            let col_str =
                String::from_utf8(vec![col_letter]).expect("Could not convert columm letter");
            return col_str + "3";
        }
    }

    String::from("-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_write_startpos() {
        test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    #[test]
    fn read_write_active_player() {
        test_fen("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1");
    }

    #[test]
    fn read_write_no_castling() {
        test_fen("r4k1r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4K1R w - - 0 1");
    }

    #[test]
    fn read_write_only_white_castling() {
        test_fen("r4k1r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQ - 0 1");
    }

    #[test]
    fn read_write_only_black_castling() {
        test_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R4K1R w kq - 0 1");
    }

    #[test]
    fn read_write_only_kingside_castling() {
        test_fen("1r2k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/1R2K2R w Kk - 0 1");
    }

    #[test]
    fn read_write_only_queenside_castling() {
        test_fen("r3k1r1/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K1R1 w Qq - 0 1");
    }

    #[test]
    fn read_write_black_en_passant() {
        test_fen("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq a3 0 1");
        test_fen("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq b3 0 1");
        test_fen("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq h3 0 1");
    }

    #[test]
    fn read_write_white_en_passant() {
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq a6 0 1");
        test_fen("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR w KQkq g6 0 1");
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq h6 0 1");
    }

    #[test]
    fn read_write_halfmove_clock() {
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq - 2 4");
    }

    #[test]
    fn read_write_fullmove_count() {
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR b KQkq - 2 4");
    }

    fn test_fen(fen: &str) {
        assert_eq!(write_fen(&create_from_fen(fen)), fen);
    }
}
