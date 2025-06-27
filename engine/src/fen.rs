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

use crate::board::castling::{Castling, CastlingRules, CastlingState};
use crate::board::{BlackBoardPos, Board, WhiteBoardPos};
use crate::colors::{Color, BLACK, WHITE};
use crate::pieces;
use crate::pieces::K;
use std::error::Error;
use std::fmt;
use std::process::exit;

pub const START_POS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[derive(Debug)]
pub struct FenError {
    msg: String,
}

pub struct FenParseResult {
    pub pieces: [i8; 64],
    pub active_player: Color,
    castling_rules: CastlingRules,
    pub castling_state: CastlingState,
    enpassant_target: Option<i8>,
    halfmove_clock: u8,
    fullmove_num: u16,
}

impl Error for FenError {}

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FEN error: {}", self.msg)
    }
}

pub fn read_fen(board: &mut Board, fen: &str) -> Result<(), FenError> {
    match parse_fen(fen) {
        Err(e) => Err(e),
        Ok(FenParseResult {
            pieces,
            active_player,
            castling_rules,
            castling_state,
            enpassant_target,
            halfmove_clock,
            fullmove_num,
        }) => {
            board.set_position(
                &pieces,
                active_player,
                castling_state,
                enpassant_target,
                halfmove_clock,
                fullmove_num,
                castling_rules,
            );
            Ok(())
        }
    }
}

pub fn parse_fen(fen: &str) -> Result<FenParseResult, FenError> {
    let mut fen_parts = fen.split(' ');

    let (pieces, white_king, black_king) = match fen_parts.next().and_then(read_pieces) {
        Some((pieces, white_king, black_king)) => (pieces, white_king, black_king),
        None => return Err(FenError { msg: format!("Error in piece part: {fen}") }),
    };

    let active_player = match fen_parts.next().and_then(read_color) {
        Some(color) => color,
        None => return Err(FenError { msg: format!("Error in active player part: {fen}") }),
    };

    let (castling_rules, castling_state) =
        match fen_parts.next().and_then(|castling| read_castling(castling, white_king & 7, black_king & 7)) {
            Some((castling_rules, castling_state)) => (castling_rules, castling_state),
            None => return Err(FenError { msg: format!("Error in castling part: {fen}") }),
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

    Ok(FenParseResult {
        pieces,
        active_player,
        castling_rules,
        castling_state,
        enpassant_target,
        halfmove_clock,
        fullmove_num,
    })
}

pub fn create_from_fen(fen: &str) -> Board {
    let mut items: [i8; 64] = [0; 64];
    items[4] = K;
    items[60] = -K;
    let mut board = Board::new(&items, WHITE, CastlingState::default(), None, 0, 1, CastlingRules::default());
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

fn read_pieces(piece_placements: &str) -> Option<([i8; 64], i8, i8)> {
    let mut pieces: [i8; 64] = [0; 64];
    let mut white_king = 0;
    let mut black_king = 0;

    let mut idx = 64 + 8;
    for piece_row in piece_placements.split('/') {
        idx -= 16;
        for piece in piece_row.chars() {
            if ('1'..='8').contains(&piece) {
                let empty_squares = piece.to_digit(10)?;

                for _ in 1..=empty_squares {
                    pieces[idx] = pieces::EMPTY;
                    idx += 1;
                }
                continue;
            }

            let piece_id = match PIECE_FEN_CHARS.find(piece) {
                Some(piece) => piece as i8 - 6,
                None => return None,
            };
            pieces[idx] = piece_id;
            idx += 1;

            if piece_id == K {
                white_king = (idx - 1) as i8;
            } else if piece_id == -K {
                black_king = (idx - 1) as i8;
            }
        }
    }

    Some((pieces, white_king, black_king))
}

fn read_color(color: &str) -> Option<Color> {
    match color {
        "w" => Some(WHITE),
        "b" => Some(BLACK),
        _ => None,
    }
}

fn read_castling(castling: &str, w_king_col: i8, b_king_col: i8) -> Option<(CastlingRules, CastlingState)> {
    let mut state = CastlingState::default();
    let mut is_chess960 = false;
    let mut w_king_side_rook_col = 7;
    let mut w_queen_side_rook_col = 0;
    let mut w_start_king_col = 0;

    let mut b_king_side_rook_col = 7;
    let mut b_queen_side_rook_col = 0;
    let mut b_start_king_col = 0;

    for ch in castling.bytes() {
        match ch {
            b'K' => state.set_can_castle(Castling::WhiteKingSide),
            b'Q' => state.set_can_castle(Castling::WhiteQueenSide),
            b'k' => state.set_can_castle(Castling::BlackKingSide),
            b'q' => state.set_can_castle(Castling::BlackQueenSide),
            b'A'..=b'H' => {
                is_chess960 = true;
                w_start_king_col = w_king_col;
                let rook_col = (ch - b'A') as i8;
                if rook_col < w_start_king_col {
                    state.set_can_castle(Castling::WhiteQueenSide);
                    w_queen_side_rook_col = rook_col;
                } else {
                    state.set_can_castle(Castling::WhiteKingSide);
                    w_king_side_rook_col = rook_col;
                }
            }
            b'a'..=b'h' => {
                is_chess960 = true;
                b_start_king_col = b_king_col;
                let rook_col = (ch - b'a') as i8;
                if rook_col < b_start_king_col {
                    state.set_can_castle(Castling::BlackQueenSide);
                    b_queen_side_rook_col = rook_col;
                } else {
                    state.set_can_castle(Castling::BlackKingSide);
                    b_king_side_rook_col = rook_col;
                }
            }
            b'-' => (),
            _ => return None,
        }
    }

    let rules = if is_chess960 {
        CastlingRules::new(
            true,
            w_start_king_col,
            w_king_side_rook_col,
            w_queen_side_rook_col,
            b_start_king_col,
            b_king_side_rook_col,
            b_queen_side_rook_col,
        )
    } else {
        CastlingRules::default()
    };

    Some((rules, state))
}

fn read_enpassant(en_passant: &str) -> Option<i8> {
    if en_passant == "-" {
        return None;
    }

    if en_passant.len() != 2 {
        return None;
    }

    let mut bytes = en_passant.bytes();
    let (col_char, row_char) = (bytes.next()?, bytes.next()?);

    let col_offset = (col_char.wrapping_sub(b'a')) as i8;

    Some(match row_char {
        b'6' => WhiteBoardPos::EnPassantLineStart as i8 + col_offset,
        b'3' => BlackBoardPos::EnPassantLineStart as i8 + col_offset,
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
        let item = board.get_item(pos ^ 56);
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
            _ => panic!("Unexpected piece ID {item}"),
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
    if color.is_white() {
        "w"
    } else {
        "b"
    }
}

fn write_castling(board: &Board) -> String {
    let mut result = String::new();

    if board.castling_rules.is_chess960() {
        if board.can_castle(Castling::WhiteKingSide) {
            result.push(char::from(b'A' + (board.castling_rules.ks_rook_start(WHITE) & 7) as u8));
        }

        if board.can_castle(Castling::WhiteQueenSide) {
            result.push(char::from(b'A' + (board.castling_rules.qs_rook_start(WHITE) & 7) as u8));
        }

        if board.can_castle(Castling::BlackKingSide) {
            result.push(char::from(b'a' + (board.castling_rules.ks_rook_start(BLACK) & 7) as u8));
        }

        if board.can_castle(Castling::BlackQueenSide) {
            result.push(char::from(b'a' + (board.castling_rules.qs_rook_start(BLACK) & 7) as u8));
        }
    } else {
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
    }

    if result.is_empty() {
        String::from("-")
    } else {
        result
    }
}

fn write_enpassant(board: &Board) -> String {
    let en_passant = board.enpassant_target();
    if en_passant != 0 {
        let col = en_passant % 8;
        let col_letter = b'a' + col;
        let col_str = String::from_utf8(vec![col_letter]).expect("Could not convert columm letter");
        
        let row_str = if en_passant <= BlackBoardPos::EnPassantLineEnd as u8 { "3" } else { "6" };
        return col_str + row_str;
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
        test_fen_with_en_passant("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq a3 0 1", 16);
        test_fen_with_en_passant("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq b3 0 1", 17);
        test_fen_with_en_passant("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR b KQkq h3 0 1", 23);
    }

    #[test]
    fn read_write_white_en_passant() {
        test_fen_with_en_passant("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq a6 0 1", 40);
        test_fen_with_en_passant("rnbqkbnr/p1pppppp/8/8/Pp6/8/1PPPPPPP/RNBQKBNR w KQkq g6 0 1", 46);
        test_fen_with_en_passant("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq h6 0 1", 47);
    }

    #[test]
    fn read_write_halfmove_clock() {
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR w KQkq - 2 4");
    }

    #[test]
    fn read_write_fullmove_count() {
        test_fen("rnbqkbnr/ppppppp1/8/6Pp/8/8/PPPPPP1P/RNBQKBNR b KQkq - 2 4");
    }

    #[test]
    fn read_write_chess960_fen() {
        test_fen("qrbnnkrb/pppppppp/8/8/8/8/PPPPPPPP/QRBNNKRB w GBgb - 0 1");
        test_fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9");
        test_fen("rkqbr1bn/p2ppppp/1pp2n2/8/5P2/3P1N2/PPP1PRPP/RKQB2BN w Aa - 3 9");
    }

    #[test]
    fn read_write_dfrc_fen() {
        test_fen("qrkbbrnn/pppppppp/8/8/8/8/PPPPPPPP/BQNNRKRB w GEfb - 0 1");
        test_fen("nbnrkqbr/pppppppp/8/8/8/8/PPPPPPPP/RKRNNQBB w CAhd - 0 1");
        test_fen("rnknrqbb/pppppppp/8/8/8/8/PPPPPPPP/BBNRQNKR w HDea - 0 1");
    }

    fn test_fen(fen: &str) {
        assert_eq!(write_fen(&create_from_fen(fen)), fen);
    }

    fn test_fen_with_en_passant(fen: &str, expected_en_passant_state: u8) {
        let board = create_from_fen(fen);
        assert_eq!(write_fen(&board), fen);
        assert_eq!(board.enpassant_target(), expected_en_passant_state);
    }
}
