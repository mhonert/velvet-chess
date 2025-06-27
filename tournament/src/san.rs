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
use velvet::moves::{Move, MoveType};
use velvet::pieces::P;

static PIECES: [char; 7] = ['_', 'P', 'N', 'B', 'R', 'Q', 'K'];
static FILES: [char; 8] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
static RANKS: [char; 8] = ['1', '2', '3', '4', '5', '6', '7', '8'];

// Convert a Move to a SAN (Standard Algebraic Notation) string
pub fn move_to_san(m: Move, all_moves: &[Move], gives_check: bool) -> String {
    if matches!(m.move_type(), MoveType::KingKSCastling) {
        return "O-O".to_string();
    } else if matches!(m.move_type(), MoveType::KingQSCastling) {
        return "O-O-O".to_string();
    }
    
    let target_piece = m.move_type().piece_id();
    let source_piece = if m.move_type().is_promotion() { P } else { target_piece };
    let (start_file_required, start_rank_required) = determine_required_disambiguation(m, all_moves);
    let (start_file, start_rank) = get_file_rank(m.start());
    let (end_file, end_rank) = get_file_rank(m.end());
    let piece = PIECES[source_piece as usize];

    let mut san = String::new();
    if source_piece != P {
        san.push(piece);
    }

    if start_file_required || m.move_type().is_pawn_capture() {
        san.push(FILES[start_file as usize]);
    }

    if start_rank_required {
        san.push(RANKS[start_rank as usize]);
    }

    if m.move_type().is_capture() {
        san.push('x');
    }

    san.push(FILES[end_file as usize]);
    san.push(RANKS[end_rank as usize]);

    if m.is_promotion() {
        let promotion_piece = PIECES[target_piece as usize];
        san.push('=');
        san.push(promotion_piece);
    }

    if gives_check {
        san.push('+');
    }

    san
}

fn determine_required_disambiguation(m: Move, all_moves: &[Move]) -> (bool, bool) {
    if m.is_promotion() {
        return (false, false)
    }

    let start = m.start();
    let (start_file, start_rank) = get_file_rank(start);

    let end = m.end();
    let piece = m.move_type().piece_id();
    let mut file_required = false;
    let mut rank_required = false;

    for &om in all_moves {
        if om == m {
            continue;
        }
        if om.end() != end || om.move_type().piece_id() != piece {
            continue;
        }

        let (om_file, om_rank) = get_file_rank(om.start());
        if om_file != start_file {
            file_required = true;
        }
        if om_rank != start_rank {
            rank_required = true;
        }
    }

    (file_required, rank_required)
}

fn get_file_rank(pos: i8) -> (i8, i8) {
    (pos % 8, pos / 8)
}

#[cfg(test)]
mod tests {
    use velvet::moves::MoveType;
    use super::*;

    #[test]
    fn test_move_to_san_pawn_move() {
        let m = Move::new(MoveType::PawnDoubleQuiet, pos("e2"), pos("e4"));
        let all_moves = vec![m];
        assert_eq!(m.start(), 12);
        assert_eq!(m.end(), 28);

        assert_eq!(move_to_san(m, &all_moves, false), "e4");
    }

    #[test]
    fn test_move_to_san_pawn_capture() {
        let m = Move::new(MoveType::PawnCapture, pos("e4"), pos("d5"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "exd5");
    }

    #[test]
    fn test_move_to_san_pawn_en_passant() {
        let m = Move::new(MoveType::PawnEnPassant, pos("e5"), pos("d6"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "exd6");
    }

    #[test]
    fn test_move_to_san_capture() {
        let m = Move::new(MoveType::RookCapture, pos("a7"), pos("a8"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "Rxa8");
    }

    #[test]
    fn test_move_to_san_rank_disambiguation() {
        let m1 = Move::new(MoveType::QueenQuiet, pos("h4"), pos("e1"));
        let m2 = Move::new(MoveType::QueenQuiet, pos("e4"), pos("e1"));
        let all_moves = vec![m1, m2];

        assert_eq!(move_to_san(m1, &all_moves, false), "Qhe1");
    }

    #[test]
    fn test_move_to_san_file_disambiguation() {
        let m1 = Move::new(MoveType::QueenQuiet, pos("h4"), pos("e1"));
        let m2 = Move::new(MoveType::QueenQuiet, pos("h1"), pos("e1"));
        let all_moves = vec![m1, m2];

        assert_eq!(move_to_san(m1, &all_moves, false), "Q4e1");
    }

    #[test]
    fn test_move_to_san_rank_file_disambiguation() {
        let m1 = Move::new(MoveType::QueenQuiet, pos("h4"), pos("e1"));
        let m2 = Move::new(MoveType::QueenQuiet, pos("e4"), pos("e1"));
        let m3 = Move::new(MoveType::QueenQuiet, pos("h1"), pos("e1"));
        let all_moves = vec![m1, m2, m3];

        assert_eq!(move_to_san(m1, &all_moves, false), "Qh4e1");
    }

    #[test]
    fn test_move_to_san_gives_check() {
        let m = Move::new(MoveType::QueenCapture, pos("h4"), pos("e1"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, true), "Qxe1+");
    }

    #[test]
    fn test_move_to_san_promotion() {
        let m = Move::new(MoveType::QueenQuietPromotion, pos("a7"), pos("a8"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "a8=Q");
    }

    #[test]
    fn test_move_to_san_capture_promotion() {
        let m = Move::new(MoveType::QueenCapturePromotion, pos("a7"), pos("b8"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "axb8=Q");
    }

    #[test]
    fn test_move_to_san_promotion_with_queen_on_same_rank() {
        let m1 = Move::new(MoveType::QueenQuietPromotion, pos("a7"), pos("a8"));
        let m2 = Move::new(MoveType::QueenQuiet, pos("d8"), pos("a8"));
        let all_moves = vec![m1, m2];

        assert_eq!(move_to_san(m1, &all_moves, false), "a8=Q");
    }
    
    #[test]
    fn test_move_to_san_king_side_castling() {
        let m = Move::new(MoveType::KingKSCastling, pos("e1"), pos("g1"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "O-O");
    }
    
    #[test]
    fn test_move_to_san_queen_side_castling() {
        let m = Move::new(MoveType::KingQSCastling, pos("e1"), pos("c1"));
        let all_moves = vec![m];

        assert_eq!(move_to_san(m, &all_moves, false), "O-O-O");
    }

    fn pos(file_rank: &str) -> i8 {
        let file = file_rank.chars().next().unwrap();
        let rank = file_rank.chars().nth(1).unwrap();

        (rank.to_digit(10).unwrap() as i8 - 1) * 8 + (file as i8 - 'a' as i8)
    }
}
