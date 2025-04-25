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
use crate::slices::SliceElementAccess;

const PLAYER: u64 = 0x8000000000000001;
const EP: u64 = 0x42a6344d1227098d;
const CASTLING: u64 = 0xab28bc31b46cbb3c;
static PIECE: [u64; 13] = [ 0x7eb5140a57a894c8, 0x467813d5c298de63, 0xc5c1f1e2594b941c, 0xf319da8df6cf96b4, 0xdc8b55eebfca3a40, 0x5418f15d4c08f4e2, 0x1d3350493f26ec1e, 0xd0c4b14bdb230807, 0x73ef23b69de88e14, 0xb9219d4683de93d9, 0xe8c0a3740dbb1c7a, 0x59fd9c7dc2c9298a, 0x1ffc53c9670efd27 ];

#[inline(always)]
pub fn player_zobrist_key() -> u64 {
    PLAYER
}

#[inline(always)]
pub fn enpassant_zobrist_key(en_passant_state: u8) -> u64 {
    EP.rotate_left(en_passant_state as u32)
}

#[inline(always)]
pub fn castling_zobrist_key(castling_state: u8) -> u64 {
    CASTLING.rotate_left(castling_state as u32)
}

#[inline(always)]
pub fn piece_zobrist_key(piece: i8, pos: usize) -> u64 {
    let piece_key = *PIECE.el((piece + 6) as usize);
    piece_key.rotate_left(pos as u32)
}

#[cfg(test)]
mod tests {
    use crate::board::castling::{Castling, CastlingState};
    use crate::board::{BlackBoardPos, WhiteBoardPos};
    use crate::zobrist::{enpassant_zobrist_key, player_zobrist_key};

    #[test]
    fn check_key_quality() {
        let mut all_keys = Vec::new();
        all_keys.push(player_zobrist_key());

        let mut castlings = Vec::new();
        for side1 in [Some(Castling::WhiteKingSide), Some(Castling::WhiteQueenSide), Some(Castling::BlackKingSide), Some(Castling::BlackQueenSide), None].iter() {
            for side2 in [Some(Castling::WhiteKingSide), Some(Castling::WhiteQueenSide), Some(Castling::BlackKingSide), Some(Castling::BlackQueenSide), None].iter() {
                for side3 in [Some(Castling::WhiteKingSide), Some(Castling::WhiteQueenSide), Some(Castling::BlackKingSide), Some(Castling::BlackQueenSide), None].iter() {
                    for side4 in [Some(Castling::WhiteKingSide), Some(Castling::WhiteQueenSide), Some(Castling::BlackKingSide), Some(Castling::BlackQueenSide), None].iter() {
                        let mut cs = CastlingState::ALL;
                        side1.map(|side| cs.clear_side(side));
                        side2.map(|side| cs.clear_side(side));
                        side3.map(|side| cs.clear_side(side));
                        side4.map(|side| cs.clear_side(side));
                        castlings.push(cs);
                    }
                }
            }
        }
        castlings.sort();
        castlings.dedup();
        castlings.iter().for_each(|c| all_keys.push(c.zobrist_key()));

        for ep in (WhiteBoardPos::EnPassantLineStart as u8)..=(WhiteBoardPos::EnPassantLineEnd as u8) {
            all_keys.push(enpassant_zobrist_key(ep));
        }
        for ep in (BlackBoardPos::EnPassantLineStart as u8)..=(BlackBoardPos::EnPassantLineEnd as u8) {
            all_keys.push(enpassant_zobrist_key(ep));
        }

        for piece in -6..=6 {
            if piece == 0 {
                continue;
            }
            for pos in 0..64 {
                all_keys.push(super::piece_zobrist_key(piece, pos));
            }
        }

        let mut duplicates = all_keys.len();
        all_keys.sort_unstable();
        all_keys.dedup();
        duplicates -= all_keys.len();

        assert_eq!(0, duplicates);
        assert_eq!(0, all_keys.iter().filter(|&k| *k == 0 || *k == u64::MAX).count());
    }
}