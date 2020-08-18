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
use crate::pieces::{P, N, B, R, Q};
use crate::castling::Castling;
use crate::colors::{WHITE, BLACK};
use crate::bitboard::{black_left_pawn_attacks, black_right_pawn_attacks, white_left_pawn_attacks, white_right_pawn_attacks};
use std::cmp::{max};

pub trait Eval {
    fn get_score(&self) -> i32;
}

// Evaluation constants
const DOUBLED_PAWN_PENALTY: i32 = 17;

const PASSED_PAWN_THRESHOLD: u32 = 4;

const KING_SHIELD_BONUS: i32 = 20;

const CASTLING_BONUS: i32 = 28;
const LOST_QUEENSIDE_CASTLING_PENALTY: i32 = 24;
const LOST_KINGSIDE_CASTLING_PENALTY: i32 = 39;

const KING_DANGER_PIECE_PENALTY: [i32; 16] = [ 0, -3, -3, 8, 25, 52, 95, 168, 258, 320, 1200, 1200, 1200, 1200, 1200, 1200 ];

const PAWN_COVER_BONUS: i32 = 12;

const KNIGHT_MOB_BONUS: [i32; 9] = [ -9, 3, 10, 11, 20, 22, 29, 29, 55 ];
const BISHOP_MOB_BONUS: [i32; 14] = [ -2, 6, 14, 19, 22, 24, 29, 30, 35, 32, 49, 105, 73, 56 ];
const ROOK_MOB_BONUS: [i32; 15] = [ -13, -10, -6, -1, 2, 9, 14, 23, 29, 36, 59, 64, 52, 62, 57 ];
const QUEEN_MOB_BONUS: [i32; 28] = [ -2, -6, 0, 3, 11, 13, 15, 17, 20, 28, 28, 35, 51, 42, 50, 62, 99, 105, 102, 159, 100, 122, 131, 131, 115, 64, 75, 61 ];

const EG_KNIGHT_MOB_BONUS: [i32; 9] = [ -65, -27, -2, 9, 13, 23, 20, 25, 4 ];
const EG_BISHOP_MOB_BONUS: [i32; 14] = [ -46, -16, 1, 9, 16, 24, 27, 25, 30, 29, 20, 11, 35, 22 ];
const EG_ROOK_MOB_BONUS: [i32; 15] = [ -72, -31, -8, 2, 15, 25, 28, 31, 35, 37, 36, 41, 50, 48, 45 ];
const EG_QUEEN_MOB_BONUS: [i32; 28] = [ -77, -7, -18, 11, 4, 35, 42, 61, 70, 66, 85, 87, 85, 100, 108, 109, 98, 86, 109, 95, 123, 121, 118, 129, 127, 128, 159, 123 ];

const BASE_PIECE_PHASE_VALUE: i32 = 2;
const PAWN_PHASE_VALUE: i32 = -1; // relative to the base piece value
const QUEEN_PHASE_VALUE: i32 = 4; // relative to the base piece value

const MAX_PHASE: i32 = 16 * PAWN_PHASE_VALUE + 30 * BASE_PIECE_PHASE_VALUE + 2 * QUEEN_PHASE_VALUE;


impl Eval for Board {
    fn get_score(&self) -> i32 {
        let mut score = self.score as i32;
        let mut eg_score = self.eg_score as i32;

        let white_pawns = self.get_bitboard(P);
        let black_pawns = self.get_bitboard(-P);

        // Add bonus for pawns which form a shield in front of the king
        let white_king_shield = (white_pawns & self.bb.get_white_king_shield(self.white_king)).count_ones() as i32;
        let black_king_shield = (black_pawns & self.bb.get_black_king_shield(self.black_king)).count_ones() as i32;

        score += white_king_shield * KING_SHIELD_BONUS;
        score -= black_king_shield * KING_SHIELD_BONUS;

        // Castling
        if self.has_white_castled() {
            score += CASTLING_BONUS
        } else {
            if !self.can_castle(Castling::WhiteQueenSide) {
                score -= LOST_QUEENSIDE_CASTLING_PENALTY;
            }

            if !self.can_castle(Castling::WhiteKingSide) {
                score -= LOST_KINGSIDE_CASTLING_PENALTY;

            }
        }

        if self.has_black_castled() {
            score -= CASTLING_BONUS
        } else {
            if !self.can_castle(Castling::BlackQueenSide) {
                score += LOST_QUEENSIDE_CASTLING_PENALTY;
            }

            if !self.can_castle(Castling::BlackKingSide) {
                score += LOST_KINGSIDE_CASTLING_PENALTY;

            }
        }

        let white_pieces = self.get_all_piece_bitboard(WHITE);
        let black_pieces = self.get_all_piece_bitboard(BLACK);

        // Mobility evaluation
        let empty_board = !white_pieces & !black_pieces;
        let empty_or_black = empty_board | black_pieces;

        let black_pawn_attacks = black_left_pawn_attacks(black_pawns) | black_right_pawn_attacks(black_pawns);
        let mut white_safe_targets = empty_or_black & !black_pawn_attacks;

        let mut white_king_threat = 0;
        let mut black_king_threat = 0;

        let black_king_danger_zone = self.bb.get_king_danger_zone(self.black_king);
        let white_king_danger_zone = self.bb.get_king_danger_zone(self.white_king);

        // Knights
        let mut white_knight_attacks: u64 = 0;
        let white_knights = self.get_bitboard(N);
        {
            let mut knights = white_knights;
            while knights != 0 {
                let pos = knights.trailing_zeros();
                knights ^= 1 << pos as u64; // unset bit

                let possible_moves = self.bb.get_knight_attacks(pos as i32);
                white_knight_attacks |= possible_moves;

                let move_count = (possible_moves & white_safe_targets).count_ones();
                score += KNIGHT_MOB_BONUS[move_count as usize];
                eg_score += EG_KNIGHT_MOB_BONUS[move_count as usize];

                if possible_moves & black_king_danger_zone != 0 {
                    black_king_threat += 1
                }
            }
        }

        let empty_or_white = empty_board | white_pieces;
        let white_pawn_attacks = white_left_pawn_attacks(white_pawns) | white_right_pawn_attacks(white_pawns);
        let mut black_safe_targets = empty_or_white & !white_pawn_attacks;

        let mut black_knight_attacks: u64 = 0;
        let black_knights = self.get_bitboard(-N);
        {
            let mut knights = black_knights;
            while knights != 0 {
                let pos = knights.trailing_zeros();
                knights ^= 1 << pos as u64; // unset bit

                let possible_moves = self.bb.get_knight_attacks(pos as i32);
                black_knight_attacks |= possible_moves;

                let move_count = (possible_moves & black_safe_targets).count_ones();
                score -= KNIGHT_MOB_BONUS[move_count as usize];
                eg_score -= EG_KNIGHT_MOB_BONUS[move_count as usize];

                if possible_moves & white_king_danger_zone != 0 {
                    white_king_threat += 1
                }
            }
        }

        white_safe_targets &= !black_knight_attacks;
        black_safe_targets &= !white_knight_attacks;

        // Bishops
        let occupied = !empty_board;
        let mut white_bishops = self.get_bitboard(B);
        let mut white_bishop_attacks: u64 = 0;
        while white_bishops != 0 {
            let pos = white_bishops.trailing_zeros();
            white_bishops ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_diagonal_attacks(occupied, pos as i32) | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);
            white_bishop_attacks |= possible_moves;
            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += BISHOP_MOB_BONUS[move_count as usize];
            eg_score += EG_BISHOP_MOB_BONUS[move_count as usize];

            if possible_moves & black_king_danger_zone != 0 {
                black_king_threat += 1;
            }
        }

        let mut black_bishops = self.get_bitboard(-B);
        let mut black_bishop_attacks: u64 = 0;
        while black_bishops != 0 {
            let pos = black_bishops.trailing_zeros();
            black_bishops ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_diagonal_attacks(occupied, pos as i32) | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);
            black_bishop_attacks |= possible_moves;
            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= BISHOP_MOB_BONUS[move_count as usize];
            eg_score -= EG_BISHOP_MOB_BONUS[move_count as usize];

            if possible_moves & white_king_danger_zone != 0 {
                white_king_threat += 1;
            }
        }

        white_safe_targets &= !black_bishop_attacks;
        black_safe_targets &= !white_bishop_attacks;

        // Rooks
        let mut white_rooks = self.get_bitboard(R);
        let mut white_rook_attacks: u64 = 0;
        while white_rooks != 0 {
            let pos = white_rooks.trailing_zeros();
            white_rooks ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32) | self.bb.get_vertical_attacks(occupied, pos as i32);
            white_rook_attacks |= possible_moves;
            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += ROOK_MOB_BONUS[move_count as usize];
            eg_score += EG_ROOK_MOB_BONUS[move_count as usize];

            if possible_moves & black_king_danger_zone != 0 {
                black_king_threat += 1;
            }
        }

        let mut black_rooks = self.get_bitboard(-R);
        let mut black_rook_attacks: u64 = 0;
        while black_rooks != 0 {
            let pos = black_rooks.trailing_zeros();
            black_rooks ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32) | self.bb.get_vertical_attacks(occupied, pos as i32);
            black_rook_attacks |= possible_moves;
            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= ROOK_MOB_BONUS[move_count as usize];
            eg_score -= EG_ROOK_MOB_BONUS[move_count as usize];

            if possible_moves & white_king_danger_zone != 0 {
                white_king_threat += 1;
            }
        }

        white_safe_targets &= !black_rook_attacks;
        black_safe_targets &= !white_rook_attacks;

        // Queens
        let mut white_queens = self.get_bitboard(Q);
        while white_queens != 0 {
            let pos = white_queens.trailing_zeros();
            white_queens ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32)
                | self.bb.get_vertical_attacks(occupied, pos as i32)
                | self.bb.get_diagonal_attacks(occupied, pos as i32)
                | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += QUEEN_MOB_BONUS[move_count as usize];
            eg_score += EG_QUEEN_MOB_BONUS[move_count as usize];

            if possible_moves & black_king_danger_zone != 0 {
                black_king_threat += 1;
            }
        }

        let mut black_queens = self.get_bitboard(-Q);
        while black_queens != 0 {
            let pos = black_queens.trailing_zeros();
            black_queens ^= 1 << pos as u64;  // unset bit

            let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32)
                | self.bb.get_vertical_attacks(occupied, pos as i32)
                | self.bb.get_diagonal_attacks(occupied, pos as i32)
                | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= QUEEN_MOB_BONUS[move_count as usize];
            eg_score -= EG_QUEEN_MOB_BONUS[move_count as usize];

            if possible_moves & white_king_danger_zone != 0 {
                white_king_threat += 1;
            }
        }

        // Passed white pawn bonus
        let pawn_blockers = black_pieces | white_pawns;
        let mut pawns = white_pawns;
        while pawns != 0 {
            let pos = pawns.trailing_zeros();
            pawns ^= 1 << pos as u64; // unset bit

            let distance_to_promotion = pos / 8;
            if distance_to_promotion <= PASSED_PAWN_THRESHOLD
                && (self.bb.get_white_pawn_freepath(pos as i32) & pawn_blockers) == 0 { // unblocked

                let col = pos & 7;
                if (col == 0 || (self.bb.get_white_pawn_freepath(pos as i32 - 1) & black_pawns == 0))
                    && (col == 7 || (self.bb.get_white_pawn_freepath(pos as i32 + 1) & black_pawns == 0)) {
                    // Unguarded by enemy pawns
                    let bonus = self.options.get_passed_pawn_bonus(distance_to_promotion - 1);
                    score += bonus;
                    eg_score += bonus;

                    let own_king_distance = max((self.white_king / 8 - pos as i32 / 8).abs(),
                                                     (self.white_king % 8 - pos as i32 % 8).abs());

                    eg_score += self.options.get_passed_pawn_king_defense_bonus(own_king_distance);

                    let opponent_king_distance = max((self.black_king / 8 - pos as i32 / 8).abs(),
                                                     (self.black_king % 8 - pos as i32 % 8).abs());
                    eg_score -= self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance);
                }
            }
        }

        // Passed black pawn bonus
        let pawn_blockers = white_pieces | black_pawns;
        pawns = black_pawns;
        while pawns != 0 {
            let pos = pawns.trailing_zeros();
            pawns ^= 1 << pos as u64; // unset bit
            let distance_to_promotion = 7 - pos / 8;
            if distance_to_promotion <= PASSED_PAWN_THRESHOLD
                && (self.bb.get_black_pawn_freepath(pos as i32) & pawn_blockers) == 0 {
                let col = pos & 7;
                if (col == 0 || (self.bb.get_black_pawn_freepath(pos as i32 - 1) & white_pawns == 0))
                    && (col == 7 || (self.bb.get_black_pawn_freepath(pos as i32 + 1) & white_pawns == 0)) {
                    // Unguarded by enemy pawns
                    let bonus = self.options.get_passed_pawn_bonus(distance_to_promotion - 1);
                    score -= bonus;
                    eg_score -= bonus;

                    let own_king_distance = max((self.black_king / 8 - pos as i32 / 8).abs(),
                                                (self.black_king % 8 - pos as i32 % 8).abs());

                    eg_score -= self.options.get_passed_pawn_king_defense_bonus(own_king_distance);

                    let opponent_king_distance = max((self.white_king / 8 - pos as i32 / 8).abs(),
                                                     (self.white_king % 8 - pos as i32 % 8).abs());
                    eg_score += self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance);
                }
            }
        }

        // Interpolate between opening/mid-game score and end game score for a smooth eval score transition
        let pawn_count: i32 = (white_pawns | black_pawns).count_ones() as i32;
        let pieces_except_king_count: i32 = (white_pieces | black_pieces).count_ones() as i32 - 2; // -2 for two kings

        let white_queen_phase_score = if white_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
        let black_queen_phase_score = if black_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
        let queen_phase_score: i32 = white_queen_phase_score + black_queen_phase_score;

        let phase: i32 = pawn_count * PAWN_PHASE_VALUE + pieces_except_king_count * BASE_PIECE_PHASE_VALUE + queen_phase_score;
        let eg_phase: i32 = MAX_PHASE - phase;

        let mut interpolated_score = ((score * phase) + (eg_score * eg_phase)) / MAX_PHASE;

        // Perform evaluations which apply to all game phases

        // Pawn cover bonus
        let white_pawns_and_knights = white_pawns | white_knights;
        interpolated_score += (white_pawns_and_knights & white_pawn_attacks).count_ones() as i32 * PAWN_COVER_BONUS;

        let black_pawns_and_knights = black_pawns | black_knights;
        interpolated_score -= (black_pawns_and_knights & black_pawn_attacks).count_ones() as i32 * PAWN_COVER_BONUS;

        // Doubled pawn penalty
        interpolated_score -= calc_doubled_pawn_penalty(white_pawns);
        interpolated_score += calc_doubled_pawn_penalty(black_pawns);

        // King threat (uses king threat values from mobility evaluation)
        black_king_threat += (white_pawn_attacks & black_king_danger_zone).count_ones() / 2;
        white_king_threat += (black_pawn_attacks & white_king_danger_zone).count_ones() / 2;

        if white_queens & black_king_danger_zone != 0 {
            black_king_threat += 3;
        }
        interpolated_score += KING_DANGER_PIECE_PENALTY[black_king_threat as usize];

        if black_queens & white_king_danger_zone != 0 {
            white_king_threat += 3;
        }
        interpolated_score -= KING_DANGER_PIECE_PENALTY[white_king_threat as usize];

        interpolated_score
    }
}

fn calc_doubled_pawn_penalty(pawns: u64) -> i32 {
    let doubled = (pawns & pawns.rotate_right(8))
        | pawns & pawns.rotate_right(16)
        | pawns & pawns.rotate_right(24)
        | pawns & pawns.rotate_right(32);

    doubled.count_ones() as i32 * DOUBLED_PAWN_PENALTY
}