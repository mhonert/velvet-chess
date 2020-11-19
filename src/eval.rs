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
use std::cmp::{max, min};

pub trait Eval {
    fn get_score(&self) -> i32;
}

const PASSED_PAWN_THRESHOLD: u32 = 4;

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

        score += white_king_shield * self.options.get_king_shield_bonus();
        score -= black_king_shield * self.options.get_king_shield_bonus();

        // Castling
        if self.has_white_castled() {
            score += self.options.get_castling_bonus();
        } else {
            if !self.can_castle(Castling::WhiteQueenSide) {
                score -= self.options.get_lost_queenside_castling_penalty()
            }

            if !self.can_castle(Castling::WhiteKingSide) {
                score -= self.options.get_lost_kingside_castling_penalty();

            }
        }

        if self.has_black_castled() {
            score -= self.options.get_castling_bonus();
        } else {
            if !self.can_castle(Castling::BlackQueenSide) {
                score += self.options.get_lost_queenside_castling_penalty()
            }

            if !self.can_castle(Castling::BlackKingSide) {
                score += self.options.get_lost_kingside_castling_penalty();

            }
        }

        let white_pieces = self.get_all_piece_bitboard(WHITE);
        let black_pieces = self.get_all_piece_bitboard(BLACK);

        // Mobility evaluation
        let empty_board = !white_pieces & !black_pieces;
        let empty_or_black = empty_board | black_pieces;

        let black_pawn_attacks = black_left_pawn_attacks(black_pawns) | black_right_pawn_attacks(black_pawns);
        let mut white_safe_targets = empty_or_black & !black_pawn_attacks;

        let mut threat_to_white_king = 0;
        let mut threat_to_black_king = 0;

        let black_king_danger_zone = self.bb.get_king_danger_zone(self.black_king);
        let white_king_danger_zone = self.bb.get_king_danger_zone(self.white_king);

        let mut white_bishops = self.get_bitboard(B);
        let mut black_bishops = self.get_bitboard(-B);
        let mut white_rooks = self.get_bitboard(R);
        let mut black_rooks = self.get_bitboard(-R);
        let white_queens = self.get_bitboard(Q);
        let black_queens = self.get_bitboard(-Q);

        // Knights
        let mut white_knight_attacks: u64 = 0;
        let mut white_knight_threat = 0;
        let white_knights = self.get_bitboard(N);
        {
            let mut knights = white_knights;
            while knights != 0 {
                let pos = knights.trailing_zeros();
                knights ^= 1 << pos as u64; // unset bit

                let possible_moves = self.bb.get_knight_attacks(pos as i32);
                white_knight_attacks |= possible_moves;

                let move_count = (possible_moves & white_safe_targets).count_ones();
                score += self.options.get_knight_mob_bonus(move_count as usize);
                eg_score += self.options.get_eg_knight_mob_bonus(move_count as usize);

                if possible_moves & black_king_danger_zone != 0 {
                    threat_to_black_king += self.options.get_knight_king_threat();
                    white_knight_threat = 1;
                }
            }
        }

        let empty_or_white = empty_board | white_pieces;
        let white_pawn_attacks = white_left_pawn_attacks(white_pawns) | white_right_pawn_attacks(white_pawns);
        let mut black_safe_targets = empty_or_white & !white_pawn_attacks;

        let mut black_knight_attacks: u64 = 0;
        let black_knights = self.get_bitboard(-N);
        let mut black_knight_threat = 0;
        {
            let mut knights = black_knights;
            while knights != 0 {
                let pos = knights.trailing_zeros();
                knights ^= 1 << pos as u64; // unset bit

                let possible_moves = self.bb.get_knight_attacks(pos as i32);
                black_knight_attacks |= possible_moves;

                let move_count = (possible_moves & black_safe_targets).count_ones();
                score -= self.options.get_knight_mob_bonus(move_count as usize);
                eg_score -= self.options.get_eg_knight_mob_bonus(move_count as usize);

                if possible_moves & white_king_danger_zone != 0 {
                    threat_to_white_king += self.options.get_knight_king_threat();
                    black_knight_threat = 1;
                }
            }
        }

        white_safe_targets &= !black_knight_attacks;
        black_safe_targets &= !white_knight_attacks;

        let occupied = !empty_board;

        // Bishops
        let mut white_bishop_attacks: u64 = 0;
        let mut white_bishop_threat = 0;
        {
            // let occupied = white_kt_occupied & !white_queens & !white_bishops;
            while white_bishops != 0 {
                let pos = white_bishops.trailing_zeros();
                white_bishops ^= 1 << pos as u64;  // unset bit

                let possible_moves = self.bb.get_diagonal_attacks(occupied, pos as i32) | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);
                white_bishop_attacks |= possible_moves;
                let move_count = (possible_moves & white_safe_targets).count_ones();
                score += self.options.get_bishop_mob_bonus(move_count as usize);
                eg_score += self.options.get_eg_bishop_mob_bonus(move_count as usize);

                if possible_moves & black_king_danger_zone != 0 {
                    threat_to_black_king += self.options.get_bishop_king_threat();
                    white_bishop_threat = 1;
                }
            }
        }

        let mut black_bishop_attacks: u64 = 0;
        let mut black_bishop_threat = 0;
        {
            // let occupied = black_kt_occupied & !black_queens & !black_bishops;
            while black_bishops != 0 {
                let pos = black_bishops.trailing_zeros();
                black_bishops ^= 1 << pos as u64;  // unset bit

                let possible_moves = self.bb.get_diagonal_attacks(occupied, pos as i32) | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);
                black_bishop_attacks |= possible_moves;
                let move_count = (possible_moves & black_safe_targets).count_ones();
                score -= self.options.get_bishop_mob_bonus(move_count as usize);
                eg_score -= self.options.get_eg_bishop_mob_bonus(move_count as usize);

                if possible_moves & white_king_danger_zone != 0 {
                    threat_to_white_king += self.options.get_bishop_king_threat();
                    black_bishop_threat = 1;
                }
            }
        }

        white_safe_targets &= !black_bishop_attacks;
        black_safe_targets &= !white_bishop_attacks;

        // Rooks
        let mut white_rook_attacks: u64 = 0;
        let mut white_rook_threat = 0;
        {
            // let occupied = white_kt_occupied & !white_queens & !white_rooks;
            while white_rooks != 0 {
                let pos = white_rooks.trailing_zeros();
                white_rooks ^= 1 << pos as u64;  // unset bit

                let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32) | self.bb.get_vertical_attacks(occupied, pos as i32);
                white_rook_attacks |= possible_moves;
                let move_count = (possible_moves & white_safe_targets).count_ones();
                score += self.options.get_rook_mob_bonus(move_count as usize);
                eg_score += self.options.get_eg_rook_mob_bonus(move_count as usize);

                if possible_moves & black_king_danger_zone != 0 {
                    threat_to_black_king += self.options.get_rook_king_threat();
                    white_rook_threat = 1;
                }
            }
        }

        let mut black_rook_attacks: u64 = 0;
        let mut black_rook_threat = 0;
        {
            // let occupied = black_kt_occupied & !black_queens & !black_rooks;
            while black_rooks != 0 {
                let pos = black_rooks.trailing_zeros();
                black_rooks ^= 1 << pos as u64;  // unset bit

                let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32) | self.bb.get_vertical_attacks(occupied, pos as i32);
                black_rook_attacks |= possible_moves;
                let move_count = (possible_moves & black_safe_targets).count_ones();
                score -= self.options.get_rook_mob_bonus(move_count as usize);
                eg_score -= self.options.get_eg_rook_mob_bonus(move_count as usize);

                if possible_moves & white_king_danger_zone != 0 {
                    threat_to_white_king += self.options.get_rook_king_threat();
                    black_rook_threat = 1;
                }
            }
        }

        white_safe_targets &= !black_rook_attacks;
        black_safe_targets &= !white_rook_attacks;

        // Queens
        let mut white_queen_threats = 0;
        {
            let mut queens = white_queens;
            while queens != 0 {
                let pos = queens.trailing_zeros();
                let pos_mask = 1 << pos as u64;
                queens ^= pos_mask;  // unset bit

                let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32)
                    | self.bb.get_vertical_attacks(occupied, pos as i32)
                    | self.bb.get_diagonal_attacks(occupied, pos as i32)
                    | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);

                let move_count = (possible_moves & white_safe_targets).count_ones();
                score += self.options.get_queen_mob_bonus(move_count as usize);
                eg_score += self.options.get_eg_queen_mob_bonus(move_count as usize);

                if possible_moves & black_king_danger_zone != 0 {
                    threat_to_black_king += self.options.get_queen_king_threat();
                    white_queen_threats += 1;

                    if pos_mask & black_king_danger_zone != 0 {
                        white_queen_threats += 1
                    }
                }
            }
        }

        let mut black_queen_threats = 0;
        {
            let mut queens = black_queens;
            while queens != 0 {
                let pos = queens.trailing_zeros();
                let pos_mask = 1 << pos as u64;
                queens ^= pos_mask;  // unset bit

                let possible_moves = self.bb.get_horizontal_attacks(occupied, pos as i32)
                    | self.bb.get_vertical_attacks(occupied, pos as i32)
                    | self.bb.get_diagonal_attacks(occupied, pos as i32)
                    | self.bb.get_anti_diagonal_attacks(occupied, pos as i32);

                let move_count = (possible_moves & black_safe_targets).count_ones();
                score -= self.options.get_queen_mob_bonus(move_count as usize);
                eg_score -= self.options.get_eg_queen_mob_bonus(move_count as usize);

                if possible_moves & white_king_danger_zone != 0 {
                    threat_to_white_king += self.options.get_queen_king_threat();
                    black_queen_threats += 1;

                    if pos_mask & white_king_danger_zone != 0 {
                        black_queen_threats += 1;
                    }
                }
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
                && (self.bb.get_white_pawn_freepath(pos as i32) & pawn_blockers) == 0
                && (self.bb.get_white_pawn_freesides(pos as i32) & black_pawns) == 0 {

                // Not blocked and unguarded by enemy pawns
                score += self.options.get_passed_pawn_bonus((distance_to_promotion - 1) as usize);
                eg_score += self.options.get_eg_passed_pawn_bonus((distance_to_promotion - 1) as usize);

                // Passed pawn - king distance
                let own_king_distance = max((self.white_king / 8 - pos as i32 / 8).abs(),
                                            (self.white_king % 8 - pos as i32 % 8).abs());

                eg_score += self.options.get_passed_pawn_king_defense_bonus(own_king_distance as usize);

                let opponent_king_distance = max((self.black_king / 8 - pos as i32 / 8).abs(),
                                                 (self.black_king % 8 - pos as i32 % 8).abs());
                eg_score -= self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance as usize);
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
                && (self.bb.get_black_pawn_freepath(pos as i32) & pawn_blockers) == 0
                && (self.bb.get_black_pawn_freesides(pos as i32) & white_pawns) == 0 {

                // Not blocked and unguarded by enemy pawns
                score -= self.options.get_passed_pawn_bonus((distance_to_promotion - 1) as usize);
                eg_score -= self.options.get_eg_passed_pawn_bonus((distance_to_promotion - 1) as usize);

                // Passed pawn - king distance
                let own_king_distance = max((self.black_king / 8 - pos as i32 / 8).abs(),
                                            (self.black_king % 8 - pos as i32 % 8).abs());

                eg_score -= self.options.get_passed_pawn_king_defense_bonus(own_king_distance as usize);

                let opponent_king_distance = max((self.white_king / 8 - pos as i32 / 8).abs(),
                                                 (self.white_king % 8 - pos as i32 % 8).abs());
                eg_score += self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance as usize);
            }
        }

        // Interpolate between opening/mid-game score and end game score for a smooth eval score transition
        let pieces_except_king_count: i32 = (white_pieces | black_pieces).count_ones() as i32 - 2; // -2 for two kings

        // Material difference relative to total piece count
        if pieces_except_king_count > 0 {
            score += self.score as i32 / pieces_except_king_count;
        }

        let white_queen_phase_score = if white_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
        let black_queen_phase_score = if black_queens != 0 { QUEEN_PHASE_VALUE } else { 0 };
        let queen_phase_score: i32 = white_queen_phase_score + black_queen_phase_score;
        let pawn_count: i32 = (white_pawns | black_pawns).count_ones() as i32;

        let phase: i32 = pawn_count * PAWN_PHASE_VALUE + pieces_except_king_count * BASE_PIECE_PHASE_VALUE + queen_phase_score;
        let eg_phase: i32 = MAX_PHASE - phase;

        let mut interpolated_score = ((score * phase) + (eg_score * eg_phase)) / MAX_PHASE;

        // Perform evaluations which apply to all game phases

        // Pawn cover bonus
        let white_pawns_and_knights = white_pawns | white_knights;
        interpolated_score += (white_pawns_and_knights & white_pawn_attacks).count_ones() as i32 * self.options.get_pawn_cover_bonus();

        let black_pawns_and_knights = black_pawns | black_knights;
        interpolated_score -= (black_pawns_and_knights & black_pawn_attacks).count_ones() as i32 * self.options.get_pawn_cover_bonus();

        // Doubled pawn penalty
        interpolated_score -= calc_doubled_pawn_penalty(white_pawns, self.options.get_doubled_pawn_penalty());
        interpolated_score += calc_doubled_pawn_penalty(black_pawns, self.options.get_doubled_pawn_penalty());

        // King threat (uses king threat values from mobility evaluation)

        if threat_to_black_king > 1 {
            let white_pawn_threats = min(3, (white_pawn_attacks & black_king_danger_zone).count_ones());
            let threat_pattern = white_pawn_threats |
                white_knight_threat << 2 |
                white_bishop_threat << 3 |
                white_rook_threat << 4 |
                min(3, white_queen_threats) << 5;

            interpolated_score += self.options.get_king_danger_piece_penalty(threat_to_black_king as usize);
            interpolated_score += self.options.get_king_threat_adjustment(threat_pattern as usize);
        }

        if threat_to_white_king > 1 {
            let black_pawn_threats = min(3, (black_pawn_attacks & white_king_danger_zone).count_ones());
            let threat_pattern = black_pawn_threats |
                black_knight_threat << 2 |
                black_bishop_threat << 3 |
                black_rook_threat << 4 |
                min(3, black_queen_threats) << 5;

            interpolated_score -= self.options.get_king_danger_piece_penalty(threat_to_white_king as usize);
            interpolated_score -= self.options.get_king_threat_adjustment(threat_pattern as usize);
        }

        interpolated_score
    }
}

#[inline]
fn calc_doubled_pawn_penalty(pawns: u64, penalty: i32) -> i32 {
    let doubled = (pawns & pawns.rotate_right(8))
        | pawns & pawns.rotate_right(16)
        | pawns & pawns.rotate_right(24)
        | pawns & pawns.rotate_right(32);

    doubled.count_ones() as i32 * penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::create_from_fen;

    #[test]
    fn check_correct_eval_score_for_mirrored_pos() {
        assert_eq!(create_from_fen("1b1r2k1/r4pp1/2p2n1p/1pPp3P/1P1PpPqQ/4P1P1/1B1N4/1K2R2R w - - 0 38").get_score(),
                   -create_from_fen("1k2r2r/1b1n4/4p1p1/1p1pPpQq/1PpP3p/2P2N1P/R4PP1/1B1R2K1 b - - 0 38").get_score());

        assert_eq!(create_from_fen("8/8/8/5k2/4r1p1/6P1/3K1P2/8 b - - 0 80").get_score(),
                   -create_from_fen("8/3k1p2/6p1/4R1P1/5K2/8/8/8 w - - 0 80").get_score());

        assert_eq!(create_from_fen("2kr1b1r/pp1nnp1b/4p2p/2qpP1p1/3N1B1N/8/PPP1BPPP/2RQ1RK1 w - - 0 14").get_score(),
                   -create_from_fen("2rq1rk1/ppp1bppp/8/3n1b1n/2QPp1P1/4P2P/PP1NNP1B/2KR1B1R b - - 0 14").get_score());
    }
}
