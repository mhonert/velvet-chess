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

use crate::board::{Board, interpolate_score};
use crate::pieces::{P, N, B, R, Q};
use crate::colors::{WHITE, BLACK};
use crate::bitboard::{black_left_pawn_attacks, black_right_pawn_attacks, white_left_pawn_attacks, white_right_pawn_attacks, BitBoard, get_white_king_shield, get_black_king_shield, get_king_danger_zone, get_knight_attacks, get_white_pawn_freepath, get_white_pawn_freesides, get_black_pawn_freepath, get_black_pawn_freesides, get_column_mask};
use std::cmp::{max};
use crate::magics::{get_bishop_attacks, get_rook_attacks, get_queen_attacks};

pub trait Eval {
    fn get_score(&self) -> i32;
    fn eval_passed_pawns(&self, white_pieces: u64, black_pieces: u64, white_pawns: u64, black_pawns: u64) -> (i32, i32);
}

// Bit-Patterns for piece combos to calculate an index into the king safety table
const PAWN_KING_THREAT: usize       = 0b0000001;
const KNIGHT_KING_THREAT1: usize    = 0b0000010;
const KNIGHT_KING_THREAT2: usize    = 0b0010010;
const BISHOP_KING_THREAT1: usize    = 0b0000100;
const BISHOP_KING_THREAT2: usize    = 0b0010100;
const ROOK_KING_THREAT1: usize      = 0b0001000;
const ROOK_KING_THREAT2: usize      = 0b0011000;
const QUEEN_KING_THREAT: usize      = 0b0100000;
const HIGH_QUEEN_KING_THREAT: usize = 0b1100000;

impl Eval for Board {
    fn get_score(&self) -> i32 {
        let phase = self.calc_phase_value();

        let mut score = self.state.score as i32;
        let mut eg_score = self.state.eg_score as i32;

        let white_pawns = self.get_bitboard(P);
        let black_pawns = self.get_bitboard(-P);

        if white_pawns == 0 && black_pawns == 0 {
            return interpolate_score(phase, score, eg_score);
        }

        let white_queens = self.get_bitboard(Q);
        let black_queens = self.get_bitboard(-Q);

        let white_king = self.king_pos(WHITE);
        let black_king = self.king_pos(BLACK);

        let white_rooks = self.get_bitboard(R);
        let black_rooks = self.get_bitboard(-R);

        // Mobility evaluation
        let white_pieces = self.get_all_piece_bitboard(WHITE);
        let black_pieces = self.get_all_piece_bitboard(BLACK);

        let occupied = white_pieces | black_pieces;
        let empty_board = !occupied;
        let empty_or_black = empty_board | black_pieces;

        let white_pawn_attacks = white_left_pawn_attacks(white_pawns) | white_right_pawn_attacks(white_pawns);
        let black_pawn_attacks = black_left_pawn_attacks(black_pawns) | black_right_pawn_attacks(black_pawns);

        let black_king_danger_zone = get_king_danger_zone(black_king);
        let white_king_danger_zone = get_king_danger_zone(white_king);

        let white_safe_targets = empty_or_black & !black_pawn_attacks;

        let mut white_king_threat_combo = if (white_pawns & black_king_danger_zone) != 0 { PAWN_KING_THREAT } else { 0 };
        let mut black_king_threat_combo = if (black_pawns & white_king_danger_zone) != 0 { PAWN_KING_THREAT } else { 0 };

        // Knights
        let white_knights = self.get_bitboard(N);
        for pos in BitBoard(white_knights) {
            let possible_moves = get_knight_attacks(pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_knight_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_knight_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                if white_king_threat_combo & KNIGHT_KING_THREAT1 != 0 {
                    white_king_threat_combo |= KNIGHT_KING_THREAT2;
                } else {
                    white_king_threat_combo |= KNIGHT_KING_THREAT1;
                }
            }
        }

        let empty_or_white = empty_board | white_pieces;
        let black_safe_targets = empty_or_white & !white_pawn_attacks;

        let black_knights = self.get_bitboard(-N);
        for pos in BitBoard(black_knights) {
            let possible_moves = get_knight_attacks(pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_knight_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_knight_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                if black_king_threat_combo & KNIGHT_KING_THREAT1 != 0 {
                    black_king_threat_combo |= KNIGHT_KING_THREAT2;
                } else {
                    black_king_threat_combo |= KNIGHT_KING_THREAT1;
                }
            }

        }

        let mut pinnable_black_pieces = black_knights | black_rooks;

        // Bishops
        let white_bishops = self.get_bitboard(B);
        for pos in BitBoard(white_bishops) {
            let possible_moves = get_bishop_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_bishop_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_bishop_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                if white_king_threat_combo & BISHOP_KING_THREAT1 != 0 {
                    white_king_threat_combo |= BISHOP_KING_THREAT2;
                } else {
                    white_king_threat_combo |= BISHOP_KING_THREAT1;
                }
            }

            if possible_moves & pinnable_black_pieces != 0 {
                score += self.options.get_bishop_pin_bonus();
                eg_score += self.options.get_eg_bishop_pin_bonus();
            }

        }

        let mut pinnable_white_pieces = white_knights | white_rooks;
        let black_bishops = self.get_bitboard(-B);
        for pos in BitBoard(black_bishops) {
            let possible_moves = get_bishop_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_bishop_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_bishop_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                if black_king_threat_combo & BISHOP_KING_THREAT1 != 0 {
                    black_king_threat_combo |= BISHOP_KING_THREAT2;
                } else {
                    black_king_threat_combo |= BISHOP_KING_THREAT1;
                }
            }

            if possible_moves & pinnable_white_pieces != 0 {
                score -= self.options.get_bishop_pin_bonus();
                eg_score -= self.options.get_eg_bishop_pin_bonus();
            }
        }

        // Rooks
        pinnable_black_pieces = black_knights | black_bishops;
        for pos in BitBoard(white_rooks) {
            // Rook on (half) open file?
            let rook_column_mask = get_column_mask(pos as i32);
            if rook_column_mask & white_pawns == 0 {
                score += self.options.get_rook_on_half_open_file_bonus();
                eg_score += self.options.get_eg_rook_on_half_open_file_bonus();

                if rook_column_mask & black_pawns == 0 {
                    score += self.options.get_rook_on_open_file_bonus();
                    eg_score += self.options.get_eg_rook_on_open_file_bonus();
                }
            }

            let possible_moves = get_rook_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_rook_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_rook_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                if white_king_threat_combo & ROOK_KING_THREAT1 != 0 {
                    white_king_threat_combo |= ROOK_KING_THREAT2;
                } else {
                    white_king_threat_combo |= ROOK_KING_THREAT1;
                }
            }

            if possible_moves & pinnable_black_pieces != 0 {
                score += self.options.get_rook_pin_bonus();
                eg_score += self.options.get_eg_rook_pin_bonus();
            }
        }

        pinnable_white_pieces = white_knights | white_bishops;
        for pos in BitBoard(black_rooks) {
            // Rook on (half) open file?
            let rook_column_mask = get_column_mask(pos as i32);
            if rook_column_mask & black_pawns == 0 {
                score -= self.options.get_rook_on_half_open_file_bonus();
                eg_score -= self.options.get_eg_rook_on_half_open_file_bonus();

                if rook_column_mask & white_pawns == 0 {
                    score -= self.options.get_rook_on_open_file_bonus();
                    eg_score -= self.options.get_eg_rook_on_open_file_bonus();
                }
            }

            let possible_moves = get_rook_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_rook_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_rook_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                if black_king_threat_combo & ROOK_KING_THREAT1 != 0 {
                    black_king_threat_combo |= ROOK_KING_THREAT2;
                } else {
                    black_king_threat_combo |= ROOK_KING_THREAT1;
                }
            }

            if possible_moves & pinnable_white_pieces != 0 {
                score -= self.options.get_rook_pin_bonus();
                eg_score -= self.options.get_eg_rook_pin_bonus();
            }
        }

        // Queens
        for pos in BitBoard(white_queens) {
            let possible_moves = get_queen_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_queen_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_queen_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                white_king_threat_combo |= QUEEN_KING_THREAT;
            }
        }

        if white_queens & black_king_danger_zone != 0 {
            white_king_threat_combo |= HIGH_QUEEN_KING_THREAT;
        }

        for pos in BitBoard(black_queens) {
            let possible_moves = get_queen_attacks(empty_board, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_queen_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_queen_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                black_king_threat_combo |= QUEEN_KING_THREAT;
            }
        }

        if black_queens & white_king_danger_zone != 0 {
            black_king_threat_combo |= HIGH_QUEEN_KING_THREAT;
        }

        // Passed pawn evaluation
        let (pp_score, eg_pp_score) = self.eval_passed_pawns(white_pieces, black_pieces, white_pawns, black_pawns);
        score += pp_score;
        eg_score += eg_pp_score;

        // Piece imbalances
        if white_queens == 0 && black_queens != 0 {
            score -= self.options.get_queen_imbalance_penalty();
            eg_score -= self.options.get_eg_queen_imbalance_penalty();
        } else if black_queens == 0 && white_queens != 0 {
            score += self.options.get_queen_imbalance_penalty();
            eg_score += self.options.get_eg_queen_imbalance_penalty();
        }

        // Interpolate between opening/mid-game score and end game score for a smooth eval score transition
        let mut interpolated_score = interpolate_score(phase, score, eg_score);

        // Perform evaluations which apply to all game phases

        // White king shield pawn structure
        if black_queens != 0 {
            let white_king_shield = (white_pawns & get_white_king_shield(white_king)).count_ones() as i32;
            interpolated_score += white_king_shield * self.options.get_king_shield_bonus();

            if phase >= self.options.get_king_pawn_phase_threshold() {
                let king_row = white_king / 8;
                if king_row >= 5 {
                    let hash = calc_white_pawn_hash(white_pawns, white_king);
                    let pattern_bonus = self.options.get_king_pawn_pattern_bonus(hash as usize);
                    interpolated_score += pattern_bonus;
                }
            }
        }

        // Black king shield pawn structure
        if white_queens != 0 {
            let black_king_shield = (black_pawns & get_black_king_shield(black_king)).count_ones() as i32;
            interpolated_score -= black_king_shield * self.options.get_king_shield_bonus();

            if phase >= self.options.get_king_pawn_phase_threshold() {
                let king_row = black_king / 8;
                if king_row <= 2 {
                    let hash = calc_black_pawn_hash(black_pawns, black_king);
                    let pattern_bonus = self.options.get_king_pawn_pattern_bonus(hash as usize);
                    interpolated_score -= pattern_bonus
                }
            }
        }

        // Pawn cover bonus
        let white_pawns_and_knights = white_pawns | white_knights;
        let white_pawn_attacks = white_left_pawn_attacks(white_pawns) | white_right_pawn_attacks(white_pawns);
        interpolated_score += (white_pawns_and_knights & white_pawn_attacks).count_ones() as i32 * self.options.get_pawn_cover_bonus();

        let black_pawns_and_knights = black_pawns | black_knights;
        let black_pawn_attacks = black_left_pawn_attacks(black_pawns) | black_right_pawn_attacks(black_pawns);
        interpolated_score -= (black_pawns_and_knights & black_pawn_attacks).count_ones() as i32 * self.options.get_pawn_cover_bonus();

        // Doubled pawn penalty
        interpolated_score -= calc_doubled_pawn_penalty(white_pawns, self.options.get_doubled_pawn_penalty());
        interpolated_score += calc_doubled_pawn_penalty(black_pawns, self.options.get_doubled_pawn_penalty());

        // King threat (uses king threat values from mobility evaluation)
        interpolated_score += self.options.get_king_threat_by_piece_combo(white_king_threat_combo);
        interpolated_score -= self.options.get_king_threat_by_piece_combo(black_king_threat_combo);


        interpolated_score
    }

    #[inline]
    fn eval_passed_pawns(&self, white_pieces: u64, black_pieces: u64, white_pawns: u64, black_pawns: u64) -> (i32, i32) {
        let mut score: i32 = 0;
        let mut eg_score: i32 = 0;

        let white_king = self.king_pos(WHITE);
        let black_king = self.king_pos(BLACK);


        // Passed white pawn bonus
        let pawn_blockers = black_pieces | white_pawns;
        for pos in BitBoard(white_pawns & 0x000000FFFFFFFF00) { // skip pawns close to own board half
            if (get_white_pawn_freepath(pos as i32) & pawn_blockers) == 0 {
                let distance_to_promotion = pos / 8;

                score += self.options.get_half_open_file_bonus((distance_to_promotion - 1) as usize);
                eg_score += self.options.get_eg_half_open_file_bonus((distance_to_promotion - 1) as usize);

                // Pawn - king distance
                let own_king_distance = max((white_king / 8 - pos as i32 / 8).abs(), ((white_king & 7) - (pos as i32 & 7)).abs());
                eg_score += self.options.get_pawn_king_defense_bonus(own_king_distance as usize);

                let opponent_king_distance = max((black_king / 8 - pos as i32 / 8).abs(), ((black_king & 7) - (pos as i32 & 7)).abs());
                eg_score -= self.options.get_pawn_king_attacked_penalty(opponent_king_distance as usize);

                if (get_white_pawn_freesides(pos as i32) & black_pawns) == 0 {
                    // Not blocked and unguarded by enemy pawns
                    eg_score += self.options.get_passed_pawn_bonus((distance_to_promotion - 1) as usize);
                    eg_score += self.options.get_passed_pawn_king_defense_bonus(own_king_distance as usize);
                    eg_score -= self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance as usize);
                }
            }
        }

        // Passed black pawn bonus
        let pawn_blockers = white_pieces | black_pawns;
        for pos in BitBoard(black_pawns & 0x00FFFFFFFF000000) { // skip pawns close to own board half
            if (get_black_pawn_freepath(pos as i32) & pawn_blockers) == 0 {
                let distance_to_promotion = 7 - pos / 8;

                score -= self.options.get_half_open_file_bonus((distance_to_promotion - 1) as usize);
                eg_score -= self.options.get_eg_half_open_file_bonus((distance_to_promotion - 1) as usize);

                // Pawn - king distance
                let own_king_distance = max((black_king / 8 - pos as i32 / 8).abs(), ((black_king & 7) - (pos as i32 & 7)).abs());
                eg_score -= self.options.get_pawn_king_defense_bonus(own_king_distance as usize);

                let opponent_king_distance = max((white_king / 8 - pos as i32 / 8).abs(), ((white_king & 7) - (pos as i32 & 7)).abs());
                eg_score += self.options.get_pawn_king_attacked_penalty(opponent_king_distance as usize);

                if (get_black_pawn_freesides(pos as i32) & white_pawns) == 0 {
                    // Not blocked and unguarded by enemy pawns
                    eg_score -= self.options.get_passed_pawn_bonus((distance_to_promotion - 1) as usize);
                    eg_score -= self.options.get_passed_pawn_king_defense_bonus(own_king_distance as usize);
                    eg_score += self.options.get_passed_pawn_king_attacked_penalty(opponent_king_distance as usize);
                }
            }
        }

        (score, eg_score)
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

#[inline]
fn calc_hash(pattern: u64) -> u64 {
    let mut hash = (pattern.wrapping_mul(54043197675929600) >> 12) ^ (pattern << 15) ^ (pattern << 20);
    hash ^= hash.wrapping_mul(54043197675929600);

    hash >> (64 - 13)
}

#[inline]
pub fn calc_white_pawn_hash(pawns: u64, king_pos: i32) -> u64 {
    let king_half = (king_pos & 7) / 4;
    let pattern = if king_half == 0 {
        ((pawns & 0xF000000000000000) >> 60) |
            ((pawns & 0x00F0000000000000) >> 48) |
            ((pawns & 0x0000F00000000000) >> 36) |
            ((pawns & 0x000000F000000000) >> 24)

    } else {
        ((pawns & 0x0F00000000000000) >> 56) |
            ((pawns & 0x000F000000000000) >> 44) |
            ((pawns & 0x00000F0000000000) >> 32) |
            ((pawns & 0x0000000F00000000) >> 20)
    };

    calc_hash(pattern) * (2 - king_half as u64)
}

#[inline]
pub fn calc_black_pawn_hash(pawns: u64, king_pos: i32) -> u64 {
    let king_half = (king_pos & 7) / 4;
    let pattern = if king_half == 0 {
        (pawns & 0x000000000000000F) |
            ((pawns & 0x0000000000000F00) >> 4) |
            ((pawns & 0x00000000000F0000) >> 8) |
            ((pawns & 0x000000000F000000) >> 12)

    } else {
        ((pawns & 0x00000000000000F0) >> 4) |
            ((pawns & 0x000000000000F000) >> 8) |
            ((pawns & 0x0000000000F00000) >> 12) |
            ((pawns & 0x00000000F0000000) >> 16)
    };

    calc_hash(pattern) * (king_half as u64 + 1)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::{create_from_fen};
    use crate::magics::initialize_magics;

    #[test]
    fn check_correct_eval_score_for_mirrored_pos() {
        initialize_magics();

        assert_eq!(create_from_fen("8/8/8/5k2/4r1p1/6P1/3K1P2/8 b - - 0 80").get_score(),
                   -create_from_fen("8/3k1p2/6p1/4R1P1/5K2/8/8/8 w - - 0 80").get_score());
    }

    #[test]
    fn check_white_pawn_patterns() {
        assert_eq!(calc_white_pawn_hash(0b00001111_00001111_00001111_00001111_00000000_00000000_00000000_00000000, 63),
                   calc_black_pawn_hash(0b00000000_00000000_00000000_00000000_00001111_00001111_00001111_00001111, 0));

        assert_eq!(calc_white_pawn_hash(0b00001110_00001101_00001011_00000111_00000000_00000000_00000000_00000000, 63),
                   calc_black_pawn_hash(0b00000000_00000000_00000000_00000000_00000111_00001011_00001101_00001110, 0));

        assert_eq!(calc_white_pawn_hash(0b11110000_11110000_11110000_11110000_00000000_00000000_00000000_00000000, 56),
                   calc_black_pawn_hash(0b00000000_00000000_00000000_00000000_11110000_11110000_11110000_11110000, 7));

        assert_eq!(calc_white_pawn_hash(0b11100000_11010000_10110000_01110000_00000000_00000000_00000000_00000000, 56),
                   calc_black_pawn_hash(0b00000000_00000000_00000000_00000000_01110000_10110000_11010000_11100000, 7));
    }
}
