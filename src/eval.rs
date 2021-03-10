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
    fn king_threat(&self, pawn_threats: u32, minor_threats: u32, rook_threats: u32, queen_threats: u32) -> i32;
}

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

        // Knights
        let mut white_minor_threats = 0;
        let white_knights = self.get_bitboard(N);
        for pos in BitBoard(white_knights) {
            let possible_moves = get_knight_attacks(pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_knight_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_knight_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                white_minor_threats += 1;
            }
        }

        let empty_or_white = empty_board | white_pieces;
        let black_safe_targets = empty_or_white & !white_pawn_attacks;

        let black_knights = self.get_bitboard(-N);
        let mut black_minor_threats = 0;
        for pos in BitBoard(black_knights) {
            let possible_moves = get_knight_attacks(pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_knight_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_knight_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                black_minor_threats += 1;
            }
        }

        let mut pinnable_black_pieces = black_knights | black_rooks;

        // Bishops
        let white_bishops = self.get_bitboard(B);
        for pos in BitBoard(white_bishops) {
            let possible_moves = get_bishop_attacks(occupied, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_bishop_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_bishop_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                white_minor_threats += 1;
            }

            if possible_moves & pinnable_black_pieces != 0 {
                score += self.options.get_bishop_pin_bonus();
                eg_score += self.options.get_eg_bishop_pin_bonus();
            }
        }

        let mut pinnable_white_pieces = white_knights | white_rooks;
        let black_bishops = self.get_bitboard(-B);
        for pos in BitBoard(black_bishops) {
            let possible_moves = get_bishop_attacks(occupied, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_bishop_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_bishop_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                black_minor_threats += 1;
            }

            if possible_moves & pinnable_white_pieces != 0 {
                score -= self.options.get_bishop_pin_bonus();
                eg_score -= self.options.get_eg_bishop_pin_bonus();
            }
        }

        // Rooks
        let mut white_rook_threats = 0;
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

            let possible_moves = get_rook_attacks(occupied, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_rook_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_rook_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                white_rook_threats += 1;
            }

            if possible_moves & pinnable_black_pieces != 0 {
                score += self.options.get_rook_pin_bonus();
                eg_score += self.options.get_eg_rook_pin_bonus();
            }
        }

        let mut black_rook_threats = 0;
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

            let possible_moves = get_rook_attacks(occupied, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_rook_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_rook_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                black_rook_threats += 1;
            }

            if possible_moves & pinnable_white_pieces != 0 {
                score -= self.options.get_rook_pin_bonus();
                eg_score -= self.options.get_eg_rook_pin_bonus();
            }
        }

        // Queens
        let mut white_queen_threats = 0;
        for pos in BitBoard(white_queens) {
            let possible_moves = get_queen_attacks(occupied, pos as i32);

            let move_count = (possible_moves & white_safe_targets).count_ones();
            score += self.options.get_queen_mob_bonus(move_count as usize);
            eg_score += self.options.get_eg_queen_mob_bonus(move_count as usize);

            if possible_moves & black_king_danger_zone != 0 {
                white_queen_threats += 1;
            }
        }

        if white_queens & black_king_danger_zone != 0 {
            white_queen_threats += 1
        }

        let mut black_queen_threats = 0;
        for pos in BitBoard(black_queens) {
            let possible_moves = get_queen_attacks(occupied, pos as i32);

            let move_count = (possible_moves & black_safe_targets).count_ones();
            score -= self.options.get_queen_mob_bonus(move_count as usize);
            eg_score -= self.options.get_eg_queen_mob_bonus(move_count as usize);

            if possible_moves & white_king_danger_zone != 0 {
                black_queen_threats += 1;
            }
        }

        if black_queens & white_king_danger_zone != 0 {
            black_queen_threats += 1;
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
        let white_pawn_threats = (white_pawn_attacks & black_king_danger_zone).count_ones();
        interpolated_score += self.king_threat(white_pawn_threats, white_minor_threats, white_rook_threats, white_queen_threats);

        let black_pawn_threats = (black_pawn_attacks & white_king_danger_zone).count_ones();
        interpolated_score -= self.king_threat(black_pawn_threats, black_minor_threats, black_rook_threats, black_queen_threats);

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

    #[inline]
    fn king_threat(&self, pawn_threats: u32, minor_threats: u32, rook_threats: u32, queen_threats: u32) -> i32 {
        let threat_combo = unsafe { PAWN_COMBO_INDEXES.get_unchecked(pawn_threats as usize) } +
            unsafe { MINOR_COMBO_INDEXES.get_unchecked(minor_threats as usize ) } +
            unsafe { ROOK_COMBO_INDEXES.get_unchecked(rook_threats as usize ) } +
            unsafe { QUEEN_COMBO_INDEXES.get_unchecked(queen_threats as usize ) };
        self.options.get_king_threat_by_piece_combo(threat_combo as usize)
    }
}

// Numbers offsets for different piece types and number of pieces that can be combined
// to an index into the king threat value table.
// These are prime numbers to ensure that the resulting index for each piece combination is unique.
const PAWN_COMBO_INDEXES: [u32; 8] = [0, 1, 3, 3, 3, 3, 3, 3];
const MINOR_COMBO_INDEXES: [u32; 8] = [0, 5, 7, 11, 11, 11, 11, 11];
const ROOK_COMBO_INDEXES: [u32; 8] = [0, 13, 17, 17, 17, 17, 17, 17];
const QUEEN_COMBO_INDEXES: [u32; 8] = [0, 19, 23, 23, 23, 23, 23, 23];

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
