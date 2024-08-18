/*
 * Velvet Chess Engine
 * Copyright (C) 2024 mhonert (https://github.com/mhonert)
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
use crate::moves::Move;

#[cfg(not(feature = "fathomrs"))]
pub const HAS_TB_SUPPORT: bool = false;

#[cfg(feature = "fathomrs")]
pub const HAS_TB_SUPPORT: bool = true;

pub const DEFAULT_TB_PROBE_DEPTH: i32 = 0;

pub trait ProbeTB {
    fn probe_wdl(&self) -> Option<tb::TBResult>;
    fn probe_root(&self) -> Option<(tb::TBResult, Vec<Move>)>;
}

#[cfg(not(feature = "fathomrs"))]
pub mod tb {
    use crate::board::Board;
    use crate::moves::Move;
    use crate::syzygy::ProbeTB;

    #[derive(Eq, PartialEq, Copy, Clone)]
    #[allow(dead_code)]
    pub enum TBResult { Loss, BlessedLoss, Draw, CursedWin, Win }

    pub fn init(_path: String) -> bool {
        false
    }

    pub fn max_piece_count() -> u32 {
        0
    }

    impl ProbeTB for Board {
        fn probe_wdl(&self) -> Option<TBResult> {
            None
        }

        fn probe_root(&self) -> Option<(TBResult, Vec<Move>)> {
            None
        }
    }
}

#[cfg(feature = "fathomrs")]
pub mod tb {
    use fathomrs::tb::{extract_move, Promotion};
    use crate::bitboard::{v_mirror_i8};
    use crate::board::Board;
    use crate::colors::{BLACK, WHITE};
    use crate::moves::Move;
    use crate::pieces::{B, EMPTY, K, N, P, Q, R};
    use crate::syzygy::ProbeTB;
    use crate::uci_move::UCIMove;

    pub type TBResult = fathomrs::tb::TBResult;

    pub fn init(path: String) -> bool {
        fathomrs::tb::init(path)
    }

    pub fn max_piece_count() -> u32 {
        fathomrs::tb::max_piece_count()
    }

    impl ProbeTB for Board {
        fn probe_wdl(&self) -> Option<TBResult> {
            if self.halfmove_clock() != 0 || self.any_castling() || self.piece_count() > fathomrs::tb::max_piece_count() {
                return None;
            }

            let ep_target = self.enpassant_target();
            let ep = if ep_target != 0 {
                v_mirror_i8(ep_target as i8) as u16
            } else {
                0
            };

            fathomrs::tb::probe_wdl(
                self.get_all_piece_bitboard(WHITE).0.swap_bytes(),
                self.get_all_piece_bitboard(BLACK).0.swap_bytes(),
                (self.get_bitboard(K) | self.get_bitboard(-K)).0.swap_bytes(),
                (self.get_bitboard(Q) | self.get_bitboard(-Q)).0.swap_bytes(),
                (self.get_bitboard(R) | self.get_bitboard(-R)).0.swap_bytes(),
                (self.get_bitboard(B) | self.get_bitboard(-B)).0.swap_bytes(),
                (self.get_bitboard(N) | self.get_bitboard(-N)).0.swap_bytes(),
                (self.get_bitboard(P) | self.get_bitboard(-P)).0.swap_bytes(),
                ep,
                self.active_player().is_white()
            )
        }

        fn probe_root(&self) -> Option<(TBResult, Vec<Move>)> {
            if self.any_castling() || self.piece_count() > fathomrs::tb::max_piece_count() {
                return None;
            }


            let ep_target = self.enpassant_target();
            let ep = if ep_target != 0 {
                v_mirror_i8(ep_target as i8) as u16
            } else {
                0
            };

            let (result, moves) = fathomrs::tb::probe_root(
                self.get_all_piece_bitboard(WHITE).0.swap_bytes(),
                self.get_all_piece_bitboard(BLACK).0.swap_bytes(),
                (self.get_bitboard(K) | self.get_bitboard(-K)).0.swap_bytes(),
                (self.get_bitboard(Q) | self.get_bitboard(-Q)).0.swap_bytes(),
                (self.get_bitboard(R) | self.get_bitboard(-R)).0.swap_bytes(),
                (self.get_bitboard(B) | self.get_bitboard(-B)).0.swap_bytes(),
                (self.get_bitboard(N) | self.get_bitboard(-N)).0.swap_bytes(),
                (self.get_bitboard(P) | self.get_bitboard(-P)).0.swap_bytes(),
                self.halfmove_clock(),
                ep,
                self.active_player().is_white()
            );

            if fathomrs::tb::is_failed_result(result) {
                return None;
            }

            let best_tb_result = TBResult::from_result(result);

            Some((best_tb_result, moves.iter().filter(|&m| *m != 0).map(|&m| {
                let tb_result = TBResult::from_result(m);
                let (from, to, promotion) = extract_move(m);
                (tb_result, from, to, promotion)
            }).filter(|(tb_result, _, _, _)| tb_result < &best_tb_result)
                .map(|(_, from, to, promotion)| {
                    let promotion_piece = match promotion {
                        Promotion::Queen => Q,
                        Promotion::Rook => R,
                        Promotion::Bishop => B,
                        Promotion::Knight => N,
                        _ => EMPTY
                    };

                    UCIMove::new(v_mirror_i8(from), v_mirror_i8(to), promotion_piece).to_move(self)
                }).collect()))
        }
    }
}