/*
 * Velvet Chess Engine
 * Copyright (C) 2022 mhonert (https://github.com/mhonert)
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

use std::io::{BufReader, Error};
use std::sync::Arc;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::bitboard::{h_mirror, v_mirror};
use crate::colors::{BLACK, Color, WHITE};
use crate::compression::{BitReader, CodeBook, read_value};
use crate::pieces::{B, N, P, Q, R};

pub struct BitBases {
    king_indices_rotate: [[u16; 64]; 64],
    king_indices: [[u16; 64]; 64],
    bitbases: Vec<BitBase>
}

const KPK: usize = 0;
const KKP: usize = 1;
const KRK: usize = 2;
const KKR: usize = 3;
const KQK: usize = 4;
const KKQ: usize = 5;
const KPKP: usize = 6;
const KNKP: usize = 7;
const KPKN: usize = 8;
const KBKP: usize = 9;
const KPKB: usize = 10;

const KRKP: usize = 11;
const KPKR: usize = 12;
const KQKP: usize = 13;
const KPKQ: usize = 14;

const KRKR: usize = 15;
const KRKN: usize = 16;
const KNKR: usize = 17;
const KRKB: usize = 18;
const KBKR: usize = 19;

const KQKN: usize = 20;
const KNKQ: usize = 21;
const KQKB: usize = 22;
const KBKQ: usize = 23;
const KQKR: usize = 24;
const KRKQ: usize = 25;

const KQKQ: usize = 26;

const KNKN: usize = 27;
const KBKB: usize = 28;

const KBKN: usize = 29;
const KNKB: usize = 30;

const KPPK: usize = 31;
const KKPP: usize = 32;


impl BitBases {
    pub fn new() -> Arc<BitBases> {
        let mut bitbases = BitBases{
            king_indices: [[0; 64]; 64],
            king_indices_rotate: [[0; 64]; 64],
            bitbases: Vec::with_capacity(35)
        };

        let mut idx = 0;
        let mut rotate_idx = 0;
        for wk in 0..64 {
            if wk & 7 > 3 {
                continue;
            }
            for bk in 0..64 {
                let row_distance = (wk / 8).max(bk / 8) - (wk / 8).min(bk / 8);
                let col_distance = (wk & 7).max(bk & 7) - (wk & 7).min(bk & 7);
                let distance = row_distance.max(col_distance);

                if distance <= 1 {
                    continue;
                }

                if wk / 8 <= 3 {
                    bitbases.king_indices_rotate[wk][bk] = rotate_idx;
                    rotate_idx += 1;
                }

                bitbases.king_indices[wk][bk] = idx;
                idx += 1;
            }
        }

        // 3-men
        bitbases.register(KPK, BitBase::new(64, false, &include_bytes!("../egbb/kPk.ebb")[..]));
        bitbases.register(KKP, BitBase::new(64, false, &include_bytes!("../egbb/kkP.ebb")[..]));
        bitbases.register(KRK, BitBase::new(64, true, &include_bytes!("../egbb/kRk.ebb")[..]));
        bitbases.register(KKR, BitBase::new(64, true, &include_bytes!("../egbb/kkR.ebb")[..]));
        bitbases.register(KQK, BitBase::new(64, true, &include_bytes!("../egbb/kQk.ebb")[..]));
        bitbases.register(KKQ, BitBase::new(64, true, &include_bytes!("../egbb/kkQ.ebb")[..]));

        // 4-men
        bitbases.register(KPKP, BitBase::new(4096, false, &include_bytes!("../egbb/kPkP.ebb")[..]));
        bitbases.register(KNKP, BitBase::new(4096, false, &include_bytes!("../egbb/kNkP.ebb")[..]));
        bitbases.register(KPKN, BitBase::new(4096, false, &include_bytes!("../egbb/kPkN.ebb")[..]));
        bitbases.register(KBKP, BitBase::new(4096, false, &include_bytes!("../egbb/kBkP.ebb")[..]));
        bitbases.register(KPKB, BitBase::new(4096, false, &include_bytes!("../egbb/kPkB.ebb")[..]));
        bitbases.register(KRKP, BitBase::new(4096, false, &include_bytes!("../egbb/kRkP.ebb")[..]));
        bitbases.register(KPKR, BitBase::new(4096, false, &include_bytes!("../egbb/kPkR.ebb")[..]));
        bitbases.register(KQKP, BitBase::new(4096, false, &include_bytes!("../egbb/kQkP.ebb")[..]));
        bitbases.register(KPKQ, BitBase::new(4096, false, &include_bytes!("../egbb/kPkQ.ebb")[..]));

        bitbases.register(KRKR, BitBase::new(4096, true, &include_bytes!("../egbb/kRkR.ebb")[..]));
        bitbases.register(KRKN, BitBase::new(4096, true, &include_bytes!("../egbb/kRkN.ebb")[..]));
        bitbases.register(KNKR, BitBase::new(4096, true, &include_bytes!("../egbb/kNkR.ebb")[..]));

        bitbases.register(KRKB, BitBase::new(4096, true, &include_bytes!("../egbb/kRkB.ebb")[..]));
        bitbases.register(KBKR, BitBase::new(4096, true, &include_bytes!("../egbb/kBkR.ebb")[..]));

        bitbases.register(KQKN, BitBase::new(4096, true, &include_bytes!("../egbb/kQkN.ebb")[..]));
        bitbases.register(KNKQ, BitBase::new(4096, true, &include_bytes!("../egbb/kNkQ.ebb")[..]));

        bitbases.register(KQKB, BitBase::new(4096, true, &include_bytes!("../egbb/kQkB.ebb")[..]));
        bitbases.register(KBKQ, BitBase::new(4096, true, &include_bytes!("../egbb/kBkQ.ebb")[..]));

        bitbases.register(KQKR, BitBase::new(4096, true, &include_bytes!("../egbb/kQkR.ebb")[..]));
        bitbases.register(KRKQ, BitBase::new(4096, true, &include_bytes!("../egbb/kRkQ.ebb")[..]));

        bitbases.register(KQKQ, BitBase::new(4096, true, &include_bytes!("../egbb/kQkQ.ebb")[..]));

        bitbases.register(KNKN, BitBase::new(4096, true, &include_bytes!("../egbb/kNkN.ebb")[..]));
        bitbases.register(KBKB, BitBase::new(4096, true, &include_bytes!("../egbb/kBkB.ebb")[..]));

        bitbases.register(KBKN, BitBase::new(4096, true, &include_bytes!("../egbb/kBkN.ebb")[..]));
        bitbases.register(KNKB, BitBase::new(4096, true, &include_bytes!("../egbb/kNkB.ebb")[..]));

        bitbases.register(KPPK, BitBase::new(4096, false, &include_bytes!("../egbb/kPPk.ebb")[..]));
        bitbases.register(KKPP, BitBase::new(4096, false, &include_bytes!("../egbb/kkPP.ebb")[..]));

        Arc::new(bitbases)
    }

    fn register(&mut self, id: usize, bitbase: BitBase) {
        assert_eq!(self.bitbases.len(), id);
        self.bitbases.push(bitbase);
    }

    pub fn probe(&self, stm: Color, piece_count: usize, white_king: u32, black_king: u32,
                 bitboards: &[u64; 13], can_castle: bool, can_en_passant: bool) -> Option<i8> {
        if piece_count > 4 {
            return None;
        }

        let white_pawns = bitboards[(P + 6) as usize];
        let black_pawns = bitboards[(-P + 6) as usize];

        let white_knights = bitboards[(N + 6) as usize];
        let black_knights = bitboards[(-N + 6) as usize];

        let white_bishops = bitboards[(B + 6) as usize];
        let black_bishops = bitboards[(-B + 6) as usize];

        let white_queens = bitboards[(Q + 6) as usize];
        let black_queens = bitboards[(-Q + 6) as usize];

        let white_rooks = bitboards[(R + 6) as usize];
        let black_rooks = bitboards[(-R + 6) as usize];

        if piece_count == 3 {
            return if white_pawns != 0 {
                self.probe3(KPK, KKP, stm, white_king, black_king, white_pawns)
            } else if black_pawns != 0 {
                self.probe3(KKP, KPK, stm, white_king, black_king, black_pawns)
            } else if white_rooks != 0 {
                // if !can_castle {
                    self.probe3(KRK, KKR, stm, white_king, black_king, white_rooks)
                // } else {
                //     None
                // }
            } else if black_rooks != 0 {
                // if !can_castle {
                    self.probe3(KKR, KRK, stm, white_king, black_king, black_rooks)
                // } else {
                //     None
                // }
            } else if white_queens != 0 {
                self.probe3(KQK, KKQ, stm, white_king, black_king, white_queens)
            } else if black_queens != 0 {
                self.probe3(KKQ, KQK, stm, white_king, black_king, black_queens)
            } else {
                None
            };
        }

        if white_pawns != 0 && black_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KPKP, KPKP, stm, white_king, black_king, white_pawns, black_pawns)
            // } else {
            //     None
            // }
        } else if white_knights != 0 && black_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KNKP, KPKN, stm, white_king, black_king, white_knights, black_pawns)
            // } else {
            //     None
            // }
        } else if black_knights != 0 && white_pawns != 0 {
            self.probe4(KPKN, KNKP, stm, white_king, black_king, white_pawns, black_knights)
        } else if white_bishops != 0 && black_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KBKP, KPKB, stm, white_king, black_king, white_bishops, black_pawns)
            // } else {
            //     None
            // }
        } else if black_bishops != 0 && white_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KPKB, KBKP, stm, white_king, black_king, white_pawns, black_bishops)
            // } else {
            //     None
            // }
        } else if white_rooks != 0 && black_pawns != 0 {
            // if !can_castle {
                self.probe4(KRKP, KPKR, stm, white_king, black_king, white_rooks, black_pawns)
            // } else {
            //     None
            // }
        } else if black_rooks != 0 && white_pawns != 0 {
            // if !can_castle {
                self.probe4(KPKR, KRKP, stm, white_king, black_king, white_pawns, black_rooks)
            // } else {
            //     None
            // }
        } else if white_queens != 0 && black_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KQKP, KPKQ, stm, white_king, black_king, white_queens, black_pawns)
            // } else {
            //     None
            // }
        } else if black_queens != 0 && white_pawns != 0 {
            // if !can_en_passant {
                self.probe4(KPKQ, KQKP, stm, white_king, black_king, white_pawns, black_queens)
            // } else {
            //     None
            // }
        } else if white_rooks != 0 && black_rooks != 0 {
            // if !can_castle {
                self.probe4(KRKR, KRKR, stm, white_king, black_king, white_rooks, black_rooks)
            // } else {
            //     None
            // }
        } else if white_rooks != 0 && black_knights != 0 {
            // if !can_castle {
                self.probe4(KRKN, KNKR, stm, white_king, black_king, white_rooks, black_knights)
            // } else {
            //     None
            // }
        } else if black_rooks != 0 && white_knights != 0 {
            // if !can_castle {
                self.probe4(KNKR, KRKN, stm, white_king, black_king, white_knights, black_rooks)
            // } else {
            //     None
            // }
        } else if white_rooks != 0 && black_bishops != 0 {
            // if !can_castle {
                self.probe4(KRKB, KBKR, stm, white_king, black_king, white_rooks, black_bishops)
            // } else {
            //     None
            // }
        } else if black_rooks != 0 && white_bishops != 0 {
            // if !can_castle {
                self.probe4(KBKR, KRKB, stm, white_king, black_king, white_bishops, black_rooks)
            // } else {
            //     None
            // }
        } else if white_queens != 0 && black_knights != 0 {
            self.probe4(KQKN, KNKQ, stm, white_king, black_king, white_queens, black_knights)
        } else if black_queens != 0 && white_knights != 0 {
            self.probe4(KNKQ, KQKN, stm, white_king, black_king, white_knights, black_queens)
        } else if white_queens != 0 && black_bishops != 0 {
            self.probe4(KQKB, KBKQ, stm, white_king, black_king, white_queens, black_bishops)
        } else if black_queens != 0 && white_bishops != 0 {
            self.probe4(KBKQ, KQKB, stm, white_king, black_king, white_bishops, black_queens)
        } else if white_queens != 0 && black_rooks != 0 {
            // if !can_castle {
                self.probe4(KQKR, KRKQ, stm, white_king, black_king, white_queens, black_rooks)
            // } else {
            //     None
            // }
        } else if black_queens != 0 && white_rooks != 0 {
            // if !can_castle {
                self.probe4(KRKQ, KQKR, stm, white_king, black_king, white_rooks, black_queens)
            // } else {
            //     None
            // }
        } else if black_queens != 0 && white_queens != 0 {
            self.probe4(KQKQ, KQKQ, stm, white_king, black_king, white_queens, black_queens)
        } else if black_knights != 0 && white_knights != 0 {
            self.probe4(KNKN, KNKN, stm, white_king, black_king, white_knights, black_knights)
        } else if black_bishops != 0 && white_bishops != 0 {
            self.probe4(KBKB, KBKB, stm, white_king, black_king, white_bishops, black_bishops)
        } else if white_bishops != 0 && black_knights != 0 {
            self.probe4(KBKN, KNKB, stm, white_king, black_king, white_bishops, black_knights)
        } else if black_bishops != 0 && white_knights != 0 {
            self.probe4(KNKB, KBKN, stm, white_king, black_king, white_knights, black_bishops)
        // } else if white_pawns != 0 {
        //     let (p1, p2) = split_bb(white_pawns);
        //     self.probe4(KPPK, KKPP, stm, white_king, black_king, p1, p2)
        // } else if black_pawns != 0 {
        //     let (p1, p2) = split_bb(black_pawns);
        //     self.probe4(KKPP, KPPK, stm, white_king, black_king, p1, p2)
        } else {
            None
        }
    }

    fn probe3(&self, wtm_bitbase_idx: usize, btm_bitbase_idx: usize, stm: Color, white_king: u32, black_king: u32, piece3: u64) -> Option<i8> {
        if stm == WHITE {
            let bitbase = unsafe { self.bitbases.get_unchecked(wtm_bitbase_idx) };
            self.probe_bb(bitbase, 3, 1, white_king, black_king, board_pos(piece3), 0)
        } else {
            let bitbase = unsafe { self.bitbases.get_unchecked(btm_bitbase_idx) };
            self.probe_bb(bitbase, 3, -1, v_mirror(black_king), v_mirror(white_king), v_mirror(board_pos(piece3)), 0)
        }
    }

    fn probe4(&self, wtm_bitbase_idx: usize, btm_bitbase_idx: usize, stm: Color, white_king: u32, black_king: u32, piece3: u64, piece4: u64) -> Option<i8> {
        if stm == WHITE {
            let bitbase = unsafe { self.bitbases.get_unchecked(wtm_bitbase_idx) };
            self.probe_bb(bitbase, 4, 1, white_king, black_king, board_pos(piece3), board_pos(piece4))
        } else {
            let bitbase = unsafe { self.bitbases.get_unchecked(btm_bitbase_idx) };
            self.probe_bb(bitbase, 4, -1, v_mirror(black_king), v_mirror(white_king),
                          v_mirror(board_pos(piece4)), v_mirror(board_pos(piece3)))
        }
    }

    fn probe_bb(&self, bitbase: &BitBase, piece_count: usize, stm_adjustment: i8, mut own_king: u32, mut opp_king: u32, mut piece3: u32, mut piece4: u32) -> Option<i8> {
        if own_king & 7 > 3 {
            own_king = h_mirror(own_king);
            opp_king = h_mirror(opp_king);
            piece3 = h_mirror(piece3);
            if piece_count > 3 {
                piece4 = h_mirror(piece4);
            }
        }

        if bitbase.rotate && own_king / 8 > 3 {
            own_king = v_mirror(own_king);
            opp_king = v_mirror(opp_king);
            piece3 = v_mirror(piece3);
            if piece_count > 3 {
                piece4 = v_mirror(piece4);
            }
        }

        let king_index = if bitbase.rotate {
            *unsafe { self.king_indices_rotate.get_unchecked(own_king as usize).get_unchecked(opp_king as usize) }
        } else {
            *unsafe { self.king_indices.get_unchecked(own_king as usize).get_unchecked(opp_king as usize) }
        };

        let index = if piece_count == 3 { piece3 } else { piece3 * 64 + piece4 };

        if let Some(result) = bitbase.probe(king_index as usize, index as usize) {
            return Some(result * stm_adjustment);
        }

        None
    }

}

fn split_bb(bb: u64) -> (u64, u64) {
    let p1 = 1 << bb.trailing_zeros() as u64;
    let p2 = bb ^ p1;
    assert_ne!(p2, 0);
    (p1, p2)
}

fn board_pos(bb: u64) -> u32 {
    bb.trailing_zeros()
}

struct BitBase {
    rotate: bool,
    entries: Vec<Entry>,
}

struct Entry {
    default_result: i8,
    bucket1: Vec<(u16, u8)>,
    bucket2: Vec<(u16, u8)>,
}

impl BitBase {
    pub fn new(entry_len: usize, rotate: bool, input: &[u8]) -> BitBase {
        let mut reader = BufReader::new(input);

        let mut bitbase = BitBase{
            rotate,
            entries: Vec::with_capacity(entry_len),
        };


        let count_shift = if rotate { 10 } else { 11 };
        let idx_mask = (1 << count_shift) - 1;

        for i in 0..entry_len {
            let entry_header = reader.read_u16::<LittleEndian>().expect("Could not read entry header");
            let default_result = (entry_header >> 13) as i8 - 1;
            let inputs1_len = entry_header & ((1 << 13) - 1);

            let inputs2_len = reader.read_u16::<LittleEndian>().expect("Could not read entry header");


            bitbase.entries.push(Entry{
                default_result,
                bucket1: Vec::with_capacity(inputs1_len as usize),
                bucket2: Vec::with_capacity(inputs2_len as usize),
            });


            for _ in 0..inputs1_len {
                let entry = reader.read_u16::<LittleEndian>().expect("Could not read entry header");
                let king_idx = entry & idx_mask;
                let count = entry >> count_shift;
                bitbase.entries[i].bucket1.push((king_idx, count as u8));
            }

            for _ in 0..inputs2_len {
                let entry = reader.read_u16::<LittleEndian>().expect("Could not read entry header");
                let king_idx = entry & idx_mask;
                let count = entry >> count_shift;
                bitbase.entries[i].bucket2.push((king_idx, count as u8));
            }
        }

        bitbase
    }

    pub fn probe(&self, king_index: usize, index: usize) -> Option<i8> {
        let entry = &self.entries[index];

        if contains(&entry.bucket1, king_index as u16) {
            Some(to_wdl(entry.default_result, 0))
        } else if contains(&entry.bucket2, king_index as u16) {
            Some(to_wdl(entry.default_result, 1))
        } else {
            Some(entry.default_result)
        }
    }

}

fn to_wdl(default_result: i8, bucket: i8) -> i8 {
    if default_result == -1 {
        if bucket == 0 { 0 } else { 1 }
    } else if default_result == 0 {
        if bucket == 0 { -1 } else { 1 }
    } else {
        if bucket == 0 { 0 } else { -1 }
    }
}


// fn contains(entries: &[(u16, u8)], king_index: u16) -> bool {
//     if entries.is_empty() {
//         return false;
//     }
//
//     for i in 0..entries.len() {
//         let (idx, _) = entries[i];
//         if idx == king_index {
//             return true;
//         }
//
//         if idx > king_index {
//             if i == 0 {
//                 return false;
//             }
//
//             let (idx, count) = entries[i - 1];
//             return (idx + count as u16) >= king_index;
//         }
//     }
//
//     if let Some(&(idx, count)) = entries.last() {
//         return (idx + count as u16) >= king_index;
//     }
//
//     false
// }

fn contains(entries: &[(u16, u8)], king_index: u16) -> bool {
    contains_rec(entries, None, king_index)
}

fn contains_rec(entries: &[(u16, u8)], prev: Option<(u16, u8)>, king_index: u16) -> bool {
    if entries.is_empty() {
        return if let Some((idx, count)) = prev {
            (idx + count as u16) >= king_index
        } else {
            false
        }
    }

    let i = entries.len() / 2;
    let entry = entries[i];
    let (idx, _) = entry;
    if idx == king_index {
        return true;
    }

    if idx > king_index {
        contains_rec(&entries[0..i], prev, king_index)
    } else {
        contains_rec(&entries[i + 1..], Some(entry), king_index)
    }
}
