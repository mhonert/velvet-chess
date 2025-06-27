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
use crate::align::A64;
use crate::bitboard::{BitBoard, create_blocker_permutations, gen_bishop_attacks, gen_rook_attacks, mask_without_outline};

pub fn get_bishop_attacks(occupied_bb: u64, pos: usize) -> BitBoard {
    unsafe {
        let magics_ptr = &raw const MAGICS.0;
        let &Magic { b_offset, b_mask, b_number, .. } = (*magics_ptr).get_unchecked(pos);
        let attacks_ptr = &raw const ATTACKS.0;
        BitBoard(*(*attacks_ptr).get_unchecked(b_offset + ((occupied_bb | b_mask).wrapping_mul(b_number) >> (64 - 9)) as usize))
    }
}

pub fn get_rook_attacks(occupied_bb: u64, pos: usize) -> BitBoard {
    unsafe {
        let magics_ptr = &raw const MAGICS.0;
        let &Magic { r_offset, r_mask, r_number, .. } = (*magics_ptr).get_unchecked(pos);
        let attacks_ptr = &raw const ATTACKS.0;
        
        BitBoard(*(*attacks_ptr).get_unchecked(r_offset + ((occupied_bb | r_mask).wrapping_mul(r_number) >> (64 - 12)) as usize))
    }
}

pub fn get_queen_attacks(occupied_bb: u64, pos: usize) -> BitBoard {
    unsafe {
        let magics_ptr = &raw const MAGICS.0;
        let &Magic { b_offset, r_offset, r_mask, b_mask, r_number, b_number } = (*magics_ptr).get_unchecked(pos);
        let attacks_ptr = &raw const ATTACKS.0;
        
        BitBoard(*(*attacks_ptr).get_unchecked(b_offset + ((occupied_bb | b_mask).wrapping_mul(b_number) >> (64 - 9)) as usize)
            | *(*attacks_ptr).get_unchecked(r_offset  + ((occupied_bb | r_mask).wrapping_mul(r_number) >> (64 - 12)) as usize))
    }
}

struct Magic {
    r_mask: u64,
    r_number: u64,
    r_offset: usize,
    b_mask: u64,
    b_number: u64,
    b_offset: usize,
}

const EMPTY_MAGIC: Magic = Magic { r_mask: 0, r_number: 0, r_offset: 0, b_mask: 0, b_number: 0, b_offset: 0 };


// 2025: replaced own (black) magic numbers with more compact ones from
// http://talkchess.com/forum/viewtopic.php?t=64790
#[rustfmt::skip]
static ROOK_MAGIC_NUMS: [u64; 64] = [
    0x80280013ff84ffff, 0x5ffbfefdfef67fff, 0xffeffaffeffdffff, 0x003000900300008a,
    0x0050028010500023, 0x0020012120a00020, 0x0030006000c00030, 0x0058005806b00002,
    0x7fbff7fbfbeafffc, 0x0000140081050002, 0x0000180043800048, 0x7fffe800021fffb8,
    0xffffcffe7fcfffaf, 0x00001800c0180060, 0x4f8018005fd00018, 0x0000180030620018,
    0x00300018010c0003, 0x0003000c0085ffff, 0xfffdfff7fbfefff7, 0x7fc1ffdffc001fff,
    0xfffeffdffdffdfff, 0x7c108007befff81f, 0x20408007bfe00810, 0x0400800558604100,
    0x0040200010080008, 0x0010020008040004, 0xfffdfefff7fbfff7, 0xfebf7dfff8fefff9,
    0xc00000ffe001ffe0, 0x4af01f00078007c3, 0xbffbfafffb683f7f, 0x0807f67ffa102040,
    0x200008e800300030, 0x0000008780180018, 0x0000010300180018, 0x4000008180180018,
    0x008080310005fffa, 0x4000188100060006, 0xffffff7fffbfbfff, 0x0000802000200040,
    0x20000202ec002800, 0xfffff9ff7cfff3ff, 0x000000404b801800, 0x2000002fe03fd000,
    0xffffff6ffe7fcffd, 0xbff7efffbfc00fff, 0x000000100800a804, 0x6054000a58005805,
    0x0829000101150028, 0x00000085008a0014, 0x8000002b00408028, 0x4000002040790028,
    0x7800002010288028, 0x0000001800e08018, 0xa3a80003f3a40048, 0x2003d80000500028,
    0xfffff37eefefdfbe, 0x40000280090013c1, 0xbf7ffeffbffaf71f, 0xfffdffff777b7d6e,
    0x48300007e8080c02, 0xafe0000fff780402, 0xee73fffbffbb77fe, 0x0002000308482882,
];

#[rustfmt::skip]
static ROOK_MAGIC_OFFSETS: [u32; 64] = [
    10890, 50579, 62020, 67322, 80251, 58503, 51175, 83130,
    50430, 21613, 72625, 80755, 69753, 26973, 84972, 31958,
    69272, 48372, 65477, 43972, 57154, 53521, 30534, 16548,
    46407, 11841, 21112, 44214, 57925, 29574, 17309, 40143,
    64659, 70469, 62917, 60997, 18554, 14385,     0, 38091,
    25122, 60083, 72209, 67875, 56290, 43807, 73365, 76398,
    20024,  9513, 24324, 22996, 23213, 56002, 22809, 44545,
    36072,  4750,  6014, 36054, 78538, 28745,  8555,  1009,
];

#[rustfmt::skip]
static BISHOP_MAGIC_NUMS: [u64; 64] = [
    0xa7020080601803d8, 0x13802040400801f1, 0x0a0080181001f60c, 0x1840802004238008, 
    0xc03fe00100000000, 0x24c00bffff400000, 0x0808101f40007f04, 0x100808201ec00080, 
    0xffa2feffbfefb7ff, 0x083e3ee040080801, 0xc0800080181001f8, 0x0440007fe0031000,
    0x2010007ffc000000, 0x1079ffe000ff8000, 0x3c0708101f400080, 0x080614080fa00040, 
    0x7ffe7fff817fcff9, 0x7ffebfffa01027fd, 0x53018080c00f4001, 0x407e0001000ffb8a, 
    0x201fe000fff80010, 0xffdfefffde39ffef, 0xcc8808000fbf8002, 0x7ff7fbfff8203fff,
    0x8800013e8300c030, 0x0420009701806018, 0x7ffeff7f7f01f7fd, 0x8700303010c0c006, 
    0xc800181810606000, 0x20002038001c8010, 0x087ff038000fc001, 0x00080c0c00083007, 
    0x00000080fc82c040, 0x000000407e416020, 0x00600203f8008020, 0xd003fefe04404080,
    0xa00020c018003088, 0x7fbffe700bffe800, 0x107ff00fe4000f90, 0x7f8fffcff1d007f8, 
    0x0000004100f88080, 0x00000020807c4040, 0x00000041018700c0, 0x0010000080fc4080, 
    0x1000003c80180030, 0xc10000df80280050, 0xffffffbfeff80fdc, 0x000000101003f812,
    0x0800001f40808200, 0x084000101f3fd208, 0x080000000f808081, 0x0004000008003f80, 
    0x08000001001fe040, 0x72dd000040900a00, 0xfffffeffbfeff81d, 0xcd8000200febf209, 
    0x100000101ec10082, 0x7fbaffffefe0c02f, 0x7f83fffffff07f7f, 0xfff1fffffff7ffc1,
    0x0878040000ffe01f, 0x945e388000801012, 0x0840800080200fda, 0x100000c05f582008, 
];

#[rustfmt::skip]
static BISHOP_MAGIC_OFFSETS: [u32; 64] = [
    60984, 66046, 32910, 16369, 42115,   835, 18910, 25911, 
    63301, 16063, 17481, 59361, 18735, 61249, 68938, 61791, 
    21893, 62068, 19829, 26091, 15815, 16419, 59777, 16288,
    33235, 15459, 15863, 75555, 79445, 15917,  8512, 73069, 
    16078, 19168, 11056, 62544, 80477, 75049, 32947, 59172, 
    55845, 61806, 73601, 15546, 45243, 20333, 33402, 25917,
    32875,  4639, 17077, 62324, 18159, 61436, 57073, 61025, 
    81259, 64083, 56114, 57058, 58912, 22194, 70880, 11140
];

static mut ATTACKS: A64<[u64; 87988]> = A64([0; 87988]);

static mut MAGICS: A64<[Magic; 64]> = A64([EMPTY_MAGIC; 64]);

static mut RAYS: A64<[u64; 4096]> = A64([0; 4096]);

pub fn init() {
    init_attack_tables();
    init_ray_tables();
}

pub fn get_ray(idx: u16) -> BitBoard {
    unsafe {
        let ptr = &raw const RAYS.0;
        BitBoard(*(*ptr).get_unchecked(idx as usize))
    }
}

fn init_attack_tables() {
    for pos in 0..64 {
        // Rook
        let r_move_mask = gen_rook_attacks(0, pos);
        let r_block_mask = mask_without_outline(r_move_mask, pos as u32);
        let r_blocker_count = r_block_mask.count_ones();

        let mut r_permutations: Vec<u64> = Vec::with_capacity(1 << r_blocker_count);
        create_blocker_permutations(&mut r_permutations, 0, r_block_mask);

        let r_magic_num = ROOK_MAGIC_NUMS[pos as usize];
        let r_magic_offset = ROOK_MAGIC_OFFSETS[pos as usize];

        let r_inv_block_mask = !r_block_mask;
        for &p in r_permutations.iter() {
            let move_targets = gen_rook_attacks(p, pos);

            let index = ((p | r_inv_block_mask).wrapping_mul(r_magic_num)) >> (64 - 12);
            unsafe { ATTACKS.0[index as usize + r_magic_offset as usize] = move_targets }
        }

        // Bishop
        let b_move_mask = gen_bishop_attacks(0, pos);
        let b_block_mask = mask_without_outline(b_move_mask, pos as u32);

        let b_blocker_count = b_block_mask.count_ones();

        let mut b_permutations: Vec<u64> = Vec::with_capacity(1 << b_blocker_count);
        create_blocker_permutations(&mut b_permutations, 0, b_block_mask);

        let b_magic_num = BISHOP_MAGIC_NUMS[pos as usize];

        let b_magic_offset = BISHOP_MAGIC_OFFSETS[pos as usize];

        let b_inv_block_mask = !b_block_mask;
        for &p in b_permutations.iter() {
            let move_targets = gen_bishop_attacks(p, pos);

            let index = ((p | b_inv_block_mask).wrapping_mul(b_magic_num)) >> (64 - 9);
            unsafe { ATTACKS.0[index as usize + b_magic_offset as usize] = move_targets }
        }

        unsafe {
            MAGICS.0[pos as usize] = Magic {
                r_mask: r_inv_block_mask,
                r_number: r_magic_num,
                r_offset: r_magic_offset as usize,
                b_mask: b_inv_block_mask,
                b_number: b_magic_num,
                b_offset: b_magic_offset as usize,
            }
        };
    }
}

fn init_ray_tables() {
    for start in 0..64 {
        for end in 0..64 {
            let idx = (start << 6) | end;
            if get_bishop_attacks(0, start).is_set(end) {
                let ray = get_bishop_attacks(1 << end, start).0 & get_bishop_attacks(1 << start, end).0;
                unsafe { RAYS.0[idx] = ray }
            } else if get_rook_attacks(0, start).is_set(end) {
                let ray = get_rook_attacks(1 << end, start).0 & get_rook_attacks(1 << start, end).0; 
                unsafe { RAYS.0[idx] = ray }
            }
        }
    }
}
