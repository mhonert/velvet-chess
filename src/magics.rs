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

use crate::bitboard::{gen_rook_attacks, gen_bishop_attacks, create_blocker_permutations, mask_without_outline};

#[derive(Copy, Clone)]
pub struct Magic {
    mask: u64,
    number: u64,
    offset: usize
}

pub const EMPTY_MAGIC: Magic = Magic{mask: 0, number: 0, offset: 0};

const ROOK_MAGIC_NUMS: [u64; 64] = [
    0x0080021220854005, 0x0020000800100020, 0x0040080004401000, 0x80200408201e0020, 0x0040020040040001, 0x8040010005a84002, 0x0040010000800040, 0x2600040128420c85, 0x080040002000fe40, 0x0000100400080010, 0x80004008003dc010, 0x0000200200040020, 0x0000202001000200, 0x0000200080010020, 0x0000202000800040, 0x800020002000417d, 0x80c0002000100020, 0x0008001000040010, 0x0004000800070200, 0x0004002002002004, 0x0002002001002002, 0x0001002000802001, 0x0000004000804001, 0x4100002000417e20, 0x0048210010080010, 0x0010020008000c00, 0x0b3d48004002c00e, 0x0000040020020020, 0x0000020020010020, 0x0008400040010002, 0x0000400040008001, 0x80017e0020004020, 0x0040001000200020, 0x807c000800100010, 0x0000400800400410, 0x0000200400200200, 0x0000200100200200, 0x4007e00100200080, 0x0000200080200040, 0x0000802000200040, 0x80dfbf1004000800, 0x406e880201000402, 0x80003dc008004010, 0x0004002002002004, 0x0002002001002002, 0x0001002000802001, 0x0000400080004001, 0x8000016c02009001, 0x0040002000100020, 0x0004000800100010, 0x80020032ffc40010, 0x0000200200040020, 0x0000010020020020, 0x0000200080010020, 0x0000200040008020, 0x800000417e200020, 0x010a000618308116, 0x4001001063800841, 0x1001000822400411, 0x0000100440200802, 0x100ff0000403e402, 0xc80ff8000407fe01, 0x000a000024006201, 0x008a04000c6c4132
];

const ROOK_MAGIC_OFFSETS: [u32; 64] = [
    110157, 46132, 77626, 70611, 48182, 76170, 50230, 114252, 70989, 14673, 37248, 10577, 24914, 16721, 27986, 67735, 55359, 11601, 42034, 23890, 32082, 22866, 17746, 64653, 73088, 36625, 40631, 21842, 20818, 25938, 13649, 59444, 44086, 28889, 37904, 19794, 7505, 16657, 9553, 52275, 63084, 33625, 36224, 5457, 18770, 30034, 31058, 80844, 56375, 12625, 40471, 6481, 4433, 29010, 8529, 61519, 98002, 94276, 102096, 0, 83765, 87242, 90700, 106063
];

const BISHOP_MAGIC_NUMS: [u64; 64] = [
    0x027f80a040800038, 0x807fa01f98001000, 0x8001000fe3fe0400, 0x80007ff810000000, 0x80403bfe80000000, 0x805fe101006c0000, 0x04700fc002008000, 0x40000780807fff80, 0x80007e3effc00ffd, 0x080002004feffbff, 0x104001000fe40100, 0x0074008020042400, 0x2040003c01800000, 0x0500401fff008000, 0x800007e01ffe0080, 0x0c0000080f7fffc0, 0x207e8000c0210002, 0x404060003f2fc802, 0x0001000081010002, 0x0040100101002000, 0x004040008013c400, 0x0050100080440020, 0x047808000fc10002, 0x04a3fd3007e0800c, 0x800080807e3e1ff4, 0x804080403dffc004, 0x0052010200810002, 0x0a48028208020004, 0x00d0018084008400, 0x021fe08080404028, 0x202ff020000f800c, 0x407007f20007c002, 0x20007fff90040080, 0x100ffeffc1005fa0, 0x4017ff0080040020, 0x0040200800810031, 0x0040220021020080, 0x043fe40040802020, 0x801efff80707e020, 0x040087f808000ffd, 0x80001ffeff880400, 0x8007ff8080803fc0, 0x8003ff00ff870100, 0x0070000802010040, 0x0000020040100100, 0x807f801c00200040, 0x10005fffcff817bc, 0x401fd007e80383e6, 0x1007f01f3f008000, 0x2007ffdfa03fc210, 0x80080007ff808000, 0x0804000007ffc040, 0x0000000100202000, 0x04007fffa0c00a00, 0x2000ffc01fbfd840, 0x047f000030240800, 0x401007ff0101004a, 0x807e03e800203f40, 0x4008000007ff8080, 0x104800000007c012, 0x0000000001002020, 0x401000003e20400f, 0x10407e40800f9006, 0x8001bffe3f901003
];

const BISHOP_MAGIC_OFFSETS: [u32; 64] = [
    90, 380, 3445, 1852, 2921, 1882, 1217, 858, 2395, 1713, 2103, 1568, 2681, 1056, 3020, 1116, 1787, 301, 4093, 3839, 4821, 4477, 1790, 223, 775, 1997, 4949, 6485, 6993, 4351, 77, 1780, 2198, 175, 3807, 5973, 5461, 4228, 485, 707, 3349, 2428, 4147, 4670, 3325, 4795, 1253, 2407, 1149, 2147, 1326, 1194, 3135, 675, 157, 565, 2348, 2514, 1389, 636, 3167, 2703, 659, 2973
];

static mut ATTACKS: [u64; 118348] = [0; 118348];

static mut ROOK_MAGICS: [Magic; 64] = [EMPTY_MAGIC; 64];
static mut BISHOP_MAGICS: [Magic; 64] = [EMPTY_MAGIC; 64];


pub fn initialize_magics() {
    initialize_attacks(gen_rook_attacks, &ROOK_MAGIC_NUMS, &ROOK_MAGIC_OFFSETS[..], unsafe {&mut ATTACKS}, unsafe { &mut ROOK_MAGICS }, 12);
    initialize_attacks(gen_bishop_attacks, &BISHOP_MAGIC_NUMS, &BISHOP_MAGIC_OFFSETS[..], unsafe {&mut ATTACKS}, unsafe { &mut BISHOP_MAGICS }, 9);
}

pub fn initialize_attacks(gen_attacks: fn(u64, i32) -> u64, magic_nums: &[u64], magic_offsets: &[u32],
                          target: &mut[u64], magics: &mut[Magic; 64], shift: u32) {

    for pos in 0..64 {
        let move_mask = gen_attacks(0, pos);
        let block_mask = mask_without_outline(move_mask, pos as u32);

        let blocker_count = block_mask.count_ones();

        let mut permutations: Vec<u64> = Vec::with_capacity(1 << blocker_count);
        create_blocker_permutations(&mut permutations, 0, block_mask);

        let magic_num = magic_nums[pos as usize];

        let magic_offset = magic_offsets[pos as usize];

        for &p in permutations.iter() {
            let move_targets = gen_attacks(p, pos);

            let index = ((!p).wrapping_mul(magic_num)) >> (64 - shift);
            target[index as usize + magic_offset as usize] = move_targets;
        }

        let inv_block_mask = !block_mask;
        magics[pos as usize] = Magic{
            mask: inv_block_mask,
            number: magic_num,
            offset: magic_offset as usize
        }
    }
}

#[inline]
pub fn get_bishop_attacks(empty: u64, pos: i32) -> u64 {
    let magic = unsafe { *BISHOP_MAGICS.get_unchecked(pos as usize) };
    unsafe { *ATTACKS.get_unchecked(magic.offset + ((empty | magic.mask).wrapping_mul(magic.number) >> (64 - 9)) as usize) }
}

#[inline]
pub fn get_rook_attacks(empty: u64, pos: i32) -> u64 {
    let magic = unsafe { *ROOK_MAGICS.get_unchecked(pos as usize) };
    unsafe { *ATTACKS.get_unchecked(magic.offset + ((empty | magic.mask).wrapping_mul(magic.number) >> (64 - 12)) as usize) }
}

#[inline]
pub fn get_queen_attacks(empty: u64, pos: i32) -> u64 {
    get_bishop_attacks(empty, pos) | get_rook_attacks(empty, pos)
}
