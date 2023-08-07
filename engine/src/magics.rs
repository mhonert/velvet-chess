/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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

use crate::bitboard::{create_blocker_permutations, gen_bishop_attacks, gen_rook_attacks, mask_without_outline};

pub fn get_bishop_attacks(empty_bb: u64, pos: i32) -> u64 {
    let magic = unsafe { MAGICS.get_unchecked(pos as usize) };
    unsafe {
        *ATTACKS.get_unchecked(magic.b_offset + ((empty_bb | magic.b_mask).wrapping_mul(magic.b_number) >> (64 - 9)) as usize)
    }
}

#[inline]
pub fn get_rook_attacks(empty_bb: u64, pos: i32) -> u64 {
    let magic = unsafe { MAGICS.get_unchecked(pos as usize) };
    unsafe {
        *ATTACKS.get_unchecked(magic.r_offset + ((empty_bb | magic.r_mask).wrapping_mul(magic.r_number) >> (64 - 12)) as usize)
    }
}

#[inline]
pub fn get_queen_attacks(empty_bb: u64, pos: i32) -> u64 {
    let magic = unsafe { MAGICS.get_unchecked(pos as usize) };
    unsafe {
        *ATTACKS.get_unchecked(magic.b_offset + ((empty_bb | magic.b_mask).wrapping_mul(magic.b_number) >> (64 - 9)) as usize)
            | *ATTACKS.get_unchecked(magic.r_offset + ((empty_bb | magic.r_mask).wrapping_mul(magic.r_number) >> (64 - 12)) as usize)
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

#[rustfmt::skip]
static ROOK_MAGIC_NUMS: [u64; 64] = [
    0x0380051082e14004, 0x0018000c00060018, 0x020004200a008010, 0x0040080004004002,
    0x0040020040040001, 0x0020008020010202, 0x0018007900002040, 0x0600040253002582,
    0x0002001042008020, 0x0000100008040010, 0x0000080401020008, 0x0000200400200200,
    0x0000200200010020, 0x0000200100008020, 0x087fe80040800020, 0x0002000041008024,
    0x34cf0018010c0004, 0x0008001000040010, 0x0001000804020008, 0x0004002002002004,
    0x0002001001008010, 0xc001002000802001, 0x0000004040008001, 0x0000802000400020,
    0x0040200010080010, 0x0000080010040010, 0x0004010008020008, 0x0000020020040020,
    0x0000020020010020, 0x0000010020200080, 0x0000400040008001, 0x200fc00040000080,
    0x0040001000200020, 0x0000080400100010, 0x040401ff00080008, 0x0000200200200400,
    0x0000200200200100, 0x0000200100200080, 0x0000200080200040, 0x0000802000200040,
    0x31cf00010c001800, 0x0010080201000400, 0x09003dc008004010, 0x0002002004002002,
    0x0001002002002001, 0x0008400100004002, 0x0000400080004001, 0x0008001058005802,
    0x0019000808370010, 0x000c00030086000c, 0x023f880000408028, 0x000a000125032e00,
    0x210e7000708f8010, 0x0000200080010020, 0x200f980004080068, 0x0008001262300030,
    0x000e000242248412, 0x0000401009002202, 0x0000080411002001, 0x024e0000830a4806,
    0x000500080001c401, 0x0002000084011002, 0x0001000082000041, 0x020e0007684824e2
];

#[rustfmt::skip]
static ROOK_MAGIC_OFFSETS: [u32; 64] = [
    0, 27478, 23381, 13144, 65447, 11094, 2022, 63395, 38375, 44182, 49063, 25430, 28501,
    4939, 78782, 73777, 40019, 77746, 19285, 15188, 3045, 87095, 17236, 46231, 86077, 5963,
    53160, 14164, 50087, 74801, 43031, 69681, 42011, 55208, 54184, 81331, 52135, 67633, 9036,
    8012, 80319, 24407, 66609, 48038, 75825, 18260, 51111, 92609, 83000, 47014, 76727, 90184,
    91061, 16212, 4011, 30199, 59289, 38391, 34279, 59308, 34295, 73761, 23365, 90158
];

#[rustfmt::skip]
static BISHOP_MAGIC_NUMS: [u64; 64] = [
    0x407f40a0106003d2, 0x087fdfdfdfdc0001, 0x107f400818002000, 0x02007ff810000000,
    0x14603bfe54000000, 0x10401efeff700000, 0x0800101f3fff8000, 0x100020101e3fff80,
    0x1000023fdfbfeffe, 0x8040003f3fe7e801, 0x007f008048100020, 0x0482007fe0040000,
    0x1040003c01800000, 0x0480401fff008000, 0x100000101effff80, 0x080000082fbfffc0,
    0x407f007f40a0a004, 0xfd4060003f3fd002, 0x20018000c00f3fff, 0x0040100101002000,
    0x0060200100080000, 0x104010001e001000, 0x08000807efe0bffe, 0x0400040007e05fff,
    0x0000808000420080, 0x0000420000401040, 0x140ffe0204002008, 0x0841004004040001,
    0x0144040000410020, 0x20002038001c7ff0, 0x0840300c07efa008, 0x0840080e0007c004,
    0x207f4000607e7fa0, 0x0070010030010040, 0x0070008180030018, 0x046020031b880080,
    0x0001010400020020, 0x0000404040008010, 0x08604007c1002010, 0x204007c40303f010,
    0x00704002004800c0, 0x087fc00040a10040, 0x2078003e81010080, 0x0070000080fc4080,
    0x0e40003c80180030, 0x0c7a0004000c0018, 0x10007fffc7dc8008, 0x107f50000bfc080a,
    0x00f8100080ffa000, 0x00fc2000203fc200, 0x07fc40000f808000, 0x1001000007ffc060,
    0x087a007f001fe026, 0x02007fffa0a00a00, 0xfe807fffbfe7dc20, 0x04011fffd8080204,
    0x107fd800203e8080, 0x05ff00000fa04024, 0x087c4000000f8080, 0x080090000007ffc1,
    0x0830000001001fe2, 0x104000003e80300c, 0xfc01fffee0200812, 0x207f8000304013d9
];

#[rustfmt::skip]
static BISHOP_MAGIC_OFFSETS: [u32; 64] = [
    98493, 85776, 79931, 5024, 98308, 98894, 4349, 85759, 82124, 99300, 4890, 81457, 99198,
    4483, 5260, 99231, 3986, 5988, 79905, 98842, 98518, 98684, 4767, 4222, 4613, 3616, 46742,
    96807, 29526, 99343, 3739, 99364, 5416, 81585, 96295, 97398, 98132, 97525, 81970, 81883,
    99118, 99435, 96903, 97620, 98248, 99018, 99104, 99159, 99041, 79941, 3998, 93866, 82230,
    4634, 86288, 99385, 98561, 98906, 98715, 94602, 81722, 99426, 82578, 98388
];

static mut ATTACKS: [u64; 99947] = [0; 99947];

static mut MAGICS: [Magic; 64] = [EMPTY_MAGIC; 64];

pub fn initialize_attack_tables() {
    for pos in 0..64 {
        // Rook
        let r_move_mask = gen_rook_attacks(0, pos);
        let r_block_mask = mask_without_outline(r_move_mask, pos as u32);

        let r_blocker_count = r_block_mask.count_ones();

        let mut r_permutations: Vec<u64> = Vec::with_capacity(1 << r_blocker_count);
        create_blocker_permutations(&mut r_permutations, 0, r_block_mask);

        let r_magic_num = ROOK_MAGIC_NUMS[pos as usize];

        let r_magic_offset = ROOK_MAGIC_OFFSETS[pos as usize];

        for &p in r_permutations.iter() {
            let move_targets = gen_rook_attacks(p, pos);

            let index = ((!p).wrapping_mul(r_magic_num)) >> (64 - 12);
            unsafe { ATTACKS[index as usize + r_magic_offset as usize] = move_targets }
        }

        let r_inv_block_mask = !r_block_mask;

        // Bishop
        let b_move_mask = gen_bishop_attacks(0, pos);
        let b_block_mask = mask_without_outline(b_move_mask, pos as u32);

        let b_blocker_count = b_block_mask.count_ones();

        let mut b_permutations: Vec<u64> = Vec::with_capacity(1 << b_blocker_count);
        create_blocker_permutations(&mut b_permutations, 0, b_block_mask);

        let b_magic_num = BISHOP_MAGIC_NUMS[pos as usize];

        let b_magic_offset = BISHOP_MAGIC_OFFSETS[pos as usize];

        for &p in b_permutations.iter() {
            let move_targets = gen_bishop_attacks(p, pos);

            let index = ((!p).wrapping_mul(b_magic_num)) >> (64 - 9);
            unsafe { ATTACKS[index as usize + b_magic_offset as usize] = move_targets }
        }

        let b_inv_block_mask = !b_block_mask;

        unsafe { MAGICS[pos as usize] = Magic {
            r_mask: r_inv_block_mask, r_number: r_magic_num, r_offset: r_magic_offset as usize,
            b_mask: b_inv_block_mask, b_number: b_magic_num, b_offset: b_magic_offset as usize,
        } };
    }
}
