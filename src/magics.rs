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
use std::cmp::max;

#[derive(Copy, Clone)]
pub struct Magic {
    mask: u64,
    number: u64,
    offset: usize
}

pub const EMPTY_MAGIC: Magic = Magic{mask: 0, number: 0, offset: 0};

const ROOK_MAGIC_NUMS: [u64; 64] = [0xc28001400781a2b2, 0x0020000800100020, 0x3fdfffdff3fbffe8, 0x001000100604fc08, 0x0200040200011008, 0x0200040802000081, 0x000810c021800844, 0x0200003a8110f4f5, 0x0000082104104200, 0x0000100400080010, 0x0000400800400410, 0x0000202004000200, 0x0000202001000200, 0x0000202000800100, 0x0000400080004001, 0x0000200040840020, 0x0040002000100020, 0x0004001000080010, 0x0004000802010008, 0x0004002002002004, 0x0001002020020001, 0x0001002000802001, 0x0000004040008001, 0x000000200040c020, 0x0021440008110408, 0x0002010004081020, 0x0004020008010008, 0x0000040020200200, 0x0000010020200200, 0x0000010020200080, 0x0000400040008001, 0x0000820020004020, 0x0040001000200020, 0x0000080400100010, 0x0004010200080008, 0x0000200200200400, 0x0000200200200100, 0x0000200080200100, 0x0000200080200040, 0x0000802000200040, 0x0010204204000800, 0x0010080102000400, 0x0000044008004010, 0x0000040002002020, 0x0002002001002002, 0x0001002000802001, 0x0000400080004001, 0x00000800a0009002, 0x0040001000200020, 0x0004001000080010, 0x0002010008040008, 0x0000200200040020, 0x0000020020010020, 0x0000200080010020, 0x0000200040008020, 0x0000802000400020, 0x00fff7bc8eefefde, 0x007ffff7ef007fbf, 0x000200201008c006, 0x0000401020080402, 0x0000100204080001, 0x0002001804010082, 0x0003ffffbf7803fe, 0x0000011c01383076];

const BISHOP_MAGIC_NUMS: [u64; 64] = [
    0x0002008020084040, 0x0000208020082000, 0x0000808010020000, 0x0000806004000000, 0x0000440200000000, 0x000021c100800000,
    0x0000208044004000, 0x0000020080410080, 0x0000010041000820, 0x0000004100202002, 0x0000010010200200, 0x0000008060040000,
    0x0000004402000000, 0x00000021c1008000, 0x0000002082008040, 0x0000002010420020, 0x0004000200801010, 0x0002000020404008,
    0x0000400400102040, 0x0000200200801000, 0x0000240080840000, 0x0000080080840080, 0x0000040010808040, 0x0000040010108010,
    0x0001008000408100, 0x0000420000202040, 0x0000804000820080, 0xa202002208008020, 0x05810070c3014000, 0x0000204000804020,
    0x0000200800208010, 0x0000081000084008, 0x0000820080010020, 0x0000208040020010, 0x0001008080004100, 0x24014008a19a0200,
    0x11010304002a0020, 0x0002040008004020, 0x0000108008001040, 0x0000200410002008, 0x0000404200400080, 0x0000208040200040,
    0x0000808080800040, 0x0000008840200040, 0x0000020041000010, 0x0000810200800008, 0x0002008010100004, 0x0000208020080002,
    0x5100402084004202, 0x0000084042002000, 0x0000001010806000, 0x0000000008403000, 0x0000000100202000, 0x0000008101000800,
    0x0000802040404000, 0x0000210040080800, 0x0000201008010080, 0x0000001080804020, 0x0000000008208060, 0x0000000000084030,
    0x0000000001002020, 0x0000000100210010, 0x0000040080204004, 0x0000810041000420
];

static mut ATTACKS: [u64; 110503] = [0; 110503];

static mut ROOK_MAGICS: [Magic; 64] = [EMPTY_MAGIC; 64];
static mut BISHOP_MAGICS: [Magic; 64] = [EMPTY_MAGIC; 64];

const ROOK_MAGIC_OFFSETS: [u32; 64] = [100777, 47683, 67316, 88868, 84804, 80164, 65235, 104873, 45579, 3072, 33801, 21512, 8192, 4096, 30729, 53929, 39434, 24584, 36361, 6144, 0, 19464, 16392, 58078, 49731, 15361, 22536, 14337, 26633, 32777, 10240, 56013, 51881, 17416, 28681, 5120, 23560, 12289, 31753, 43530, 60645, 25608, 35081, 2048, 29705, 7168, 13313, 62731, 41482, 20488, 11264, 27657, 18440, 1024, 9216, 37385, 73739, 76564, 80660, 92586, 92842, 84276, 70112, 96682];
const BISHOP_MAGIC_OFFSETS: [u32; 64] = [59108, 59172, 59204, 89452, 91452, 90724, 71594, 88420, 59236, 59268, 59300, 90154, 91486, 90218, 77838, 77902, 59332, 59364, 59396, 65773, 90780, 66541, 59524, 59556, 63955, 65901, 67317, 69364, 108967, 68595, 65933, 65965, 66669, 66701, 69876, 109479, 109991, 70643, 66733, 67445, 67477, 68723, 89260, 90538, 89898, 90026, 68755, 68787, 77966, 78030, 71498, 91308, 70004, 70036, 70068, 70771, 88484, 91406, 71562, 91349, 70803, 70835, 71438, 89388];

pub fn initialize_magics() {
    initialize_attacks(gen_rook_attacks, &ROOK_MAGIC_NUMS, &ROOK_MAGIC_OFFSETS[..], unsafe {&mut ATTACKS}, unsafe { &mut ROOK_MAGICS }, 12);
    initialize_attacks(gen_bishop_attacks, &BISHOP_MAGIC_NUMS, &BISHOP_MAGIC_OFFSETS[..], unsafe {&mut ATTACKS}, unsafe { &mut BISHOP_MAGICS }, 9);
}

pub fn initialize_attacks(gen_attacks: fn(u64, i32) -> u64, magic_nums: &[u64], magic_offsets: &[u32],
                          target: &mut[u64], magics: &mut[Magic; 64], shift: u32) {

    let mut max_index = 0;
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

            let index = (p.wrapping_mul(magic_num)) >> (64 - shift);
            target[index as usize + magic_offset as usize] = move_targets;
            max_index = max(index as usize + magic_offset as usize, max_index);
        }

        magics[pos as usize] = Magic{
            mask: block_mask,
            number: magic_num,
            offset: magic_offset as usize
        }
    }
    println!("Max index: {}", max_index);
}

#[inline]
pub fn get_bishop_attacks(mut occupied: u64, pos: i32) -> u64 {
    let magic = unsafe { *BISHOP_MAGICS.get_unchecked(pos as usize) };
    occupied &= magic.mask;
    unsafe { *ATTACKS.get_unchecked(magic.offset + (occupied.wrapping_mul(magic.number) >> (64 - 9)) as usize) }
}

#[inline]
pub fn get_rook_attacks(mut occupied: u64, pos: i32) -> u64 {
    let magic = unsafe { *ROOK_MAGICS.get_unchecked(pos as usize) };
    occupied &= magic.mask;
    unsafe { *ATTACKS.get_unchecked(magic.offset + (occupied.wrapping_mul(magic.number) >> (64 - 12)) as usize) }
}

#[inline(always)]
pub fn get_queen_attacks(occupied: u64, pos: i32) -> u64 {
    get_bishop_attacks(occupied, pos) | get_rook_attacks(occupied, pos)
}
