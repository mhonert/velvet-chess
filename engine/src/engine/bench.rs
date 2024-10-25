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

use std::time::Instant;
use crate::engine::Engine;
use crate::fen::START_POS;
use crate::time_management::SearchLimits;

static BENCH_FENS: [(&str, i32); 34] = [
    ("8/4R3/8/p7/8/6Rk/8/6K1 b - - 0 59", 14),
    ("8/8/8/1k3Q1K/1b1q1P2/8/7P/8 b - - 0 62", 10),
    ("8/8/1P6/6R1/4k2K/1r5P/8/8 b - - 0 67", 15),
    ("6kb/8/4K1N1/6P1/5P2/4B3/8/8 b - - 0 65", 17),
    ("8/3nN3/3Pk3/6K1/8/8/8/8 b - - 0 69", 16),
    ("8/8/4n3/7k/3BB2p/3P1KpP/8/8 b - - 0 45", 14),
    ("2r4k/4q3/7Q/P7/4B1P1/3R3K/8/8 b - - 0 61", 18),
    ("8/p5k1/5p2/6p1/2bp4/q2n2Q1/2B1R3/1K6 w - - 0 57", 17),
    ("8/2R5/7p/5p1k/BK4b1/2p1r3/8/8 w - - 0 48", 13),
    ("3Q4/1B2Rpk1/6n1/8/4N2p/8/2K5/6q1 w - - 0 50", 10),
    ("8/8/2k5/R5K1/5N2/8/P4n1r/8 b - - 0 60", 11),
    ("8/4nk2/b1n5/1B2P3/1P6/2N4P/6P1/6K1 b - - 0 39", 13),
    ("k3Q3/pp6/2b4P/6p1/5n2/5PP1/1P1qP3/R4K1R b - - 0 24", 13),
    ("r4rk1/pp2p2p/2n1pnp1/8/8/2P5/P1P1BPPP/2BR1RK1 b - - 1 18", 11),
    ("4r2k/2rb1pb1/3p1pp1/5P1p/3BP2P/n5P1/Q5BK/8 w - - 0 44", 12),
    ("8/1N3k2/P1r3pp/1n2K3/7P/6Pb/8/8 w - - 0 56", 13),
    ("3k4/2q5/1bB2r2/7Q/3P4/P6P/1P6/1KR5 b - - 0 58", 10),
    ("8/5p2/3p1q1k/pQ6/5r2/2N3R1/2K3B1/5n2 w - - 0 53", 11),
    ("3b3k/n4q2/P4N2/1r1P4/4BP2/7Q/4PP2/4BK2 b - - 0 44", 15),
    ("6k1/1p1n4/8/1p2N1p1/1B1K3p/8/r5b1/1R6 w - - 0 40", 12),
    ("4r1k1/p2q1ppp/2p2n2/4p3/2Pr4/1P2Q1P1/P3RPBP/R5K1 b - - 0 19", 12),
    ("r7/1p3kp1/2p1b2p/p2p1q2/3P3P/2NQ2P1/PPB2PK1/8 w - - 0 29", 13),
    ("4r1k1/5p1p/3p2p1/p3p3/Q1rPP3/P1P4P/3b1PP1/BR2R1K1 b - - 0 30", 13),
    ("8/1R4pk/q3p1np/3pp2n/4P1PP/3P1N2/2QN1PK1/8 b - - 0 38", 11),
    ("3rr1k1/p5p1/2p1p2p/N1P1qp2/1Pp1nP2/8/P5PP/R3R1K1 w - - 0 23", 12),
    ("5k2/3pRb2/2pN1Np1/p5n1/P5B1/1P6/2PP2Pb/2K5 b - - 0 26", 15),
    ("2k3r1/4bR2/N2q4/2B5/4P3/3P1Qn1/PKP5/8 b - - 0 48", 12),
    ("3r1rk1/p1p1q1pp/Q4p2/3P4/1b3P2/3BB3/PP4PP/R2K2NR b - - 0 17", 11),
    ("1k2r3/8/1p3p2/2R2qp1/p2p1Pn1/P2P1B1p/1P1QN2P/6K1 b - - 0 29", 11),
    ("5k2/5p2/1p1n4/3P4/3P3b/3Q3N/4PB1K/rBq3r1 w - - 0 39", 11),
    ("r1b2rk1/ppp1pp1p/1n4p1/4P3/1nP5/2N2N1P/PP1qBPP1/R3K2R w KQ - 0 13", 11),
    ("rnbqkbnr/pppp2pp/4pp2/8/6P1/8/PPPPPP1P/RNBQKBNR w KQkq - 0 3", 12),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 10),
    (START_POS, 12),
];

impl Engine {
    pub fn bench(&mut self) {
        self.check_readiness();
        self.search.clear_tt();
        self.search.hh.clear();

        let mut limit = SearchLimits::default();

        let mut nodes = 0;
        let start = Instant::now();
        for &(fen, depth) in BENCH_FENS.iter() {
            limit.set_depth_limit(depth);
            self.set_position(String::from(fen), Vec::with_capacity(0));
            self.go(limit, false, None);
            nodes += self.search.node_count();
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let nps = nodes * 1000 / duration_ms;

        println!("\ninfo string bench total time    : {}ms", duration_ms);
        println!("info string bench nodes         : {}", nodes);
        println!("info string bench NPS           : {}", nps);
        
        self.tt_clean = false;
    }
}