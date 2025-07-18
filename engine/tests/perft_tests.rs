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

extern crate velvet;

use velvet::fen::create_from_fen;
use velvet::history_heuristics::HistoryHeuristics;
use velvet::init::init;
use velvet::perft::perft;
use velvet::search_context::SearchContext;

#[test]
fn test_perft_startpos() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    assert_eq!(1, perft_for_fen(fen, 0));
    assert_eq!(20, perft_for_fen(fen, 1));
    assert_eq!(400, perft_for_fen(fen, 2));
    assert_eq!(8902, perft_for_fen(fen, 3));
    assert_eq!(197281, perft_for_fen(fen, 4));
}

#[test]
fn test_perft_testpos2() {
    let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

    assert_eq!(48, perft_for_fen(fen, 1));
    assert_eq!(2039, perft_for_fen(fen, 2));
    assert_eq!(97862, perft_for_fen(fen, 3));
}

#[test]
fn test_perft_testpos3() {
    let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

    assert_eq!(14, perft_for_fen(fen, 1));
    assert_eq!(191, perft_for_fen(fen, 2));
    assert_eq!(2812, perft_for_fen(fen, 3));
    assert_eq!(43238, perft_for_fen(fen, 4));
    assert_eq!(674624, perft_for_fen(fen, 5));
}

#[test]
fn test_perft_testpos4() {
    let fen = "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1";

    assert_eq!(6, perft_for_fen(fen, 1));
    assert_eq!(264, perft_for_fen(fen, 2));
    assert_eq!(9467, perft_for_fen(fen, 3));
    assert_eq!(422333, perft_for_fen(fen, 4));
}

#[test]
fn test_perft_testpos5() {
    let fen = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";

    assert_eq!(44, perft_for_fen(fen, 1));
    assert_eq!(1486, perft_for_fen(fen, 2));
    assert_eq!(62379, perft_for_fen(fen, 3));
    assert_eq!(2103487, perft_for_fen(fen, 4));
}

#[test]
fn test_perft_testpos6() {
    let fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

    assert_eq!(46, perft_for_fen(fen, 1));
    assert_eq!(2079, perft_for_fen(fen, 2));
    assert_eq!(89890, perft_for_fen(fen, 3));
}

#[test]
fn test_perft_chess960() {
    assert_eq!(6417013, perft_for_fen("b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9", 5));
    assert_eq!(9183776, perft_for_fen("qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9", 5));
    assert_eq!(6718715, perft_for_fen("b1qnrrkb/ppp1pp1p/n2p1Pp1/8/8/P7/1PPPP1PP/BNQNRKRB w GE - 0 9", 5));
    assert_eq!(7697880, perft_for_fen("nq1bbrkr/pp2nppp/2pp4/4p3/1PP1P3/1B6/P2P1PPP/NQN1BRKR w HFhf - 2 9", 5));
    assert_eq!(22760, perft_for_fen("rk2r1bn/pqbppppp/1pp2n2/8/5P2/3P1N2/PPPQPRPP/RK1B2BN b a - 8 11", 3));
}

fn perft_for_fen(fen: &str, depth: i32) -> u64 {
    init();
    let mut ctx = SearchContext::default();
    let mut board = create_from_fen(fen);
    let hh = HistoryHeuristics::default();
    perft(&mut ctx, &hh, &mut board, depth)
}
