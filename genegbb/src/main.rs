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

use std::borrow::Borrow;
use std::fs::File;
use std::io::{BufWriter, Error, Write};
use std::num::NonZeroU32;
use std::ops::Not;
use std::process::exit;
use std::str::FromStr;
use std::sync::{Arc};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration};
use byteorder::{LittleEndian, WriteBytesExt};

use clap::{App};
use crossbeam_channel::{Sender, Receiver, SendError};
use itertools::Itertools;
use shakmaty::{Bitboard, Board, ByColor, CastlingMode, Chess, Color, FromSetup, Material, Outcome, Piece, Position, PositionError, PositionErrorKinds, RemainingChecks, Role, Setup, Square};
use shakmaty::fen::Fen;
use shakmaty_syzygy::{Tablebase};
use velvet::colors::{BLACK, WHITE};
use velvet::compression::{BitWriter, CodeBook};
use velvet::egbb::BitBases;
use velvet::fen::{create_from_fen, read_fen, START_POS};

#[derive(Clone)]
enum Message {
    WorkItem(String, Vec<SetupFn>, bool),
    Quit
}

fn main() {
    let matches = App::new("gensets")
        .version("1.0.0")
        .about("Generates labeled training sets of chess positions from self-play games")
        .args_from_usage("
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -t  --table-base-path=<FILE>           'Sets the Syzygy tablebase path'
             -v  --verify                           'Verifies the embedded bitbases'")
        .get_matches();

    let tb_path = matches.value_of("table-base-path").unwrap();

    let concurrency = match matches.value_of("concurrency") {
        Some(v) => usize::from_str(v).expect("Concurrency must be an integer >= 1"),
        None => {
            eprintln!("Missing -c (concurrency) option");
            exit(1);
        }
    };

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let (tx, rx) = crossbeam_channel::bounded(concurrency);

    println!("Starting worker threads ...");
    let stopped = Arc::new(AtomicUsize::default());
    start_workers(rx, &stopped, tb_path, concurrency);

    let verify = matches.occurrences_of("verify") > 0;
    gen_generic(&tx, verify).expect("Generation failed");

    for _ in 0..concurrency {
        tx.send(Message::Quit).expect("Sending quit message failed");
    }

    while stopped.load(Ordering::Acquire) < concurrency  {
        thread::sleep(Duration::from_millis(100));
    }

    println!("End");
}

#[derive(Clone)]
struct TBSetup {
    board: Board,
    stm: Color,
}

impl TBSetup {
    pub fn new(stm: Color) -> TBSetup {
        TBSetup{board: Board::empty(), stm}
    }

    pub fn set(&mut self, pos: u32, mut color: Color, role: Role) {
        if self.stm == Color::Black {
            color = color.not();
        }
        let sq = Square::new(pos).flip_vertical();
        self.board.set_piece_at(sq, Piece{ color, role });
    }

    pub fn count(&self) -> usize {
        self.board.occupied().count()
    }
}

impl Setup for TBSetup {
    fn board(&self) -> &Board {
        &self.board
    }

    fn promoted(&self) -> Bitboard {
        Bitboard(0)
    }

    fn pockets(&self) -> Option<&Material> {
        None
    }

    fn turn(&self) -> Color {
       self.stm
    }

    fn castling_rights(&self) -> Bitboard {
        Bitboard::EMPTY
    }

    fn ep_square(&self) -> Option<Square> {
        None
    }

    fn remaining_checks(&self) -> Option<&ByColor<RemainingChecks>> {
        None
    }

    fn halfmoves(&self) -> u32 {
        0
    }

    fn fullmoves(&self) -> NonZeroU32 {
        NonZeroU32::new(1).unwrap()
    }
}

struct Pack {
    results: Vec<Vec<PackEntry>>,
    draws: usize,
    wins: usize,
    losses: usize,
    idx_map: [[u16; 64]; 64],
    max_count: u8,
    count_shift: u16,
    max_size: usize,
    bitbases: Arc<BitBases>,
    board: velvet::board::Board,
}

#[derive(Debug)]
struct PackEntry {
    prev_king_index: u16,
    start_king_index: u16,
    count: u8,
    result: i8,
}

impl Pack {
    pub fn new(piece_count: usize, can_rotate: bool) -> Self {
        assert!(piece_count > 2);
        let result_count = 64usize.pow(piece_count as u32 - 2);
        let mut results = Vec::with_capacity(result_count);
        for _ in 0..result_count {
            results.push(Vec::new());
        }

        let mut idx_map = [[0; 64]; 64];
        let mut idx = 0;

        for wk in 0..64 {
            if wk & 7 > 3 {
                continue;
            }
            if can_rotate && wk / 8 > 3 {
                continue;
            }
            for bk in 0..64 {
                let row_distance = (wk / 8).max(bk / 8) - (wk / 8).min(bk / 8);
                let col_distance = (wk & 7).max(bk & 7) - (wk & 7).min(bk & 7);
                let distance = row_distance.max(col_distance);

                if distance <= 1 {
                    continue;
                }

                idx_map[wk][bk] = idx;
                idx += 1;
            }
        }

        let (max_count, count_shift) = if can_rotate {
            (64, 10)
        } else {
            (32, 11)
        };

        Pack{results, draws: 0, wins: 0, losses: 0, idx_map, max_count, count_shift, max_size: 0,
            bitbases: BitBases::new(),
            board: create_from_fen(START_POS),
        }
    }

    pub fn update(&mut self, index: u16, own_king: u16, opp_king: u16, result: i8) {
        self.add_result(index, self.idx_map[own_king as usize][opp_king as usize], result);
    }

    fn add_result(&mut self, index: u16, king_index: u16, result: i8) {
        assert!(king_index < (1 << self.count_shift));
        if result == 0 {
            self.draws += 1;
        } else if result == 1 {
            self.wins += 1;
        } else {
            self.losses += 1;
        }

        if let Some(entry) = self.results[index as usize].last_mut() {
            if entry.result == result && entry.prev_king_index + 1 == king_index && entry.count < self.max_count {
                entry.prev_king_index = king_index;
                entry.count += 1;
                return;
            }
        }

        self.results[index as usize].push(PackEntry{
            prev_king_index: king_index,
            start_king_index: king_index,
            count: 1,
            result
        });
    }

    pub fn print_stats(&self, name: String) {
        println!("{}: max_count: {}: wins: {}, draws: {}, losses: {}", name, self.max_count, self.wins, self.draws, self.losses);
    }

    pub fn write_to_vec(&mut self, outputs: &mut Vec<i16>) {

        let mut max_outputs = 0;

        let mut idx = 0;
        for entries in self.results.iter() {
            let mut max_group_size = 0;
            let mut max_group_result = 0;

            for (result, group) in &entries.iter().group_by(|entry| entry.result) {
                let count = group.count();
                if count > max_group_size {
                    max_group_size = count;
                    max_group_result = result;
                }
            }

            let mut bucket1 = Vec::new();
            let mut bucket2 = Vec::new();

            for entry in entries.iter().filter(|e| e.result != max_group_result) {
                let bucket = if max_group_result != 0 {
                    if entry.result == 0 { 0 } else { 1 }
                } else if entry.result == -1 { 0 } else { 1 };

                let packed = ((entry.count as u16 - 1) << self.count_shift | entry.start_king_index) as i16;

                if bucket == 0 {
                    bucket1.push(packed);
                } else {
                    bucket2.push(packed);
                }

            }

            max_outputs = max_outputs.max(bucket1.len());
            max_outputs = max_outputs.max(bucket2.len());

            outputs.push(((max_group_result as i16 + 1) << 13) as i16 | bucket1.len() as i16);
            outputs.push(bucket2.len() as i16);

            outputs.append(&mut bucket1);
            outputs.append(&mut bucket2);

            idx += 1;
        }

        self.max_size = max_outputs;

    }

    pub fn write(&mut self, file: &str) -> Result<(), Error> {
        let file = File::create(file).expect("Could not create output file");
        let mut writer = BufWriter::new(file);

        let mut output = Vec::with_capacity(4096);
        self.write_to_vec(&mut output);

        // let codes = CodeBook::from_values(&output);

        // codes.write(&mut writer)?;

        // let mut bit_writer = BitWriter::new();
        for v in output {
            // let code = codes.get_code(v);
            // bit_writer.write(&mut writer, code)?;
            writer.write_i16::<LittleEndian>(v).unwrap();
        }

        // bit_writer.flush(&mut writer)?;
        writer.flush()?;

        Ok(())
    }
}

struct Verify {
    bitbases: Arc<BitBases>,
    board: velvet::board::Board,
    correct: [u32; 3],
    wrong: [u32; 3],
}

impl Verify {
    pub fn new() -> Self {
        Verify{
            bitbases: BitBases::new(),
            board: create_from_fen(START_POS),
            correct: [0; 3],
            wrong: [0; 3],
        }
    }

    pub fn verify(&mut self, name: &str, stm: &str, own_king: u16, opp_king: u16, pieces: Vec<u16>, want_result: i8, fen: String) {
        read_fen(&mut self.board, &fen).expect("Could not parse FEN");

        let piece_count = self.board.get_occupancy_bitboard().count_ones();

        if let Some(got_result) = self.bitbases.probe(self.board.active_player(), piece_count as usize, self.board.king_pos(WHITE) as u32, self.board.king_pos(BLACK) as u32, &self.board.bitboards,
        self.board.state.castling != 0, self.board.state.en_passant != 0) {
            if got_result == want_result {
                self.correct[(want_result + 1) as usize] += 1;
            } else {
                self.wrong[(want_result + 1) as usize] += 1;
                println!("{} - {}: {}, kr={}, kc={}, ownk={}, oppk={}, pieces={:?}: want={} vs. got={}", name, stm, fen, own_king & 7, own_king / 8, own_king, opp_king, pieces, want_result, got_result);
                // self.bitbases.probe(self.board.active_player(), piece_count as usize, self.board.king_pos(WHITE), self.board.king_pos(BLACK), &self.board.bitboards, true
                //                     self.board.state.castling != 0, self.board.state.en_passant != 0) {
            }
        }
    }


    pub fn print_stats(&self, stm: &str, name: &str) {
        if self.wrong.iter().sum::<u32>() > 0 {
            print!("{} - {}: Correct: ", name, stm);
            for i in 0..3 {
                print!("{:7} | ", self.correct[i]);
            }
            println!();

            print!("{} - {}: Wrong  : ", name, stm);
            for i in 0..3 {
                print!("{:7} | ", self.wrong[i]);
            }
            println!();
        } else if self.correct.iter().sum::<u32>() == 0 {
            println!("{} - {}: No entries found?!", name, stm);
        } else {
            println!("{} - {}: [OK]", name, stm);
        }
    }
}

fn gen_generic(tx: &Sender<Message>, verify: bool) -> Result<(), SendError<Message>> {
    if verify {
        println!("Verifying embedded bit bases ...");
    } else {
        println!("Generating new bit bases ...");
    }

    // write count before each block

    // 3-men
    // tx.send(Message::WorkItem(String::from("kPk"), vec![white_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kkP"), vec![black_pawn], verify))?;
    //
    // // tx.send(Message::WorkItem(String::from("kNk"), vec![white_knight]))?; -> Draw
    // // tx.send(Message::WorkItem(String::from("kkN"), vec![black_knight]))?; -> Draw
    // // tx.send(Message::WorkItem(String::from("kBk"), vec![white_bishop]))?; -> Draw
    // // tx.send(Message::WorkItem(String::from("kkB"), vec![black_bishop]))?; -> Draw
    //
    // tx.send(Message::WorkItem(String::from("kRk"), vec![white_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kkR"), vec![black_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kQk"), vec![white_queen], verify))?;
    // tx.send(Message::WorkItem(String::from("kkQ"), vec![black_queen], verify))?;
    //
    // // //
    // // // // 4-men
    // tx.send(Message::WorkItem(String::from("kPPk"), vec![white_pawn, white_pawn], verify))?;
    //
    // tx.send(Message::WorkItem(String::from("kPkP"), vec![white_pawn, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kNkP"), vec![white_knight, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kPkN"), vec![white_pawn, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kBkP"), vec![white_bishop, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kPkB"), vec![white_pawn, black_bishop], verify))?;
    //
    // tx.send(Message::WorkItem(String::from("kRkP"), vec![white_rook, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kPkR"), vec![white_pawn, black_rook], verify))?;
    // //
    // tx.send(Message::WorkItem(String::from("kQkP"), vec![white_queen, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kPkQ"), vec![white_pawn, black_queen], verify))?;
    //
    // // tx.send(Message::WorkItem(String::from("kPBk"), vec![white_pawn, white_bishop], verify))?;
    // // tx.send(Message::WorkItem(String::from("kPNk"), vec![white_pawn, white_knight], verify))?;
    // // tx.send(Message::WorkItem(String::from("kNBk"), vec![white_knight, white_bishop], verify))?;
    // //
    // //
    // tx.send(Message::WorkItem(String::from("kRkN"), vec![white_rook, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kRkB"), vec![white_rook, black_bishop], verify))?;
    // tx.send(Message::WorkItem(String::from("kRkR"), vec![white_rook, black_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kQkN"), vec![white_queen, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kQkB"), vec![white_queen, black_bishop], verify))?;
    // //
    // tx.send(Message::WorkItem(String::from("kQkR"), vec![white_queen, black_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kRkQ"), vec![white_rook, black_queen], verify))?;
    //
    // tx.send(Message::WorkItem(String::from("kNkN"), vec![white_knight, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kBkN"), vec![white_bishop, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kNkB"), vec![white_knight, black_bishop], verify))?;
    // tx.send(Message::WorkItem(String::from("kBkB"), vec![white_bishop, black_bishop], verify))?;
    //
    // //
    // tx.send(Message::WorkItem(String::from("kQkQ"), vec![white_queen, black_queen], verify))?;
    //
    // tx.send(Message::WorkItem(String::from("kkPP"), vec![black_pawn, black_pawn], verify))?;
    // tx.send(Message::WorkItem(String::from("kPkN"), vec![white_pawn, black_knight], verify))?;
    // tx.send(Message::WorkItem(String::from("kNkR"), vec![white_knight, black_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kBkR"), vec![white_bishop, black_rook], verify))?;
    // tx.send(Message::WorkItem(String::from("kNkQ"), vec![white_knight, black_queen], verify))?;
    // tx.send(Message::WorkItem(String::from("kBkQ"), vec![white_bishop, black_queen], verify))?;

    tx.send(Message::WorkItem(String::from("kkPN"), vec![black_pawn, black_knight], verify))?;
    tx.send(Message::WorkItem(String::from("kkPB"), vec![black_pawn, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkPR"), vec![black_pawn, black_knight], verify))?;
    tx.send(Message::WorkItem(String::from("kkPQ"), vec![black_pawn, black_knight], verify))?;
    tx.send(Message::WorkItem(String::from("kkNB"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkNR"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkNQ"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkNN"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkBR"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkBQ"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkBB"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkRQ"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkRR"), vec![black_knight, black_bishop], verify))?;
    tx.send(Message::WorkItem(String::from("kkQQ"), vec![black_knight, black_bishop], verify))?;

    // let file = File::create("./bitbases.bin").expect("Could not create output file");
    // let mut writer = BufWriter::new(file);
    //
    // let codes = CodeBook::from_values(&outputs);
    //
    // codes.write(&mut writer).expect("Could not write code book");
    //
    // let mut bit_writer = BitWriter::new();
    // for v in outputs {
    //     let code = codes.get_code(v);
    //     bit_writer.write(&mut writer, code).expect("Could not write values");
    // }
    //
    // bit_writer.flush(&mut writer).expect("Could not flush");
    Result::Ok(())
}

fn start_workers(rx: Receiver<Message>, stopped: &Arc<AtomicUsize>, tb_path: &str, concurrency: usize) {
    for _ in 0..concurrency {
        let tb_path = String::from(tb_path);
        let stopped2 = stopped.clone();


        let rx = rx.clone();
        thread::spawn(move || {
            let mut tb = Tablebase::new();
            tb.add_directory(tb_path).expect("Could not add tablebase path");

            loop {
                match rx.recv() {
                    Ok(msg) => {
                        match msg {
                            Message::WorkItem(name, setup_fns, verify) => {
                                if verify {
                                    verify_bitbase(&tb, name, setup_fns)

                                } else {
                                    gen_bitbase(&tb, name, setup_fns)
                                }
                            },
                            Message::Quit => {
                                stopped2.fetch_add(1, Ordering::Release);
                                break;
                            }
                        };
                    }

                    Err(e) => {
                        panic!("Thread communication failed: {}", e);
                    }
                }
            }
        });
    }
}

fn gen_bitbase(tb: &Tablebase<Chess>, name: String, setup_fns: Vec<SetupFn>) {
    let can_rotate = !name.contains('P');

    let piece_count = setup_fns.len() + 2;
    let mut pack = Pack::new(piece_count, can_rotate);

    create_positions(tb, can_rotate, &setup_fns, &mut |idx, own_king, opp_king, result| {
        pack.update(idx, own_king, opp_king, result);
    });

    pack.write(&format!("./{}.ebb", name)).expect("Could not write bitbase file");
    // pack.write_to_vec(outputs);

    pack.print_stats(name);

}

fn verify_bitbase(tb: &Tablebase<Chess>, name: String, setup_fns: Vec<SetupFn>) {
    let mut verify = Verify::new();

    verify_positions(tb, Color::White, &setup_fns, &mut |own_king, opp_king, pieces, result, fen| {
        verify.verify(name.as_str(), "White to move", own_king, opp_king, pieces, result, fen);
    });
    verify.print_stats("White to move", name.as_str());

    verify_positions(tb, Color::Black, &setup_fns, &mut |idx, own_king, opp_king, result, fen| {
        verify.verify(name.as_str(), "Black to move", idx, own_king, opp_king, result, fen);
    });
    verify.print_stats("Black to move", name.as_str());
}

fn white_pawn(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::White, Role::Pawn)
}

fn black_pawn(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::Black, Role::Pawn)
}

fn white_knight(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::White, Role::Knight)
}

fn black_knight(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::Black, Role::Knight)
}

fn white_bishop(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::White, Role::Bishop)
}

fn black_bishop(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::Black, Role::Bishop)
}

fn white_rook(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::White, Role::Rook)
}

fn black_rook(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::Black, Role::Rook)
}

fn white_queen(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::White, Role::Queen)
}

fn black_queen(setup: &mut TBSetup, pos: u32) {
    setup.set(pos, Color::Black, Role::Queen)
}

type SetupFn = fn(&mut TBSetup, u32);

fn create_positions(tb: &Tablebase<Chess>, can_rotate: bool, setup_fns: &[SetupFn], handle_fn: &mut impl FnMut(u16, u16, u16, i8)) {
    let piece_count = 2 + setup_fns.len();

    for wk in 0..64 {
        if wk & 7 > 3 {
            continue;
        }
        if can_rotate && (wk / 8 > 3) {
            continue;
        }
        for bk in 0..64 {
            let row_distance = (wk / 8).max(bk / 8) - (wk / 8).min(bk / 8);
            let col_distance = (wk & 7).max(bk & 7) - (wk & 7).min(bk & 7);
            let distance = row_distance.max(col_distance);

            if distance <= 1 {
                continue;
            }

            let mut setup = TBSetup::new(Color::White);
            setup.set(wk, Color::White, Role::King);
            setup.set(bk, Color::Black, Role::King);

            create_positions_rec(tb, piece_count, wk as u16, bk as u16, 0, &mut setup, setup_fns, handle_fn);
        }
    }
}

fn create_positions_rec(tb: &Tablebase<Chess>, piece_count: usize, own_king: u16, opp_king: u16, index: u16, setup: &mut TBSetup, setup_fns: &[SetupFn], handle_fn: &mut impl FnMut(u16, u16, u16, i8)) {
    if setup_fns.is_empty() {
        if setup.count() == piece_count {
            match Chess::from_setup(setup, CastlingMode::Standard) {
                Ok(mut chess) => {
                    let result = tablebase_result(tb, &mut chess);
                    handle_fn(index, own_king, opp_king, result);
                }

                Err(e) => {
                    if e.kinds().contains(PositionErrorKinds::OPPOSITE_CHECK) {
                        handle_fn(index, own_king, opp_king, if setup.stm.is_white() { 1 } else { -1 } );
                    }
                }
            }
        }
        return;
    }

    if let Some((setup_fn, remaining_fns)) = setup_fns.split_first() {
        for pos in 0..64 {
            if pos == own_king || pos == opp_king {
                continue;
            }
            let mut next_setup = setup.clone();
            setup_fn(&mut next_setup, pos as u32);

            create_positions_rec(tb, piece_count, own_king, opp_king, index * 64 + pos as u16, &mut next_setup, remaining_fns, handle_fn);
        }
    }
}

fn verify_positions(tb: &Tablebase<Chess>, stm: Color, setup_fns: &[SetupFn], handle_fn: &mut impl FnMut(u16, u16, Vec<u16>, i8, String)) {
    let piece_count = 2 + setup_fns.len();

    for wk in 0..64 {
        for bk in 0..64 {
            let row_distance = (wk / 8).max(bk / 8) - (wk / 8).min(bk / 8);
            let col_distance = (wk & 7).max(bk & 7) - (wk & 7).min(bk & 7);
            let distance = row_distance.max(col_distance);

            if distance <= 1 {
                continue;
            }

            let mut setup = TBSetup::new(stm);
            setup.set(wk, Color::White, Role::King);
            setup.set(bk, Color::Black, Role::King);

            verify_positions_rec(tb, piece_count, wk as u16, bk as u16, Vec::new(), &mut setup, setup_fns, handle_fn);
        }
    }
}

fn verify_positions_rec(tb: &Tablebase<Chess>, piece_count: usize, own_king: u16, opp_king: u16, pieces: Vec<u16>, setup: &mut TBSetup, setup_fns: &[SetupFn], handle_fn: &mut impl FnMut(u16, u16, Vec<u16>, i8, String)) {
    if setup_fns.is_empty() {
        if setup.count() == piece_count {
            if let Ok(mut chess) = Chess::from_setup(setup, CastlingMode::Standard) {
                let result = tablebase_result(tb, &mut chess);
                let fen = Fen::from_setup(&chess).to_string();

                handle_fn(own_king, opp_king, pieces, result, fen);
            }
        }
        return;
    }

    if let Some((setup_fn, remaining_fns)) = setup_fns.split_first() {
        for pos in 0..64 {
            if pos == own_king || pos == opp_king {
                continue;
            }
            let mut next_setup = setup.clone();
            setup_fn(&mut next_setup, pos as u32);

            let mut pieces = pieces.clone();
            pieces.push(pos as u16);
            verify_positions_rec(tb, piece_count, own_king, opp_king, pieces, &mut next_setup, remaining_fns, handle_fn);
        }
    }
}

fn tablebase_result(tb: &Tablebase<Chess>, pos: &mut Chess) -> i8 {
    let stm_modifier = if pos.turn().is_white() { 1 } else { -1 };
    match tb.probe_wdl_after_zeroing(pos) {
        Ok(result) => {
            result.signum() as i8 * stm_modifier
        }
        Err(e) => {
            panic!("could not probe TB: {}", e);
        }
    }
}