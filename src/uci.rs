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

use crate::engine::Message;
use crate::fen::START_POS;
use crate::transposition_table::{DEFAULT_SIZE_MB, MAX_HASH_SIZE_MB, MAX_DEPTH};
use crate::uci_move::UCIMove;
use std::io;
use std::str::FromStr;
use std::sync::mpsc::Sender;
use std::thread::sleep;
use std::time::Duration;
use crate::options::parse_set_option;
use crate::magics::{find_magics, find_sparse_rook_magics, find_sparse_bishop_magics};
use crate::uci::MagicNumPiece::{Rooks, Bishops};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = "Martin Honert";

pub fn start_uci_loop(tx: &Sender<Message>) {
    println!("Velvet Chess Engine v{}", VERSION);

    loop {
        let mut line = String::new();
        io::stdin()
            .read_line(&mut line)
            .expect("Failed to read line");

        let parts: Vec<&str> = line.split_whitespace().collect();
        for (i, part) in parts.iter().enumerate() {
            match part.to_lowercase().as_str() {
                "go" => go(tx, parts[i + 1..].to_vec()),

                "isready" => is_ready(tx),

                "perft" => perft(tx, parts[i + 1..].to_vec()),

                "position" => set_position(tx, &parts[i + 1..].to_vec()),

                "profile" => {
                    profile(tx);
                    return;
                }

                "quit" => {
                    send_message(tx, Message::Quit);
                    return;
                }

                "setoption" => set_option(tx, &parts[i + 1..].to_vec()),

                "stop" => send_message(tx, Message::Stop),

                "uci" => uci(),

                "ucinewgame" => uci_new_game(tx),

                "prepare_eval" => prepare_eval(tx, parts[i + 1..].to_vec()),

                "prepare_quiet" => prepare_quiet(tx, parts[i + 1..].to_vec()),

                "eval" => eval(tx, parts[i + 1]),

                "printtestpositions" => print_test_positions(tx),

                "resettestpositions" => reset_test_positions(tx),

                "printfen" => fen(tx),

                "magics" => find_magics(),

                "sparse_rook_magics" => calc_magics(Rooks, parts[i + 1..].to_vec()),

                "sparse_bishop_magics" => calc_magics(Bishops, parts[i + 1..].to_vec()),

                _ => {
                    // Skip unknown commands
                }
            }
        }
    }
}

// Sends a message to the engine
fn send_message(tx: &Sender<Message>, msg: Message) {
    match tx.send(msg) {
        Ok(_) => {},
        Err(err) => {
            eprintln!("could not send message to engine thread: {}", err);
        }
    }
}

fn uci() {
    println!("id name Velvet v{}", VERSION);
    println!("id author {}", AUTHOR);
    println!("option name UCI_EngineAbout type string default Velvet Chess Engine (https://github.com/mhonert/velvet-chess)");
    println!(
        "option name Hash type spin default {} min 1 max {}",
        DEFAULT_SIZE_MB, MAX_HASH_SIZE_MB
    );

    println!("uciok");
}

fn uci_new_game(tx: &Sender<Message>) {
    send_message(tx, Message::NewGame);
}

fn is_ready(tx: &Sender<Message>) {
    send_message(tx, Message::IsReady);
}

fn set_position(tx: &Sender<Message>, parts: &Vec<&str>) {
    let fen = parse_position_cmd(parts);

    let moves = match parts.iter().position(|&part| part == "moves") {
        Some(idx) => parse_moves(idx, &parts),
        None => Vec::new(),
    };

    send_message(tx, Message::SetPosition(fen, moves));
}

fn set_option(tx: &Sender<Message>, parts: &Vec<&str>) {
    if parts.len() < 4 {
        println!("Missing parameters for setoption");
        return;
    }

    if parts[0] != "name" {
        println!("Missing 'name' in setoption");
        return;
    }

    let name = parts[1].to_ascii_lowercase();
    if parts[2] != "value" {
        println!("Missing 'value' in setoption");
        return;
    }

    let value = parts[3];

    if name == "hash" {
        let size_mb = match i32::from_str(value) {
            Ok(size) => size,
            Err(_) => {
                println!("Invalid hash size: {}", value);
                return;
            }
        };

        if size_mb < 1 || size_mb > MAX_HASH_SIZE_MB {
            println!("Invalid hash size: {}", size_mb);
            return;
        }

        send_message(tx, Message::SetTranspositionTableSize(size_mb));

    } else {
        parse_set_option(tx, &name, value);

    }
}

fn parse_moves(idx: usize, parts: &Vec<&str>) -> Vec<UCIMove> {
    let mut moves: Vec<UCIMove> = Vec::new();

    for i in (idx + 1)..parts.len() {
        match UCIMove::from_uci(parts[i]) {
            Some(m) => moves.push(m),
            None => {
                eprintln!("could not parse move notation: {}", parts[i]);
                return moves;
            }
        }
    }

    moves
}

fn perft(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.is_empty() {
        println!("perft cmd: missing depth");
        return;
    }

    match i32::from_str(parts[0]) {
        Ok(depth) => send_message(tx, Message::Perft(depth)),
        Err(_) => println!("perft cmd: invalid depth parameter: {}", parts[0]),
    };
}

fn go(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.is_empty() || parts.contains(&"infinite") {
        send_message(
            tx,
            Message::Go {
                depth: MAX_DEPTH as i32,
                wtime: -1,
                btime: -1,
                winc: 0,
                binc: 0,
                movetime: i32::max_value(),
                movestogo: 1,
                nodes: u64::max_value(),
            }
        );
        return;
    }

    let depth = extract_option(&parts, "depth", MAX_DEPTH as i32);
    let wtime = extract_option(&parts, "wtime", -1);
    let btime = extract_option(&parts, "btime", -1);
    let winc = extract_option(&parts, "winc", 0);
    let binc = extract_option(&parts, "binc", 0);
    let nodes = extract_option(&parts, "nodes", u64::max_value());
    let movetime = extract_option(&parts, "movetime", -1);
    let movestogo = extract_option(&parts, "movestogo", 40);

    if depth <= 0 {
        println!("go cmd: invalid depth: {}", depth);
        return;
    }

    send_message(
        tx,
        Message::Go {
            depth,
            wtime,
            btime,
            winc,
            binc,
            movetime,
            movestogo,
            nodes
        },
    );
}

fn prepare_eval(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.is_empty() {
        println!("prepare_eval cmd: missing fen positions");
        return;
    }

    let fens_str: String = parts.join(" ");
    let fens_with_result: Vec<(String, f64)> = fens_str.split(';').filter(|&s| !s.is_empty()).map(extract_fen_result).collect();

    send_message(tx, Message::PrepareEval(fens_with_result));
}

fn prepare_quiet(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.is_empty() {
        println!("prepare_quiet cmd: missing fen positions");
        return;
    }

    let fens_str: String = parts.join(" ");
    let fens_with_result: Vec<(String, f64)> = fens_str.split(';').filter(|&s| !s.is_empty()).map(extract_fen_result).collect();

    send_message(tx, Message::PrepareQuiet(fens_with_result));
}

fn extract_fen_result(s: &str) -> (String, f64) {
    let pair: Vec<&str> = s.split(':').collect();
    if pair.len() != 2 {
        panic!("prepare_eval expects a list of 'FEN:result' pairs, separated by ;")
    }
    let fen = String::from(pair[0]);
    let result = match f64::from_str(pair[1]) {
        Ok(r) => r,
        Err(e) => panic!("Could not parse result: {}", e)
    };

    (fen, result)
}

fn eval(tx: &Sender<Message>, k_str: &str) {
    let k = match f64::from_str(k_str) {
        Ok(k) => k,
        Err(e) => panic!("Could not parse K {}: {}", k_str, e)
    };

    send_message(tx, Message::Eval(k));
}

fn print_test_positions(tx: &Sender<Message>) {
    send_message(tx, Message::PrintTestPositions);
}

fn reset_test_positions(tx: &Sender<Message>) {
    send_message(tx, Message::ResetTestPositions);
}

fn profile(tx: &Sender<Message>) {
    send_message(tx, Message::Profile);
    sleep(Duration::from_secs(5));
}

fn fen(tx: &Sender<Message>) {
    send_message(tx, Message::Fen);
}

enum MagicNumPiece {
    Rooks,
    Bishops
}

fn calc_magics(piece: MagicNumPiece, parts: Vec<&str>) {
    if parts.is_empty() {
        println!("calc_magics requires a pos number");
        return;
    }

    let pos_str = parts[0];
    let pos = match i32::from_str(pos_str) {
        Ok(p) => p,
        Err(e) => {
            println!("Could not parse pos number {}: {}", pos_str, e);
            return;
        }
    };

    if pos < 0 || pos > 63 {
        println!("Position must be in the range of 0..+63");
        return;
    }

    match piece {
        Rooks => find_sparse_rook_magics(pos),
        Bishops => find_sparse_bishop_magics(pos)
    }
}

fn extract_option<T: std::str::FromStr>(parts: &Vec<&str>, name: &str, default_value: T) -> T {
    match parts.iter().position(|&item| item == name) {
        Some(pos) => {
            if pos + 1 >= parts.len() {
                return default_value;
            }

            match T::from_str(parts[pos + 1]) {
                Ok(value) => value,
                Err(_) => default_value,
            }
        }
        None => default_value,
    }
}

fn parse_position_cmd(parts: &Vec<&str>) -> String {
    if parts.is_empty() {
        eprintln!("position command: missing fen/startpos");
    }

    let pos_end = parts
        .iter()
        .position(|&part| part.to_lowercase().as_str() == "moves")
        .unwrap_or_else(|| parts.len());

    let pos_option = parts[1..pos_end].join(" ");

    if pos_option.is_empty() {
        String::from(START_POS)
    } else {
        pos_option
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fen::START_POS;

    #[test]
    fn test_parse_position_startpos() {
        assert_eq!(
            parse_position_cmd(&"   startpos moves e1e2  ".split_whitespace().collect()),
            START_POS
        );
    }

    #[test]
    fn test_parse_position_fen() {
        let fen: &str = "r3k1r1/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K1R1 w Qq - 0 1";
        assert_eq!(
            parse_position_cmd(
                &format!("   fen \t {}   moves e1e2  ", fen)
                    .split_whitespace()
                    .collect()
            ),
            fen
        );
    }
}
