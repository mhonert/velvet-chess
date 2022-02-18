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
use crate::transposition_table::{DEFAULT_SIZE_MB, MAX_HASH_SIZE_MB};
use crate::uci_move::UCIMove;
use std::io;
use std::str::FromStr;
use std::sync::mpsc::Sender;
use std::thread::{sleep};
use std::time::Duration;
use crate::search::{SearchLimits, DEFAULT_SEARCH_THREADS, MAX_SEARCH_THREADS};

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
                    send_message(tx, Message::Stop);
                    sleep(Duration::from_millis(10));

                    send_message(tx, Message::Quit);
                    sleep(Duration::from_millis(10));
                    return;
                }

                "setoption" => set_option(tx, &parts[i + 1..].to_vec()),

                "stop" => send_message(tx, Message::Stop),

                "ponderhit" => send_message(tx, Message::PonderHit),

                "uci" => uci(),

                "ucinewgame" => uci_new_game(tx),

                "printfen" => fen(tx),

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
    println!("option name Hash type spin default {} min 1 max {}", DEFAULT_SIZE_MB, MAX_HASH_SIZE_MB);
    println!("option name Ponder type check default false");
    println!("option name Threads type spin default {} min 1 max {}", DEFAULT_SEARCH_THREADS, MAX_SEARCH_THREADS);
    println!("option name UCI_Chess960 type check default false");
    println!("option name UCI_EngineAbout type string default Velvet Chess Engine (https://github.com/mhonert/velvet-chess)");

    println!("uciok");
}

fn uci_new_game(tx: &Sender<Message>) {
    send_message(tx, Message::NewGame);
}

fn is_ready(tx: &Sender<Message>) {
    send_message(tx, Message::IsReady);
}

fn set_position(tx: &Sender<Message>, parts: &[&str]) {
    let fen = parse_position_cmd(parts);

    let moves = match parts.iter().position(|&part| part == "moves") {
        Some(idx) => parse_moves(idx, &parts),
        None => Vec::new(),
    };

    send_message(tx, Message::SetPosition(fen, moves));
}

fn set_option(tx: &Sender<Message>, parts: &[&str]) {
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

    match name.as_str() {
        "hash" => {
            if let Some(size_mb) = parse_int_option(value, 1, MAX_HASH_SIZE_MB) {
                send_message(tx, Message::SetTranspositionTableSize(size_mb));
            } else {
                println!("Invalid hash size: {}", value);
            };
        }

        "threads" => {
            if let Some(threads) = parse_int_option(value, 1, MAX_SEARCH_THREADS as i32) {
                send_message(tx, Message::SetThreadCount(threads));
            } else {
                println!("Invalid thread count: {}", value);
            };
        }

        "ponder" => {}

        "uci_chess960" => {}

        _ => println!("Unknown option: {}", name)
    }
}

fn parse_int_option(value: &str, min_value: i32, max_value: i32) -> Option<i32> {
    let value = match i32::from_str(value) {
        Ok(v) => v,
        Err(_) => {
            return None;
        }
    };

    if !(min_value..=max_value).contains(&value) {
        return None;
    }

    Some(value)
}

fn parse_moves(idx: usize, parts: &[&str]) -> Vec<UCIMove> {
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
    let ponder = parts.contains(&"ponder");

    if parts.is_empty() || parts.contains(&"infinite") {
        send_message(tx, Message::Go(SearchLimits::default(), ponder));
        return;
    }

    let depth = extract_option(&parts, "depth");
    let wtime = extract_option(&parts, "wtime");
    let btime = extract_option(&parts, "btime");
    let winc = extract_option(&parts, "winc");
    let binc = extract_option(&parts, "binc");
    let nodes = extract_option(&parts, "nodes");
    let movetime = extract_option(&parts, "movetime");
    let movestogo = extract_option(&parts, "movestogo");

    let limits = match SearchLimits::new(depth, nodes, wtime, btime, winc, binc, movetime, movestogo) {
        Ok(limits) => limits,
        Err(e) => {
            eprintln!("go: invalid search params: {}", e);
            return;
        }
    };

    send_message(tx, Message::Go(limits, ponder));
}

fn profile(tx: &Sender<Message>) {
    send_message(tx, Message::Profile);
    sleep(Duration::from_millis(500));
}

fn fen(tx: &Sender<Message>) {
    send_message(tx, Message::Fen);
}

fn extract_option<T: std::str::FromStr>(parts: &[&str], name: &str) -> Option<T> {
    match parts.iter().position(|&item| item == name) {
        Some(pos) => {
            if pos + 1 >= parts.len() {
                return None;
            }

            match T::from_str(parts[pos + 1]) {
                Ok(value) => Some(value),
                Err(_) => None,
            }
        }
        None => None,
    }
}

fn parse_position_cmd(parts: &[&str]) -> String {
    if parts.is_empty() {
        eprintln!("position command: missing fen/startpos");
        return String::from(START_POS);
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
        let parts: Vec<&str> = "   startpos moves e1e2  ".split_whitespace().collect();
        assert_eq!(parse_position_cmd(&parts), START_POS);
    }

    #[test]
    fn test_parse_position_fen() {
        let fen: &str = "r3k1r1/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K1R1 w Qq - 0 1";
        let part_str = format!("   fen \t {}   moves e1e2  ", fen);
        let parts: Vec<&str> = part_str.split_whitespace().collect();

        assert_eq!(parse_position_cmd(&parts), fen);
    }
}
