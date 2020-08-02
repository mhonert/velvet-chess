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

use std::io;
use crate::fen::{read_fen, START_POS};
use std::time::SystemTime;
use crate::engine::Message;
use std::sync::mpsc::Sender;
use std::str::FromStr;

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
                "uci" =>  uci(),

                "ucinewgame" => uci_new_game(),

                "isready" => is_ready(),

                "position" => set_position(tx, parts[i + 1..].to_vec()),

                "perft" => perft(tx, parts[i + 1..].to_vec()),

                "quit" => {
                    send_message(tx, Message::Quit());
                    return;
                }

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
        Ok(_) => return,
        Err(err) => {
            eprintln!("could not send message to engine thread: {}", err);
        }
    }
}


fn uci() {
    println!("id name Velvet v{}", VERSION);
    println!("id author {}", AUTHOR);
    println!("option name UCI_EngineAbout type string default Velvet Chess Engine (https://github.com/mhonert/velvet-chess)");

    println!("uciok");
}

fn uci_new_game() {}

fn is_ready() {
    println!("readyok");
}

fn set_position(tx: &Sender<Message>, parts: Vec<&str>) {
    let fen = parse_position_cmd(parts);
    send_message(tx, Message::SetPosition(fen));
}

fn perft(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.len() == 0 {
        println!("perft cmd: missing depth");
        return
    }

    match i32::from_str(parts[0]) {
        Ok(depth) => send_message(tx, Message::Perft(depth)),
        Err(_) => println!("perft cmd: invalid depth parameter: {}", parts[0])
    };
}

fn parse_position_cmd(parts: Vec<&str>) -> String {
    if parts.is_empty() {
        eprintln!("position command: missing fen/startpos");
    }

    let pos_end = parts.iter().position(|&part| part.to_lowercase().as_str() == "moves").unwrap_or_else(|| parts.len());

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
        assert_eq!(parse_position_cmd("   startpos moves e1e2  ".split_whitespace().collect()), START_POS);
    }

    #[test]
    fn test_parse_position_fen() {
        let fen: &str = "r3k1r1/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K1R1 w Qq - 0 1";
        assert_eq!(parse_position_cmd(format!("   fen \t {}   moves e1e2  ", fen).split_whitespace().collect()), fen);
    }
}
