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

use crate::engine::{MAX_ELO, Message, MIN_ELO};
use crate::fen::START_POS;
use crate::search::{DEFAULT_SEARCH_THREADS, MAX_SEARCH_THREADS};
use crate::time_management::{DEFAULT_MOVE_OVERHEAD_MS, MAX_MOVE_OVERHEAD_MS, MIN_MOVE_OVERHEAD_MS, SearchLimits};
use crate::transposition_table::{DEFAULT_SIZE_MB, MAX_DEPTH, MAX_HASH_SIZE_MB};
use crate::uci_move::UCIMove;
use std::collections::HashSet;
use std::fmt::{Display};
use std::io;
use std::str::FromStr;
use std::sync::mpsc::Sender;
use std::thread::sleep;
use std::time::Duration;
use crate::params;
use crate::syzygy::{DEFAULT_TB_PROBE_DEPTH, HAS_TB_SUPPORT};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const AUTHOR: &str = "Martin Honert";

const MAX_MULTI_PV_MOVES: usize = 218;

const GO_CMDS: [&str; 12] = [
    "searchmoves",
    "ponder",
    "wtime",
    "btime",
    "winc",
    "binc",
    "movestogo",
    "depth",
    "nodes",
    "mate",
    "movetime",
    "infinite",
];

pub fn start_uci_loop(tx: &Sender<Message>) {
    println!("Velvet Chess Engine v{}", VERSION);

    let go_cmds = HashSet::from(GO_CMDS);

    loop {
        let mut line = String::new();
        io::stdin().read_line(&mut line).expect("Failed to read line");

        let parts: Vec<&str> = line.split_whitespace().collect();
        for (i, part) in parts.iter().enumerate() {
            match part.to_lowercase().as_str() {
                "go" => go(tx, &go_cmds, parts[i + 1..].to_vec()),

                "isready" => is_ready(tx),

                "perft" => perft(tx, parts[i + 1..].to_vec()),

                "position" => set_position(tx, &parts[i + 1..]),

                "profile" => {
                    profile(tx);
                }
                
                "bench" => send_message(tx, Message::Bench),

                "quit" => {
                    send_message(tx, Message::Stop);
                    sleep(Duration::from_millis(10));

                    send_message(tx, Message::Quit);
                    sleep(Duration::from_millis(10));
                    return;
                }

                "setoption" => set_option(tx, &parts[i + 1..]),

                "stop" => send_message(tx, Message::Stop),

                "ponderhit" => send_message(tx, Message::PonderHit),

                "uci" => uci(),

                "ucinewgame" => uci_new_game(tx),

                "printfen" => fen(tx),

                _ => {
                    // Skip unknown commands
                    continue;
                }
            }
            break;
        }
    }
}

// Sends a message to the engine
fn send_message(tx: &Sender<Message>, msg: Message) {
    match tx.send(msg) {
        Ok(_) => {}
        Err(err) => {
            eprintln!("info string error: could not send message to engine thread: {}", err);
        }
    }
}

fn uci() {
    println!("id name Velvet v{}", VERSION);
    println!("id author {}", AUTHOR);
    println!("option name Clear Hash type button");
    println!("option name Hash type spin default {} min 1 max {}", DEFAULT_SIZE_MB, MAX_HASH_SIZE_MB);
    println!("option name Move Overhead type spin default {} min {} max {}", DEFAULT_MOVE_OVERHEAD_MS, MIN_MOVE_OVERHEAD_MS, MAX_MOVE_OVERHEAD_MS);
    println!("option name MultiPV type spin default 1 min 1 max {}", MAX_MULTI_PV_MOVES);
    println!("option name Ponder type check default false");
    println!("option name SimulateThinkingTime type check default true");
    println!("option name Style type combo default Normal var Normal");
    if HAS_TB_SUPPORT {
        println!("option name SyzygyPath type string default");
        println!("option name SyzygyProbeDepth type spin default {} min 0 max {}", DEFAULT_TB_PROBE_DEPTH, MAX_DEPTH);
    }
    println!("option name Threads type spin default {} min 1 max {}", DEFAULT_SEARCH_THREADS, MAX_SEARCH_THREADS);
    println!("option name UCI_Chess960 type check default false");
    println!("option name UCI_Elo type spin default {} min {} max {}", MIN_ELO, MIN_ELO, MAX_ELO);
    println!("option name UCI_LimitStrength type check default false");
    println!("option name UCI_Opponent type string default");
    println!("option name UCI_RatingAdv type spin default 0 min -10000 max 10000");
    println!(
        "option name UCI_EngineAbout type string default Velvet Chess Engine (https://github.com/mhonert/velvet-chess)"
    );
    params::print_options();

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
        Some(idx) => parse_moves(idx, parts),
        None => Vec::new(),
    };

    send_message(tx, Message::SetPosition(fen, moves));
}

fn set_option(tx: &Sender<Message>, parts: &[&str]) {
    if parts.len() < 2 {
        println!("info string error: missing parameters for setoption");
        return;
    }

    if parts[0] != "name" {
        println!("info string error: missing 'name' in setoption");
        return;
    }

    let name = parts[1].to_ascii_lowercase();

    let value = if let Some(value_idx) = parts.iter().position(|p| p.eq_ignore_ascii_case("value")) {
        parts[value_idx + 1..].join(" ")
    } else {
        String::new()
    };

    match name.as_str() {
        "hash" => {
            if let Some(size_mb) = parse_numeric_option(value.as_str(), 1, MAX_HASH_SIZE_MB) {
                send_message(tx, Message::SetTranspositionTableSize(size_mb));
            } else {
                println!("info string error: invalid hash size: {}", value);
            };
        }

        "clear" => {
            if parts.len() > 2 && parts[2].eq_ignore_ascii_case("hash") {
                send_message(tx, Message::ClearHash);
            }
        }

        "threads" => {
            if let Some(threads) = parse_numeric_option(value.as_str(), 1, MAX_SEARCH_THREADS as i32) {
                send_message(tx, Message::SetThreadCount(threads));
            } else {
                println!("info string error: invalid thread count: {}", value);
            };
        }

        "syzygypath" => {
            if !HAS_TB_SUPPORT {
                println!("info string warning: unknown option: SyzygyPath");
                return;
            }

            println!("info string setting path to {}", value);
            send_message(tx, Message::SetTableBasePath(value.to_string()));
        }

        "syzygyprobedepth" => {
            if !HAS_TB_SUPPORT {
                println!("info string warning: unknown option: SyzygyProbeDepth");
                return;
            }

            if let Some(depth) = parse_numeric_option(value.as_str(), 0, MAX_DEPTH as i32) {
                send_message(tx, Message::SetTableBaseProbeDepth(depth));
            } else {
                println!("info string error: invalid probe depth: {}", value);
            };
        }

        "ponder" => {}

        "uci_chess960" => {}

        "uci_limitstrength" => {
            let limit_strength = value.as_str().eq_ignore_ascii_case("true");
            send_message(tx, Message::SetLimitStrength(limit_strength));
        }
        
        "uci_elo" => {
            if let Some(elo) = parse_numeric_option(value.as_str(), MIN_ELO, MAX_ELO) {
                send_message(tx, Message::SetElo(elo));
            } else {
                println!("info string error: unsupported Elo number: {}", value);
            }
        }

        "uci_opponent" => {}

        "uci_ratingadv" => {
            if let Some(rating_adv) = parse_numeric_option(value.as_str(), -10000.0, 10000.0) {
                send_message(tx, Message::SetRatingAdv(rating_adv as i32));
            } else {
                println!("info string error: invalid UCI_RatingAdv value: {}", value);
            }
        }

        // "ratingadvadaptivestyle" => {
        //     let adaptive = value.as_str().eq_ignore_ascii_case("true");
        //     send_message(tx, Message::SetRatingAdvAdaptiveStyle(adaptive));
        // }
        //
        // "ratingadvriskystylethreshold" => {
        //     if let Some(threshold) = parse_numeric_option(value.as_str(), -10000.0, 10000.0) {
        //         send_message(tx, Message::SetRatingAdvRiskyStyleThreshold(threshold as i32));
        //     } else {
        //         println!("info string error: invalid RatingAdvRiskyStyleThreshold value: {}", value);
        //     }
        // }

        "multipv" => {
            if let Some(multipv_moves) = parse_numeric_option(value.as_str(), 1, MAX_MULTI_PV_MOVES as i32) {
                send_message(tx, Message::SetMultiPV(multipv_moves));
            } else {
                println!("info string error: invalid number of MultiPV moves: {}", value);
            };
        }

        "move" => {
            if parts.len() > 2 && parts[2].eq_ignore_ascii_case("overhead") {
                if let Some(ms) = parse_numeric_option(value.as_str(), MIN_MOVE_OVERHEAD_MS, MAX_MOVE_OVERHEAD_MS) {
                    send_message(tx, Message::SetMoveOverheadMillis(ms));
                } else {
                    println!("info string error: invalid move overhead value");
                }
            }
        }
        
        "simulatethinkingtime" => {
            let simulate_thinking_time = value.as_str().eq_ignore_ascii_case("true");
            send_message(tx, Message::SetSimulateThinkingTime(simulate_thinking_time));
        }
        
        _ => {
            if let Some(value) = parse_numeric_option(value.as_str(), i16::MIN, i16::MAX) {
                send_message(tx, Message::SetParam(name, value));
            } else {
                println!("info string error: invalid value for param {}: {}", name, value)
            }
        },
        
    }
}

fn parse_numeric_option<T: FromStr + PartialOrd + Display>(value: &str, min_value: T, max_value: T) -> Option<T> {
    let value = match T::from_str(value) {
        Ok(v) => v,
        Err(_) => {
            return None;
        }
    };

    if value < min_value {
        println!("info string warning: value too low: setting to allowed minimum: {}", min_value);
        Some(min_value)
    } else if value > max_value {
        println!("info string warning: value too high: setting to allowed maximum: {}", max_value);
        Some(max_value)
    } else {
        Some(value)
    }
}

fn parse_moves(idx: usize, parts: &[&str]) -> Vec<UCIMove> {
    let mut moves: Vec<UCIMove> = Vec::new();

    for i in (idx + 1)..parts.len() {
        match UCIMove::from_uci(parts[i]) {
            Some(m) => moves.push(m),
            None => {
                eprintln!("info string error: could not parse move notation: {}", parts[i]);
                return moves;
            }
        }
    }

    moves
}

fn perft(tx: &Sender<Message>, parts: Vec<&str>) {
    if parts.is_empty() {
        println!("info string error: missing depth parameter");
        return;
    }

    match i32::from_str(parts[0]) {
        Ok(depth) => send_message(tx, Message::Perft(depth)),
        Err(_) => println!("info string error: invalid depth parameter: {}", parts[0]),
    };
}

fn go(tx: &Sender<Message>, valid_cmds: &HashSet<&str>, parts: Vec<&str>) {
    let mut depth_limit: Option<i32> = None;
    let mut node_limit: Option<u64> = None;
    let mut wtime: Option<i32> = None;
    let mut btime: Option<i32> = None;
    let mut winc: Option<i32> = None;
    let mut binc: Option<i32> = None;
    let mut move_time: Option<i32> = None;
    let mut moves_to_go: Option<i32> = None;
    let mut search_moves: Option<Vec<String>> = None;
    let mut infinite = false;
    let mut ponder = false;
    let mut mate_limit: Option<i16> = None;

    let mut i = 0;
    while i < parts.len() {
        i = match parts[i] {
            "wtime" => set_cmd_arg(&parts, &mut wtime, i + 1),
            "btime" => set_cmd_arg(&parts, &mut btime, i + 1),
            "winc" => set_cmd_arg(&parts, &mut winc, i + 1),
            "binc" => set_cmd_arg(&parts, &mut binc, i + 1),
            "movetime" => set_cmd_arg(&parts, &mut move_time, i + 1),
            "movestogo" => set_cmd_arg(&parts, &mut moves_to_go, i + 1),
            "depth" => set_cmd_arg(&parts, &mut depth_limit, i + 1),
            "nodes" => set_cmd_arg(&parts, &mut node_limit, i + 1),
            "mate" => set_cmd_arg(&parts, &mut mate_limit, i + 1),
            "searchmoves" => parse_cmd_multi_arg(valid_cmds, &parts, &mut search_moves, i + 1),
            "ponder" => {
                ponder = true;
                i + 1
            }
            "infinite" => {
                infinite = true;
                i + 1
            }
            _ => i + 1,
        }
    }

    let limits = if infinite {
        SearchLimits::infinite()
    } else {
        match SearchLimits::new(depth_limit, node_limit, wtime, btime, winc, binc, move_time, moves_to_go, mate_limit) {
            Ok(limits) => limits,
            Err(e) => {
                eprintln!("info string error: invalid search params: {}", e);
                return;
            }
        }
    };

    send_message(tx, Message::Go(limits, ponder, search_moves));
}

fn profile(tx: &Sender<Message>) {
    send_message(tx, Message::Profile);
    sleep(Duration::from_millis(500));
}

fn fen(tx: &Sender<Message>) {
    send_message(tx, Message::Fen);
}

fn set_cmd_arg<T: FromStr>(parts: &[&str], target: &mut Option<T>, pos: usize) -> usize {
    if let Some(value) = parts.get(pos) {
        *target = T::from_str(value).ok();
    }

    pos + 1
}

fn parse_cmd_multi_arg<T: FromStr>(
    valid_cmds: &HashSet<&str>, parts: &[&str], target: &mut Option<Vec<T>>, mut pos: usize,
) -> usize {
    let mut values = Vec::new();
    while let Some(&value) = parts.get(pos) {
        if valid_cmds.contains(value.to_lowercase().as_str()) {
            break;
        }

        if let Ok(value) = T::from_str(value) {
            values.push(value);
        } else {
            break;
        }

        pos += 1;
    }
    *target = Some(values);
    pos
}

fn parse_position_cmd(parts: &[&str]) -> String {
    if parts.is_empty() {
        eprintln!("info string error: missing fen/startpos");
        return String::from(START_POS);
    }

    let pos_end = parts.iter().position(|&part| part.to_lowercase().as_str() == "moves").unwrap_or(parts.len());

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
