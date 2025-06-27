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

use std::process::{Command, Stdio};
use std::io::{BufRead, Write};
use anyhow::bail;
use crate::config::EngineConfig;

pub struct UciEngine {
    config: EngineConfig,
    child: std::process::Child,
    debug_log: bool,
}

impl UciEngine {
    pub fn start(config: &EngineConfig) -> anyhow::Result<UciEngine> {
        let child = Command::new(&config.cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        anyhow::Ok(UciEngine {
            config: config.clone(),
            child,
            debug_log: false,
        })
    }

    pub fn init(&mut self) -> anyhow::Result<()> {
        for cmd in &self.config.init_commands.clone() {
            self.send_command(cmd)?;
        }

        self.send_command("uci")?;
        self.expect_response("uciok")?;

        for (opt, val) in &self.config.options.clone() {
            self.send_command(&format!("setoption name {} value {}", opt, val))?;
        }

        self.ready()?;

        anyhow::Ok(())
    }

    pub fn ready(&mut self) -> anyhow::Result<()> {
        self.send_command("isready")?;
        self.expect_response("readyok")?;

        anyhow::Ok(())
    }

    pub fn uci_newgame(&mut self) -> anyhow::Result<()> {
        self.send_command("ucinewgame")?;

        anyhow::Ok(())
    }
    pub fn go(&mut self, opening: &str, wtime: i32, btime: i32, inc: i32, moves: &String, nodes: Option<i32>) -> anyhow::Result<String> {
        let position = if moves.is_empty() {
            format!("position fen {}", opening)
        } else {
            format!("position fen {} moves {}", opening, moves)
        };

        self.send_command(&position)?;
        if let Some(nodes) = nodes {
            self.send_command(&format!("go nodes {nodes}"))?;
        } else {
            self.send_command(&format!("go wtime {wtime} winc {inc} btime {btime} binc {inc}"))?;
        }

        let response = self.expect_response("bestmove")?;

        // Extract best move from response
        // The best move appears after the "bestmove" token, which can appear anywhere in the response
        let parts: Vec<&str> = response.split_whitespace().collect();
        if let Some(best_move) = parts.iter().skip_while(|&&x| x != "bestmove").nth(1) {
            Ok(best_move.to_string())
        } else {
            bail!("Could not extract best move from response");
        }
    }

    pub fn quit(&mut self) -> anyhow::Result<()> {
        self.send_command("quit")?;
        self.child.wait()?;

        anyhow::Ok(())
    }

    fn send_command(&mut self, cmd: &str) -> anyhow::Result<()> {
        if let Some(stdin) = self.child.stdin.as_mut() {
            if self.debug_log {
                println!("{} w > {}", self.config.name, cmd);
            }
            writeln!(stdin, "{}", cmd)?;
            anyhow::Ok(())
        } else {
            bail!("Could not get stdin handle for engine process");
        }
    }

    fn expect_response(&mut self, expected_response: &str) -> anyhow::Result<String> {
        if let Some(stdout) = self.child.stdout.as_mut() {
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let line = line?;
                if self.debug_log {
                    println!("{} r < {}", self.config.name, line);
                }
                if line.contains(expected_response) {
                    return anyhow::Ok(line);
                }
            }
            bail!("Expected response '{}' not received", expected_response);
        } else {
            bail!("Could not get stdout handle for engine process");
        }
    }
    
    pub fn name(&self) -> String {
        self.config.name.clone()
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        if let Err(e) = self.quit() {
            eprintln!("Could not quit engine process: {}", e);
            if let Err(e) = self.child.kill() {
                eprintln!("Could not kill engine process: {}", e);
            }
        }
    }
}
