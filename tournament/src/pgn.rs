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
use std::fs::File;
use std::io::Write;
use std::io::BufWriter;
use chrono::{DateTime, Local};
use selfplay::selfplay::Outcome;

pub struct PgnWriter {
    writer: BufWriter<File>,
}

impl PgnWriter {
    pub fn new(file_path: &str) -> anyhow::Result<PgnWriter> {
        let file = File::create(file_path)?;
        let writer = BufWriter::new(file);

        anyhow::Ok(PgnWriter { writer })
    }

    pub fn write_game(&mut self, game: PgnGame) -> anyhow::Result<()> {
        writeln!(self.writer, "[Event \"Velvet Test Gauntlet\"]")?;
        writeln!(self.writer, "[Site \"local\"]")?;
        writeln!(self.writer, "[Date \"{}\"]", chrono::Local::now().format("%Y.%m.%d"))?;
        writeln!(self.writer, "[Round \"{}\"]", game.round)?;
        writeln!(self.writer, "[White \"{}\"]", game.white)?;
        writeln!(self.writer, "[Black \"{}\"]", game.black)?;

        let result_str = match game.result {
            Outcome::Win => "1-0",
            Outcome::Loss => "0-1",
            Outcome::Draw => "1/2-1/2",
        };

        writeln!(self.writer, "[Result \"{}\"]", result_str)?;

        writeln!(self.writer, "[TimeControl \"{}+{}\"]", game.tc as f32 / 1000.0, game.inc as f32 / 1000.0)?;
        writeln!(self.writer, "[Time \"{}\"]", game.start_time.format("%H:%M:%S"))?;
        writeln!(self.writer, "[FEN \"{}\"]", game.opening)?;

        writeln!(self.writer)?;
        
        if (game.start_move_count as usize) % 2 == 0 {
            write!(self.writer, "{}... ", game.start_move_count / 2)?;
        }

        for (i, mv) in game.moves.iter().enumerate() {
            if i % 2 == 0 {
                write!(self.writer, "{}. ", game.start_move_count as usize + i / 2)?;
            }
            write!(self.writer, "{} ", mv)?;
        }

        writeln!(self.writer, "{}", result_str)?;
        writeln!(self.writer)?;

        anyhow::Ok(())
    }

    pub fn flush(&mut self) -> anyhow::Result<()> {
        self.writer.flush()?;
        anyhow::Ok(())
    }
}

pub struct PgnGame {
    pub start_time: DateTime<Local>,
    pub white: String,
    pub black: String,
    pub tc: i32,
    pub inc: i32,
    pub round: usize,
    pub opening: String,
    pub start_move_count: u16,
    pub result: Outcome,
    pub moves: Vec<String>,
}

impl PgnGame {
    pub fn new(white: String, black: String, tc: i32, inc: i32, round: usize, opening: String, start_move_count: u16) -> PgnGame {
        PgnGame {
            start_time: Local::now(),
            white,
            black,
            tc,
            inc,
            round,
            opening,
            start_move_count,
            result: Outcome::Draw,
            moves: Vec::new(),
        }
    }

    pub fn add_move(&mut self, mv: &str) {
        self.moves.push(mv.to_string());
    }

    pub fn set_result(&mut self, result: Outcome) {
        self.result = result;
    }
}