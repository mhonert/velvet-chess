/*
 * Velvet Chess Engine
 * Copyright (C) 2023 mhonert (https://github.com/mhonert)
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

use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use glob::glob;

pub struct OutputWriter {
    output_dir: String,
    set_count: usize,
    pos_count: usize,
    writer: BufWriter<File>,
}

impl OutputWriter {
    pub fn new(suffix: &str) -> Self {
        println!("Creating output writer {} ...", suffix);
        let output_dir = format!("./data/out/{}", suffix);
        fs::create_dir_all(&output_dir).expect("could not create output folder");

        let mut max_num = 0;
        for entry in glob(&format!("{}/*.fen", &output_dir)).expect("could not glob fen files from output dir") {
            match entry {
                Ok(path) => {
                    let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                    let num_str = file_name.strip_prefix("test_pos_").unwrap().strip_suffix(".fen").unwrap();
                    let num = usize::from_str(num_str).expect("could not extract set count from file name");
                    max_num = max_num.max(num);
                }
                Err(e) => {
                    eprintln!("Error: {:?}", e)
                }
            }
        }
        let existing_pos_count = if max_num > 0 { Some(pos_count(&output_dir, max_num)) } else { None };
        if existing_pos_count.filter(|&c| c < 200_000).is_some() {
            println!("Continue {} - {}", suffix, max_num);
            let writer = continue_file(&output_dir, max_num);
            OutputWriter{output_dir, set_count: max_num, pos_count: existing_pos_count.unwrap(), writer}

        } else {
            println!("New {} - {}", suffix, max_num);
            let writer = next_file(&output_dir, max_num + 1);
            OutputWriter{output_dir, set_count: max_num + 1, pos_count: 0, writer}
        }
    }

    pub fn add(&mut self, fen: String, score: i16) {
        writeln!(&mut self.writer, "{} {}", fen, score).expect("Could not write position to file");
        self.pos_count += 1;

        if self.pos_count % 200_000 == 0 {
            self.pos_count = 0;
            self.set_count += 1;
            self.writer = next_file(&self.output_dir, self.set_count);
        }
    }

    pub fn terminate(&mut self) {
        self.writer.flush().expect("could not flush output");
    }
}

fn pos_count(path: &str, set_nr: usize) -> usize {
    let file_name = format!("{}/test_pos_{}.fen", path, set_nr);
    let file = File::open(file_name).expect("Could not open output file");
    let mut reader = BufReader::new(file);
    let mut buf = String::new();

    let mut count = 0;
    while let Ok(read) = reader.read_line(&mut buf) {
        if read == 0 {
            break;
        }
        count += 1;
    }

    count
}

fn next_file(path: &str, set_nr: usize) -> BufWriter<File> {
    let file_name = format!("{}/test_pos_{}.fen", path, set_nr);
    if Path::new(&file_name).exists() {
        panic!("Output file already exists: {}", file_name);
    }
    let file = File::create(&file_name).expect("Could not create output file");
    BufWriter::new(file)
}

fn continue_file(path: &str, set_nr: usize) -> BufWriter<File> {
    let file_name = format!("{}/test_pos_{}.fen", path, set_nr);
    println!("Continuing with output file {}", &file_name);
    let file = OpenOptions::new().append(true).open(&file_name).expect("could not open existing output file in append mode");
    BufWriter::new(file)
}
