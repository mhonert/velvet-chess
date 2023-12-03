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
use std::fs::{File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use glob::glob;
use velvet::nn::io::{write_i16, write_u16, write_u32, write_u8};

const VERSION: u8 = 1;

pub struct NextIDSource(AtomicUsize);

impl NextIDSource {
    pub fn new() -> Self {
        let output_dir = "./data/out/";
        fs::create_dir_all(output_dir).expect("could not create output folder");

        let mut max_num = 0;
        for entry in glob(&format!("{}/*.bin", &output_dir)).expect("could not glob fen files from output dir") {
            match entry {
                Ok(path) => {
                    let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                    let num_str = file_name.strip_prefix("test_pos_").unwrap().strip_suffix(".bin").unwrap();
                    let num = usize::from_str(num_str).expect("could not extract set count from file name");
                    max_num = max_num.max(num);
                }
                Err(e) => {
                    eprintln!("Error: {:?}", e)
                }
            }
        }

        println!("Next ID: {}", max_num + 1);
        NextIDSource(AtomicUsize::new(max_num + 1))
    }
}

pub struct OutputWriter {
    id_source: Arc<NextIDSource>,
    output_dir: String,
    pos_count: usize,
    writer: BufWriter<File>,
}

impl OutputWriter {
    pub fn new(id_source: Arc<NextIDSource>) -> Self {
        let output_dir = "./data/out/";

        let next_set = id_source.0.fetch_add(1, Ordering::Relaxed);
        let writer = next_file(output_dir, next_set);
        OutputWriter{id_source, output_dir: String::from(output_dir), pos_count: 0, writer}
    }

    pub fn add(&mut self, fen: String, result: i16, moves: Vec<u32>) {
        write_u8(&mut self.writer, fen.len() as u8).expect("could not write FEN len");
        write!(&mut self.writer, "{}", fen).expect("could not write FEN");
        write_i16(&mut self.writer, result).expect("could not write result");
        write_u16(&mut self.writer, moves.len() as u16).expect("could not write move count");
        for m in moves.iter() {
            write_u32(&mut self.writer, *m).expect("could not write move");
        }

        self.pos_count += moves.len();

        if self.pos_count >= 200_000 {
            self.pos_count = 0;
            let next_set = self.id_source.0.fetch_add(1, Ordering::Relaxed);
            self.writer = next_file(&self.output_dir, next_set);
        }
    }

    pub fn terminate(&mut self) {
        self.writer.flush().expect("could not flush output");
    }
}

fn next_file(path: &str, set_nr: usize) -> BufWriter<File> {
    let file_name = format!("{}/test_pos_{}.bin", path, set_nr);
    if Path::new(&file_name).exists() {
        panic!("Output file already exists: {}", file_name);
    }
    let file = File::create(&file_name).expect("Could not create output file");
    let mut w = BufWriter::new(file);
    write_u8(&mut w, VERSION).expect("could not write version");

    w
}
