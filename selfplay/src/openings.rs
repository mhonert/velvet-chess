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
use std::io::{BufRead, BufReader};
use rand::prelude::ThreadRng;
use rand::Rng;

pub struct OpeningBook(Vec<String>);

impl OpeningBook {
    pub fn new(file_name: &str) -> OpeningBook {
        let file = File::open(file_name).expect("Could not open book file");
        let mut reader = BufReader::new(file);

        let mut openings = Vec::new();

        loop {
            let mut line = String::new();

            match reader.read_line(&mut line) {
                Ok(read) => if read == 0 {
                    return OpeningBook(openings);
                },

                Err(e) => panic!("could not read line from opening book: {}", e)
            };

            openings.push(line.trim().to_string());
        }
    }

    // Get a random opening from the book
    pub fn get_random(&self) -> String {
        let mut rng = ThreadRng::default();
        let idx = rng.gen_range(0..self.0.len());
        self.0[idx].clone()
    }
}

