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

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::exit;
use std::sync::{Arc, mpsc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc::{Sender};
use std::{fs, thread};
use std::str::FromStr;
use std::time::{Instant};
use glob::{glob};
use itertools::{Itertools};
use uuid::Uuid;
use velvet::engine::{LogLevel, Message};
use velvet::fen::{create_from_fen, read_fen, START_POS};
use velvet::moves::NO_MOVE;
use velvet::nn::init_nn_params;
use velvet::pieces::EMPTY;
use velvet::search::{Search};
use velvet::syzygy;
use velvet::time_management::SearchLimits;
use velvet::transposition_table::TranspositionTable;
use crate::{RescoreArgs, Command};

pub fn rescore(args: RescoreArgs) {
    let tb_path = args.tb_path;

    if !syzygy::tb::init(tb_path.clone()) {
        eprintln!("could not initialize tablebases using path: {}", tb_path.as_str());
        exit(1);
    } else {
        let count = syzygy::tb::max_piece_count();
        if count == 0 {
            println!("debug no tablebases found");
        } else {
            println!("debug found {}-men tablebases", syzygy::tb::max_piece_count());
        }
    }

    let concurrency = args.concurrency;

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let fen_source  = Arc::new(Mutex::new(FenFileSource::new(args.input_pattern).expect("could not initialize fen source")));
    let local_fen_source = fen_source.clone();

    let (tx, rx) = mpsc::channel::<Command>();

    fs::create_dir_all("./data/out/wip").expect("could not create wip folder");
    fs::create_dir_all("./data/out/done").expect("could not create wip folder");

    for entry in glob("./data/out/wip/*.fen").expect("could not search files in wip folder") {
        match entry {
            Ok(path) => {
                fs::rename(&path, format!("./data/out/done/{}", path.file_name().unwrap().to_string_lossy())).expect("could not move wip file");
            }
            Err(e) => {
                println!("Error: {:?}", e)
            }
        }
    }

    init_nn_params();

    println!("Starting worker threads");
    spawn_threads(&tx, concurrency, fen_source);


    println!("Creating output writers");
    let mut writers = vec![
        OutputWriter::new("1000"),
        OutputWriter::new("2000"),
        OutputWriter::new("3000"),
        OutputWriter::new("4000"),
    ];

    let mut count = 0;
    let mut sub_count = 0;

    println!("Setting CTRL-C handler");

    ctrlc::set_handler(move || tx.send(Command::Terminate).expect("Could not send signal on channel."))
        .expect("Error setting Ctrl-C handler");

    let start = Instant::now();
    let mut start_batch = Instant::now();

    println!("Waiting for generated test positions ...");
    for cmd in rx {
        match cmd {
            Command::AddTestPos(fen, score) => {
                writers[score.unsigned_abs() as usize / 1000].add(fen, score);
                sub_count += 1;
                count += 1;
                if sub_count >= 10_000 {
                    let batch_duration_secs = Instant::now().duration_since(start_batch).as_millis() as f64 / 1000.0;
                    if batch_duration_secs > 0.0 {
                        let batch_per_minute = (sub_count as f64 / batch_duration_secs) * 60.0;
                        let duration_secs = Instant::now().duration_since(start).as_millis() as f64 / 1000.0;
                        let per_minute = (count as f64 / duration_secs) * 60.0;

                        println!("- generated {} test positions (curr. {:.2} per minute / avg. {:.2} per minute)", count, batch_per_minute, per_minute);
                    }
                    start_batch = Instant::now();
                    sub_count = 0;
                }
            }

            Command::Terminate => {
                println!("Stopping rescoring ...");
                local_fen_source.lock().map(|mut f| f.terminate()).expect("could not terminate fen file source");
                for writer in writers.iter_mut() {
                    writer.terminate();
                }
                break;
            }
        }
    }

    println!("End");
}

fn spawn_threads(tx: &Sender<Command>, concurrency: usize, fen_source: Arc<Mutex<FenFileSource>>) {
    for _ in 0..concurrency {
        let tx2 = tx.clone();
        let fen_source2 = fen_source.clone();
        thread::spawn(move || {
            rescore_test_positions(&tx2, fen_source2);
        });
    }
}


fn rescore_test_positions(tx: &Sender<Command>, fen_source: Arc<Mutex<FenFileSource>>) {
    let tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    let limits = SearchLimits::new(None, Some(10), None, None, None, None, None, None, None).unwrap();
    let mut search =
        Search::new(stop, Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)), LogLevel::Error, limits, tt, create_from_fen(START_POS), false);

    search.set_tb_probe_root(false);

    let (_tx, rx) = mpsc::channel::<Message>();

    let mut line = String::with_capacity(256);
    while let Some(input_fen_path) = fen_source.lock().map(|mut f| f.next()).expect("could not lock fen_source") {
        let id = Uuid::new_v4().to_string();
        let wip_file = format!("./data/out/wip/{}.fen", id);
        fs::rename(input_fen_path, &wip_file).expect("could not move to wip folder");

        let file = File::open(&wip_file).expect("Could not open input position file");
        let mut reader = BufReader::new(file);

        loop {
            line.clear();

            match reader.read_line(&mut line) {
                Ok(read) => {
                    if read == 0 {
                        break;
                    }
                }

                Err(e) => panic!("Reading test position file failed: {}", e),
            };

            let parts = line.trim_end().split(' ').collect_vec();
            let fen: String = (parts[0..=5].join(" ") as String).replace('~', "");

            read_fen(&mut search.board, &fen).unwrap();

            search.set_node_limit(1000);
            let (selected_move, _) = search.find_best_move(Some(&rx), 8, &[]);
            if selected_move == NO_MOVE || selected_move.is_capture() {
                continue;
            }

            let score = search.board.active_player().score(selected_move.score());
            let (_, removed_piece_id) = search.board.perform_move(selected_move);
            if removed_piece_id != EMPTY {
                continue;
            }
            if score.abs() < 4000 {
                tx.send(Command::AddTestPos(fen, score)).expect("could not send test position")
            }
        }
        let done_file = format!("./data/out/done/{}.fen", id);
        fs::rename(&wip_file, done_file).expect("could not move file to done folder");
    }

    println!("Rescore thread stopped");
}

struct FenFileSource {
    input_pattern: String,
    paths: Vec<Box<Path>>,
    terminated: bool
}

impl FenFileSource {
    pub fn new(input_pattern: String) -> anyhow::Result<Self> {
        Ok(FenFileSource{paths: Vec::with_capacity(10000), input_pattern, terminated: false})
    }

    pub fn next(&mut self) -> Option<Box<Path>> {
        if self.terminated {
            return None;
        }
        let mut next_path = self.paths.pop();
        if next_path.is_none() {
            self.update();
            next_path = self.paths.pop();
            if next_path.is_none() {
                self.terminated = true;
            }
        }
        next_path
    }

    pub fn terminate(&mut self) {
        self.paths.clear();
    }

    fn update(&mut self) {
        if !self.terminated && !self.paths.is_empty() {
            return;
        }
        for entry in glob(&self.input_pattern).expect("could not search for input files") {
            match entry {
                Ok(path) => {
                    self.paths.push(path.into_boxed_path());
                }
                Err(e) => {
                    println!("Error: {:?}", e)
                }
            }
        }
    }
}

struct OutputWriter {
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

    pub fn add(&mut self, fen: String, score: i32) {
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
