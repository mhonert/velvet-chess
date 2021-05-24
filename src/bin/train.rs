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
extern crate velvet;

#[cfg(feature = "lgp_training")]
use velvet::genetic_eval_trainer::{GeneticEvaluator, GeneticProgram};

#[cfg(not(feature = "lgp_training"))]
use velvet::genetic_eval::{GeneticEvaluator, GeneticProgram};
use velvet::engine::{Engine, Message};
use velvet::trainer::play_match;
use velvet::magics::initialize_magics;
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use velvet::random::Random;
use std::io::{BufReader, Read, BufRead, BufWriter, Error, Write};
use std::fs::File;
use crossbeam_queue::{ArrayQueue};
use crossbeam_utils::thread;
use crossbeam_channel::{unbounded, Sender};
use std::sync::mpsc;
use std::thread::sleep;
use serde::{Serialize, Deserialize};
use velvet::genetic_eval_trainer::{generate_program, next_gen};
use itertools::Itertools;
use std::path::Path;
use clap::{App};
use std::str::FromStr;
use std::process::exit;
use std::io;

type Team = Vec<GeneticProgram>;

type WorkID = usize;

type WinRate = f64;

type Work = (WorkID, u64, Team);

type Result = (WorkID, WinRate);

#[derive(Serialize, Deserialize)]
struct Community {
    pub gen: usize,
    pub teams: Vec<TeamEntry>
}

#[derive(Serialize, Deserialize)]
struct TeamEntry {
    pub id: usize,
    pub programs: Team,
    pub win_rate: f64,
    pub count: u64,
}

fn main() {
    let matches = App::new("Genetic Eval Trainer")
        .version("0.0.1")
        .about("Evolves a chess evaluation function using a genetic programming approach")
        .args_from_usage(
            "-o, --opening-book=<FILE>              'Sets the opening book file'
             -c, --concurrency=<CONCURRENCY>        'Sets the number of threads'
             -r, --rounds=<ROUNDS>                  'Sets the number of rounds per match-up'
             -t, --team-size=<SIZE>                 'Sets the team size'
             -p, --population-size=<SIZE>           'Sets the population size'")
        .get_matches();

    let book_file = matches.value_of("opening-book").unwrap();

    let concurrency = match matches.value_of("concurrency") {
        Some(v) => usize::from_str(v).expect("Concurrency must be an integer >= 1"),
        None => {
            eprintln!("Missing -c (concurrency) option");
            exit(1);
        }
    };

    if concurrency == 0 {
        eprintln!("-c Concurrency must be >= 1");
        exit(1);
    }
    println!("Running with {} concurrent threads", concurrency);

    let rounds = match matches.value_of("rounds") {
        Some(v) => usize::from_str(v).expect("Rounds must be an integer >= 1"),
        None => 100
    };

    if rounds == 0 {
        eprintln!("-r Rounds must be >= 1");
        exit(1);
    }

    let team_size = match matches.value_of("team-size") {
        Some(v) => usize::from_str(v).expect("Team size must be an integer >= 1"),
        None => 32
    };

    if team_size == 0 {
        eprintln!("-t Teams size must be >= 1");
        exit(1);
    }

    let pop_size = match matches.value_of("population-size") {
        Some(v) => usize::from_str(v).expect("Population size must be a multiple of the concurrency (thread) count"),
        None => 64
    };

    if (pop_size / concurrency) * concurrency != pop_size {
        eprintln!("-p Population size must be a multiple of the concurrency (thread) count");
        exit(1);
    }


    initialize_magics();

    println!("Reading opening book: {} ...", book_file);
    let openings = read_openings(book_file);

    let (tx, rcv) = unbounded::<Result>();
    let queue = ArrayQueue::<Work>::new(pop_size);

    let mut rnd = Random::new_with_seed(new_rnd_seed());

    // Initialize population
    let (community_size, team_size, mut teams) = get_start_population(&mut rnd, pop_size, team_size);

    println!("Population of {} teams with {} members each", community_size, team_size);
    println!("Playing {} rounds per team (= {} games per generation)", rounds, rounds * 2 * community_size);

    println!("Start evolution ...");

    // Start evolution
    thread::scope(|s| {

        for _ in 0..concurrency {
            sleep(Duration::from_millis(1));
            s.spawn(|_| {
                run_worker(&tx, &openings, rounds, &queue);
            });
        };

        let mut gen = 1;
        loop {
            println!("\n--- ({:04}) -----------------------------------------", gen);
            let start = Instant::now();

            let rnd_seed = new_rnd_seed();
            for (i, team) in teams.iter().enumerate() {
                match queue.push((i, rnd_seed, team.programs.clone())) {
                    Ok(_) => (),
                    Err(e) => panic!(e)
                };
            }

            let mut remaining_results = teams.len();

            print!("Results: [|");

            while remaining_results > 0 {
                match rcv.recv() {
                    Ok((id, win_rate)) => {
                        print!(" {}={:3.1}% |", id, win_rate * 100.0);
                        io::stdout().flush().unwrap();
                        teams[id].win_rate = win_rate;
                        remaining_results -= 1;
                    },

                    Err(e) => {
                        panic!(e);
                    }
                }
            }

            println!("]");

            // Sort by win rate descending
            teams.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());

            println!();
            print_team_stats(&teams[0]);
            let avg_win_rate = teams.iter().map(|t| t.win_rate).sum::<f64>() / (teams.len() as f64);
            let med_win_rate = teams[teams.len() / 2].win_rate;

            let top_90percentile = teams.len() - teams.len() / 10;
            let top_90percentile_win_rate = teams.iter().take(top_90percentile).map(|t| t.win_rate).sum::<f64>() / (top_90percentile as f64);
            println!("Community Stats: [avg win rate = {:3.2}%] [90% win rate = {:3.2}%] [med win rate = {:3.2}%]", avg_win_rate * 100.0, top_90percentile_win_rate * 100.0, med_win_rate * 100.0);

            backup_generation(&teams);
            write_rust_eval_fn(&teams[0].programs);

            // Create next generation
            let mut team_member_programs = Vec::with_capacity(team_size);
            for i in 0..team_size {
                let programs = teams.iter().map(|t| t.programs[i]).collect_vec();
                let child_programs = next_gen(&mut rnd, &programs);
                team_member_programs.push(child_programs);
            }

            teams.clear();
            for c in 0..community_size {
                let programs = team_member_programs.iter().map(|m| m[c]).collect_vec();
                teams.push(TeamEntry{id: c + 1000, programs, win_rate: 0.0, count: 0});
            }

            let duration = Instant::now().duration_since(start);
            println!("Duration: {:?}", duration);

            gen += 1;
        }

    }).unwrap();
}

fn new_rnd_seed() -> u64 {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    (duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64
}

fn print_team_stats(team: &TeamEntry) {
    let code_size: usize = team.programs.iter().map(|p| p.instr_count()).sum();
    println!("Best Team Stats: [win rate = {:3.1}%] [code size = {} instr.]", team.win_rate * 100.0, code_size);
}

fn get_start_population(rnd: &mut Random, init_pop_size: usize, init_team_size: usize) -> (usize, usize, Vec<TeamEntry>) {
    if Path::new("./community.bin").exists() {
        println!("Continue with existing population ... ");
        let community = restore_generation();
        let community_size = community.len();
        let team_size = community[0].programs.len();

        return (community_size, team_size, community);
    }

    println!("Initialize new population ... ");

    let team_size = init_team_size;
    let community_size = init_pop_size;

    let mut teams = Vec::with_capacity(community_size);

    for i in 0..community_size {
        let mut programs = Vec::with_capacity(team_size);
        for _ in 0..team_size {
            programs.push(generate_program(rnd));
        }
        teams.push(TeamEntry{id: i + 1000, programs, win_rate: 0.0, count: 0});
    }

    (community_size, team_size, teams)
}

fn backup_generation(teams: &[TeamEntry]) {
    let file = File::create("./community.bin").unwrap();
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, teams).unwrap();
}

fn restore_generation() -> Vec<TeamEntry> {
    let file = File::open("./community.bin").unwrap();
    let reader = BufReader::new(file);

    bincode::deserialize_from(reader).unwrap()
}

fn run_worker(tx: &Sender<Result>, openings: &[String], rounds: usize, queue: &ArrayQueue<Work>) {
    let (_, rx1) = mpsc::channel::<Message>();
    let (_, rx2) = mpsc::channel::<Message>();

    let genetic_eval1 = GeneticEvaluator::new();
    let genetic_eval2 = GeneticEvaluator::new();

    let mut engine1 = Engine::new(rx1, genetic_eval1);
    let mut engine2 = Engine::new(rx2, genetic_eval2);

    loop {
        let (id, rnd_seed, team) = match queue.pop() {
            Some(w) => w,
            None => {
                sleep(Duration::from_millis(1));
                continue;
            }
        };

        let mut rnd = Random::new_with_seed(rnd_seed);

        // println!("Processing work item: {}", id);

        engine1.genetic_eval.clear();
        for p in team.iter() {
            engine1.genetic_eval.add_program(*p);
        }
        engine1.genetic_eval.compile();


        let mut win_rates = 0.0;
        for _ in 0..rounds {
            let opening = &openings[(rnd.rand32() as usize) % openings.len()];
            win_rates += check_teams(opening, &mut engine1, &mut engine2);
        }

        let win_rate = win_rates / (rounds as f64);

        // println!("{}: Played {} matches", id, 50 * 2 * opponents.len());

        tx.send((id, win_rate)).unwrap();
    }
}

fn check_teams(opening: &str, engine1: &mut Engine, engine2: &mut Engine) -> f64 {

    let mut results = 0.0;

    results += match play_match(opening, &mut [engine1, engine2]) {
        Some(winner) => if winner == 0 { 1.0 } else { 0.0 }
        None => 0.5
    };

    results += match play_match(opening, &mut [engine2, engine1]) {
        Some(winner) => if winner == 1 { 1.0 } else { 0.0 }
        None => 0.5
    };

    results / 2.0
}

fn read_openings(file_name: &str) -> Vec<String> {
    let file = File::open(file_name).expect("Could not open book file");
    let mut reader = BufReader::new(file);

    let mut openings = Vec::new();

    loop {
        let mut line = String::new();

        match reader.read_line(&mut line) {
            Ok(read) => if read == 0 {
                return openings;
            },

            Err(e) => panic!(e)
        };

        openings.push(line);
    }
}

const HEADER: &str = r#"
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

// Auto-generated file (see src/bin/gen_eval.rs)

use crate::random::Random;

pub struct GeneticEvaluator {}

impl GeneticEvaluator {

    pub fn new() -> Self {
        GeneticEvaluator{}
    }

    pub fn compile(&mut self) {}
"#;

const FOOTER: &str = r#"

    pub fn add_program(&mut self, _: GeneticProgram) {}

    pub fn clear(&mut self) {}

    pub fn init_generation(&self, _: &mut Random, _: u32) {}

    pub fn create_new_generation(&self, _: &mut Random) {}

    pub fn print_rust_code(&self) {}
}

#[derive(Copy, Clone)]
#[allow(unused_mut)]
pub struct GeneticProgram {}

impl GeneticProgram {
    pub fn new_from_str(_: &str, _: [u64; 8], _: i32, _: i32) -> GeneticProgram {
        GeneticProgram{}
    }
}
"#;

fn write_rust_eval_fn(programs: &Team) {
    let mut eval = GeneticEvaluator::new();
    programs.iter().for_each(|p| eval.add_program(*p));

    let file: Box<dyn Write> = Box::new(File::create("./genetic_eval.rs").expect("Could not create output file"));
    let mut writer = BufWriter::new(file);

    writeln!(writer, "{}", HEADER).expect("could not write file");
    eval.write_rust_code(&mut writer).expect("could not write file");
    writeln!(writer, "{}", FOOTER).expect("could not write file");
}