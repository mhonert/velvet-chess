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

use velvet::random::Random;
use velvet::genetic_eval::{GeneticProgram, next_gen, compile, generate_program, optimize};
use std::time::{SystemTime, UNIX_EPOCH};
use std::cmp::{min, max};

fn main() {
    let duration = match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(e) => panic!("Duration time error: {}", e)
    };
    let mut rnd = Random::new_with_seed((duration.as_micros() & 0xFFFFFFFFFFFFFFFF) as u64);

    let pop_size = 8192;

    let mut generation: Vec<GeneticProgram> = Vec::new();
    let mut scores = vec!(0; pop_size);
    for i in 0..pop_size {
        let mut program = generate_program(&mut rnd);
        program.score_increment = i as i32;
        generation.push(program);
    }

    let mut j = 1;
    loop {
        for i in 0..pop_size {
            let program = generation[i];
            let score = eval(&mut program.clone());
            scores[i] = score;
        }

        generation.sort_by_key(|p| p.instr_count());
        generation.sort_by_key(|p| scores[p.score_increment as usize]);

        let best = generation[0];
        let constants: Vec<&u64> = best.data.iter().skip(4).collect();
        println!("Gen {}: Best: {} ({} instr.) - [{}]- {:?}", j, scores[best.score_increment as usize], best.instr_count(), best.code, constants);
        compile(&best);

        generation = next_gen(&mut rnd, &generation);
        for i in 0..pop_size {
            generation[i].score_increment = i as i32;
        }

        j += 1;
    }
}

fn eval(program: &mut GeneticProgram) -> u64 {
    let mut diffs: u64 = 0;

    let mut opt_program = optimize(program);

    for a in 0..64 {
        for b in 0..64 {
            let diff = calc_diff(&mut opt_program, a, b);
            diffs = diffs.saturating_add(diff);
        }
    }
    for b in 0..64 {
        for a in 0..64 {
            let diff = calc_diff(&mut opt_program, a, b);
            diffs = diffs.saturating_add(diff);
        }
    }

    diffs
}

fn calc_diff(opt_program: &mut GeneticProgram, a: u64, b: u64) -> u64 {

    // let expected_result = (a * a * 97) | (b * 500);
    let expected_result = 7 * a * a + 50 * b;

    opt_program.update(a, b, 0, 0);

    let actual_result = opt_program.run();

    let higher = max(actual_result, expected_result);
    let lower = min(actual_result, expected_result);

    higher.saturating_sub(lower)
}
