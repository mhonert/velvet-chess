/*
 * Velvet Chess Engine
 * Copyright (C) 2020 mhonert (https://github.com/mhonert)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation,either version 3 of the License, or
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

use std::mem::transmute;
use crate::genetic_eval::Instruction::{Xor, And, Not, Or, Mul, Add};
use crate::random::Random;
use std::cmp::{min, max};
use crate::genetic_eval::Term::{Constant, Variable, UnaryOp, BinaryOp};
use std::collections::HashMap;

const CROSS_OVER_RATE: u32 = 90;
const MUTATION_RATE: u32 = 10;
const ELITE_COUNT: u32 = 1;

const INSTR_SIZE: usize = 4;
const INSTR_LIMIT: usize = 128 / INSTR_SIZE;

// const BINARY_OPS: [Instruction; 2] = [Mul, Add];
const BINARY_OPS: [Instruction; 3] = [Mul, Or, And];

pub struct GeneticEvaluator {
    programs: Vec<GeneticProgram>,
    opt_programs: Vec<GeneticProgram>
}

impl GeneticEvaluator {
    pub fn new() -> Self {
        GeneticEvaluator{
            programs: Vec::new(),
            opt_programs: Vec::new()
        }
    }

    pub fn eval(&mut self, own_pawns: u64, opp_pawns: u64, own_king_half: u64, opp_king_half: u64) -> i32 {
        let mut value = 0;
        for program in self.opt_programs.iter_mut() {
            program.update(own_pawns, opp_pawns, own_king_half, opp_king_half);
            let result_bb = program.run();

            let count = result_bb.count_ones() as i32;
            let score = if program.score_raise > 0 {
                count * program.score_increment * (count + 1) / 2 * program.score_raise / 32
            }  else {
                count * program.score_increment
            };

            value += score;
        }

        value
    }

    pub fn add_program(&mut self, program: GeneticProgram) {
        self.programs.push(program);
        self.opt_programs.push(optimize(&program));
    }

    pub fn clear(&mut self) {
        self.programs.clear();
        self.opt_programs.clear();
    }

    pub fn init_generation(&self, rnd: &mut Random, pop_size: u32) {
        let mut programs = Vec::new();
        for _ in 0..pop_size {
            let program = generate_program(rnd);
            programs.push(program);
        }

        print_generation(&programs);
    }

    pub fn create_new_generation(&self, rnd: &mut Random) {
        let new_programs = next_gen(rnd, &self.programs);
        print_generation(&new_programs);
    }
}

fn print_generation(programs: &[GeneticProgram]) {
    print!("result ");
    for program in programs.iter() {
        let solution_size = program.instr_count();
        print!("{},{},{},{},{};",
               program.code,
               program.data.iter()
                   .skip(4)
                   .map(|&n| format!("{}", n))
                   .collect::<Vec<String>>()
                   .join(","),
               program.score_increment,
               program.score_raise,
               solution_size);
    }
    println!();
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum Instruction {
    Load0 = 0,
    Load1,
    Load2,
    Load3,
    Load4,
    Load5,
    Load6,
    Load7,
    Load8,
    Load9,
    Not,
    And,
    Or,
    Xor,
    Mul,
    Add,
}

const INSTR: u8 = 10;
const LOAD_MAX_INDEX: usize = 9;

impl Instruction {
    pub fn run_two_op(&self, op1: u64, op2: u64) -> u64 {
        match self {
            And => op1 & op2,
            Add => op1 + op2,
            Or => op1 | op2,
            Xor => op1 ^ op2,
            Mul => op1 * op2,
            _ => panic!("{:?} is not a 2-operand instruction", self)
        }
    }
}

#[derive(Copy, Clone)]
pub struct GeneticProgram {
    pub code: u128,
    pub data: [u64; 10],
    pub score_increment: i32,
    pub score_raise: i32,
}

impl GeneticProgram {
    pub fn new(code: u128, data: [u64; 10], score_increment: i32, score_raise: i32) -> GeneticProgram {
        GeneticProgram {
            code,
            data,
            score_increment,
            score_raise
        }
    }

    pub fn update(&mut self, v1: u64, v2: u64, v3: u64, v4: u64) {
        self.data[0] = v1;
        self.data[1] = v2;
        self.data[2] = v3;
        self.data[3] = v4;
    }

    pub fn instr_count(&self) -> usize {
        ((128 - self.code.leading_zeros()) as usize) / INSTR_SIZE
    }

    pub fn run(&self) -> u64 {
        let mut stack: [u64; 32] = [0; 32];
        let mut sp : usize = 0;

        let mut code = self.code;

        while code != 0 {
            let opcode = (code & 0b1111) as u8;
            code >>= 4;
            // Execute instruction
            match unsafe { transmute(opcode) } {
                Not => {
                    unsafe {
                        let v = stack.get_unchecked_mut(sp - 1);
                        *v = !*v;
                    }
                },

                And => {
                    unsafe {
                        *stack.get_unchecked_mut(sp - 2) &= *stack.get_unchecked(sp - 1);
                    }
                    sp -= 1;
                },

                Add => {
                    unsafe {
                        *stack.get_unchecked_mut(sp - 2) += *stack.get_unchecked(sp - 1);
                    }
                    sp -= 1;
                },

                Or => {
                    unsafe {
                        *stack.get_unchecked_mut(sp - 2) |= *stack.get_unchecked(sp - 1);
                    }
                    sp -= 1;
                },

                Xor => {
                    unsafe {
                        *stack.get_unchecked_mut(sp - 2) ^= *stack.get_unchecked(sp - 1);
                    }
                    sp -= 1;
                },

                Mul => {
                    unsafe {
                        *stack.get_unchecked_mut(sp - 2) *= *stack.get_unchecked(sp - 1);
                    }
                    sp -= 1;
                },

                _ => {
                    // Load data
                    unsafe {
                        let v = *self.data.get_unchecked(opcode as usize);
                        *stack.get_unchecked_mut(sp) = v;
                    }
                    sp += 1;
                }
            }
        }

        if sp > 0 {
            stack[sp - 1]
        } else {
            0
        }
    }

    pub fn disassemble(&self) {
        println!("debug GenEval Code: {}", self.code);
        println!("debug GenEval Data: [{}]", self.data.iter()
            .map(|&n| format!("{:016x}", n))
            .collect::<Vec<String>>()
            .join(", "));

        println!("debug GenEval Disassembled Code:");

        let mut code = self.code;

        let mut num = 0;
        while code != 0 {
            let opcode = (code & 0b1111) as u8;
            code >>= 4;
            if opcode >= INSTR {
                let instruction: Instruction = unsafe { transmute(opcode) };
                println!("{:03}: {:?}", num, instruction);
            } else {
                // Load data
                let slot = opcode;
                println!("{:03}: Load{}", num, slot);
            }
            num += 1;
        }
    }


    pub fn count(&self, searched_instr: Instruction) -> i32 {
        let searched_opcode = searched_instr as u8;
        let mut count = 0;
        let mut code = self.code;
        while code != 0 {
            let opcode = (code & 0b1111) as u8;
            code >>= 4;
            if opcode == searched_opcode {
                count += 1;
            }
        }

        count
    }
}

#[derive(Clone)]
enum Term {
    Constant(u64),
    Variable(usize),
    UnaryOp(Instruction, Box<Term>),
    BinaryOp(Instruction, Box<Term>, Box<Term>),
}

//const VAR_NAMES: [&str; 4] = ["a", "b", "c", "d"];
const VAR_NAMES: [&str; 4] = ["own_pawns", "opp_pawns", "own_king_half", "opp_king_half"];

impl Term {
    pub fn to_expr_str(&self) -> String {
        match self {
            Variable(slot) => String::from(VAR_NAMES[*slot]),
            // Constant(value) => format!("{:016x}", &value),
            Constant(value) => format!("{}", &value),
            UnaryOp(instr, op) => {
                match *instr {
                    Not => format!("!{}", op.to_expr_str()),
                    _ => panic!("{:?} is not a one operand instruction", *instr),
                }
            }
            BinaryOp(instr, op1, op2) => {
                match *instr {
                    And => format!("({} & {})", op1.to_expr_str(), op2.to_expr_str()),
                    Add => format!("({} + {})", op1.to_expr_str(), op2.to_expr_str()),
                    Or => format!("({} | {})", op1.to_expr_str(), op2.to_expr_str()),
                    Xor => format!("({} ^ {})", op1.to_expr_str(), op2.to_expr_str()),
                    Mul => format!("({} * {})", op1.to_expr_str(), op2.to_expr_str()),
                    _ => panic!("{:?} is not a two operand instruction", *instr),
                }
            }
        }
    }

    fn optimized(self: &Term) -> Term {
        match self {
            Constant(_) => self.clone(),
            Variable(_) => self.clone(),
            UnaryOp(instr, op) => {
                match instr {
                    Not => {
                        let opt_op = op.optimized();
                        match opt_op {
                            Constant(value) => Constant(!value),
                            _ => self.clone()
                        }
                    },
                    _ => panic!("{:?} is not a 1-operand instruction", instr)
                }
            },
            BinaryOp(instr, op1, op2) => {
                let opt_op1 = op1.optimized();
                let opt_op2 = op2.optimized();
                match instr {
                    And => {
                        match opt_op1 {
                            Constant(value1) => {
                                match opt_op2 {
                                    Constant(value2) => Constant(instr.run_two_op(value1, value2)),
                                    _ => {
                                        if value1 == 0 {
                                            Constant(0)
                                        } else {
                                            BinaryOp(And, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    }
                                }
                            },
                            _ => {
                                match opt_op2 {
                                    Constant(value2) => {
                                        if value2 == 0 {
                                            Constant(0)
                                        } else {
                                            BinaryOp(And, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    },
                                    _ => BinaryOp(And, Box::new(opt_op1), Box::new(opt_op2))
                                }
                            }
                        }
                    },

                    Add => {
                        match opt_op1 {
                            Constant(value1) => {
                                match opt_op2 {
                                    Constant(value2) => Constant(instr.run_two_op(value1, value2)),
                                    _ => {
                                        if value1 == 0 {
                                            opt_op2
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    }
                                }
                            },
                            _ => {
                                match opt_op2 {
                                    Constant(value2) => {
                                        if value2 == 0 {
                                            opt_op1
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    },
                                    _ => BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                }
                            }
                        }
                    }

                    Or | Xor => {
                        match opt_op1 {
                            Constant(value1) => {
                                match opt_op2 {
                                    Constant(value2) => Constant(instr.run_two_op(value1, value2)),
                                    _ => {
                                        if value1 == 0 {
                                            opt_op2
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    }
                                }
                            },
                            _ => {
                                match opt_op2 {
                                    Constant(value2) => {
                                        if value2 == 0 {
                                            opt_op1
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    },
                                    _ => BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                }
                            }
                        }
                    },

                    Mul => {
                        match opt_op1 {
                            Constant(value1) => {
                                match opt_op2 {
                                    Constant(value2) => Constant(instr.run_two_op(value1, value2)),
                                    _ => {
                                        if value1 == 0 {
                                            Constant(0)
                                        } else if value1 == 1 {
                                            opt_op2
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    }
                                }
                            },
                            _ => {
                                match opt_op2 {
                                    Constant(value2) => {
                                        if value2 == 0 {
                                            Constant(0)
                                        } else if value2 == 1 {
                                            opt_op1
                                        } else {
                                            BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                        }
                                    },
                                    _ => BinaryOp(*instr, Box::new(opt_op1), Box::new(opt_op2))
                                }
                            }
                        }
                    },

                    _ => self.clone()
                }
            },
        }
    }
}

pub fn compile(program: &GeneticProgram) {
    let term = parse(&program).optimized();

    // println!("let count = {}.count_ones() as i32;", stack[sp - 1]);
    // println!("let score = count * (count + 1) / 2 * {} / 32", program.score_adjustment);
    println!("let result = {}", term.to_expr_str());
}

fn parse(program: &GeneticProgram) -> Term {
    let mut stack: Vec<Term> = Vec::new();
    for _ in 0..32 {
        stack.push(Constant(0));
    }

    let mut sp : usize = 0;

    let mut code = sanitize_code(program.code);

    while code != 0 {
        let opcode = (code & 0b1111) as u8;
        code >>= 4;
        let instr = unsafe { transmute(opcode) };
        match instr {
            Not => {
                stack[sp - 1] = UnaryOp(Not, Box::new(stack[sp - 1].clone()));
            },

            Or | Xor | And | Mul | Add => {
                stack[sp - 2] = BinaryOp(instr, Box::new(stack[sp - 2].clone()), Box::new(stack[sp - 1].clone()));
                sp -= 1;
            },

            _ => {
                // Data
                let slot = opcode as usize;
                if slot < VAR_NAMES.len() {
                    stack[sp] = Variable(slot);
                } else {
                    stack[sp] = Constant(program.data[slot]);
                }

                sp += 1;
            }
        }
    }

    if sp > 0 {
        stack[sp - 1].clone()
    } else {
        Constant(0)
    }
}

pub fn optimize(program: &GeneticProgram) -> GeneticProgram {
    let term = parse(&program).optimized();
    OpcodeGenerator::new().generate(&term, program.score_increment, program.score_raise)
        .unwrap_or_else(|| sanitize(&program))
}

struct OpcodeGenerator {
    slot_by_constant: HashMap<u64, usize>,
    code: u128,
    index: usize,
    valid: bool,
}

impl OpcodeGenerator {
    pub fn new() -> Self {
        OpcodeGenerator{
            slot_by_constant: HashMap::new(),
            code: 0,
            index: 0,
            valid: true
        }
    }

    pub fn generate(&mut self, term: &Term, score_increment: i32, score_raise: i32) -> Option<GeneticProgram> {
        self.slot_by_constant.clear();
        self.index = 0;
        self.valid = true;

        self.process(term);

        let mut data = [0; 10];
        self.slot_by_constant.iter().for_each(|(&value, &slot)| data[slot] = value);

        if self.valid {
            Some(GeneticProgram::new(self.code, data, score_increment, score_raise))
        } else {
            None
        }
    }

    fn process(&mut self, term: &Term) {
        match term {
            Constant(value) => {
                let slot = self.register_constant(*value);
                self.emit_opcode(slot as u8);
            },

            Variable(slot) => {
                self.emit_opcode(*slot as u8);
            },

            UnaryOp(instr, op) => {
                self.process(op);
                self.emit_opcode(*instr as u8);
            },

            BinaryOp(instr, op1, op2) => {
                self.process(op1);
                self.process(op2);
                self.emit_opcode(*instr as u8);
            }
        }
    }

    fn emit_opcode(&mut self, opcode: u8) {
        self.code |= (opcode as u128) << self.index;
        self.index += 4;
    }

    fn register_constant(&mut self, value: u64) -> usize {
        match self.slot_by_constant.get(&value) {
            Some(&slot) => slot,
            None => {
                let slot = self.slot_by_constant.len() + VAR_NAMES.len();
                if slot >= 10 {
                    self.valid = false;
                    return slot - 1;
                }
                self.slot_by_constant.insert(value, slot);
                slot
            }
        }
    }
}

pub struct Assembler(u128, usize);

impl Assembler {
    pub fn new() -> Self {
        Assembler(0, 0)
    }

    pub fn load(&mut self, slot: u8) -> &mut Self {
        if slot > 9 {
            panic!("Slot must be in the range 0-9")
        }

        self.add_opcode(slot)
    }

    pub fn not(&mut self) -> &mut Self {
        self.add_opcode(Not as u8)
    }

    pub fn and(&mut self) -> &mut Self {
        self.add_opcode(And as u8)
    }

    pub fn add(&mut self) -> &mut Self {
        self.add_opcode(Add as u8)
    }

    pub fn or(&mut self) -> &mut Self {
        self.add_opcode(Or as u8)
    }

    pub fn xor(&mut self) -> &mut Self {
        self.add_opcode(Xor as u8)
    }

    pub fn mul(&mut self) -> &mut Self {
        self.add_opcode(Mul as u8)
    }

    fn add_opcode(&mut self, opcode: u8) -> &mut Self {
        let bitpos = self.1;
        self.0 |= (opcode as u128) << bitpos;
        self.1 += 4;
        self
    }

    #[allow(dead_code)]
    fn build_code(&self) -> u128 {
        self.0
    }
}

// next_gen generates a new generation using cross-over and mutation
// curr_gen must be sorted by decreasing fitness
pub fn next_gen(rnd: &mut Random, curr_gen: &[GeneticProgram]) -> Vec<GeneticProgram> {
    let mut new_gen = Vec::with_capacity(curr_gen.len());

    let elites = min(ELITE_COUNT as usize, curr_gen.len());
    for i in 0..elites {
        new_gen.push(curr_gen[i]);
    }

    for _ in 0..(curr_gen.len() - elites) {
        let (p1_rank, parent1) = select(rnd, &curr_gen);
        let next_program = if rnd.rand32() % 100 <= CROSS_OVER_RATE {
            let (p2_rank, parent2) = select(rnd, &curr_gen);

            if p1_rank < p2_rank {
                cross(rnd, &parent1, &parent2)
            } else {
                cross(rnd, &parent2, &parent1)
            }
        } else {
            mutate(rnd, &parent1)
        };

        new_gen.push(next_program);
    }

    new_gen
}

fn select(rnd: &mut Random, parents: &[GeneticProgram]) -> (usize, GeneticProgram) {
    // Tournament selection
    const ROUNDS: usize = 5;

    let mut selection = rnd.rand32() % parents.len() as u32;
    for _ in 0..ROUNDS {
        let opponent = rnd.rand32() % parents.len() as u32;
        selection = min(selection, opponent);
    }

    (selection as usize, parents[selection as usize])
}

// fn select(rnd: &mut Random, parents: &[GeneticProgram]) -> GeneticProgram {
//     loop {
//         for parent in parents.iter() {
//             if rnd.rand32() & 7 == 1 {
//                 return *parent
//             }
//         }
//     }
// }

// fn select(rnd: &mut Random, parents: &[GeneticProgram]) -> GeneticProgram {
//     let median = parents.len() / 2;
//     let mut threshold = median;
//     loop {
//         let index = (rnd.rand32() % parents.len() as u32) as usize;
//         if index < threshold {
//             return parents[index];
//         }
//
//         threshold += max(1, parents.len() / 16);
//     }
// }

fn cross(rnd: &mut Random, parent1: &GeneticProgram, parent2: &GeneticProgram) -> GeneticProgram {
    let p1_slices = slice(&parent1);
    let p2_slices = slice(&parent2);

    let (p1_start, p1_len) = random_slice(rnd, &p1_slices, INSTR_LIMIT / 2);

    let p1_total_length = p1_slices.last().map(|slice| slice.1).unwrap_or(0);
    let max_length = INSTR_LIMIT - (p1_total_length - p1_len);
    let (p2_start, p2_len) = random_slice(rnd, &p2_slices, max_length);

    splice(p1_start, p1_len, p2_start, p2_len, &parent1, &parent2)
}

fn slice(program: &GeneticProgram) -> Vec<(usize, usize)> {
    let mut slices = Vec::new();
    let mut starts = Vec::new();

    let mut code = program.code;
    let mut index = 0;

    while code != 0 {
        let opcode = (code & 0b1111) as u8;
        code >>= 4;
        if opcode < INSTR {
            // Data
            starts.push(index);
            slices.push((index, 1));
        } else {
            // Instruction
            match unsafe { transmute(opcode) } {
                Not => {
                    // Unary operator
                    let start = *starts.last().unwrap_or(&0);
                    let len = (index - start) + 1;
                    slices.push((start, len));
                },

                _ => {
                    // Binary operators
                    starts.pop();
                    let start = *starts.last().unwrap_or(&0);
                    let len = (index - start) + 1;
                    slices.push((start, len));
                }
            }
        }

        index += 1;
    }

    slices
}

fn random_slice(rnd: &mut Random, slices: &[(usize, usize)], max_length: usize) -> (usize, usize) {
    if slices.is_empty() {
        return (0, 0);
    }

    loop {
        let slice = slices[(rnd.rand32() as usize) % slices.len()];
        if slice.1 <= max_length {
            return slice;
        }
    }
}

fn splice(mut p1_pos: usize, mut p1_len: usize, mut p2_pos: usize, mut p2_len: usize, parent1: &GeneticProgram, parent2: &GeneticProgram) -> GeneticProgram {
    p1_pos *= 4;
    p1_len *= 4;
    p2_pos *= 4;
    p2_len *= 4;

    let p2_segment = parent2.code >> p2_pos & ((1u128 << p2_len) - 1);

    let p1_pre_segment = parent1.code & ((1u128 << p1_pos) - 1);
    let p1_post_segment = parent1.code >> (p1_pos + p1_len);
    let p1_without_segment = p1_pre_segment | p1_post_segment << (p1_pos + p2_len);

    // Find used slots
    let mut p1_used_slots = [false; 6];
    let mut code = p1_without_segment;
    while code != 0 {
        let opcode = (code & 0b1111) as u8;
        code >>= 4;
        if opcode < INSTR {
            // Data
            let slot = opcode;
            if slot >= 4 {
                p1_used_slots[slot as usize - 4] = true;
            }
        }
    }

    let mut child_data = parent1.data;
    let mut p2_used_slots = [false; 6];
    let mut p2_target_slot = [0; 6];

    // Find used slots for new code segment
    let mut code = p2_segment;
    let mut new_code = 0;
    let mut pos: usize = 0;
    while code != 0 {
        let opcode = (code & 0b1111) as u8;
        code >>= 4;
        if opcode >= 4 && opcode < LOAD_MAX_INDEX as u8 {
            // Data
            let slot = opcode;
            if p2_used_slots[slot as usize - 4] {
                new_code |= (p2_target_slot[slot as usize - 4] as u128) << pos;

            } else {
                let target_value = parent2.data[slot as usize];
                let target_slot = find_target_slot(p1_used_slots, child_data, target_value);

                child_data[target_slot] = target_value;
                p1_used_slots[target_slot - 4] = true;
                p2_used_slots[slot as usize - 4] = true;
                p2_target_slot[slot as usize - 4] = target_slot;

                new_code |= (target_slot as u128) << pos;
            }
        } else {
            new_code |= (opcode as u128) << pos;
        }

        pos += 4;
    }

    let child_code = p1_without_segment | (new_code << p1_pos);

    GeneticProgram::new(child_code, child_data, parent1.score_increment, parent1.score_raise)
}

fn find_target_slot(used_slots: [bool; 6], data: [u64; 10], value: u64) -> usize {
    // Check if value exists
    for i in 4..10 {
        if data[i] == value {
            return i;
        }
    }

    // Check for free slots
    for i in 4..10 {
        if !used_slots[i - 4] {
            return i;
        }
    }

    // Return last slot
    9
}

fn mutate(rnd: &mut Random, program: &GeneticProgram) -> GeneticProgram {
    let code;
    let mut data;

    if rnd.rand32() & 3 == 0 {

        let random_program = generate_program(rnd);

        let child = cross(rnd, &program, &random_program);
        code = child.code;
        data = child.data;

    } else {
        // Mutate constants
        code = program.code;
        data = program.data;

        let index = (4 + rnd.rand32() % 6) as usize;

        match rnd.rand32() & 7 {
            0 => data[index] ^= 1u64 << (rnd.rand32() & 63),
            1 => data[index] = data[index].saturating_sub(1),
            2 => data[index] += 1,
            3 => data[index] = data[index].saturating_sub(4),
            4 => data[index] += 4,
            5 => data[index] = data[index].saturating_sub(16),
            6 => data[index] += 16,
            _ => data[index] = !data[index],
        };
    }

    let mut score_increment = program.score_increment;
    match rnd.rand32() & 15 {
        0 => score_increment = -score_increment,
        1 => score_increment += 1,
        2 => score_increment -= 1,
        3 => score_increment += 8,
        4 => score_increment -= 8,
        _ => {}
    };
    score_increment = min(50, max(-50, score_increment));

    let mut score_raise = program.score_raise;
    match rnd.rand32() & 15 {
        1 => score_raise += 1,
        2 => score_raise -= 1,
        _ => {}
    };
    score_raise = min(10, max(0, score_raise));

    GeneticProgram::new(code, data, score_increment, score_raise)
}

pub fn generate_program(rnd: &mut Random) -> GeneticProgram {
    let min_size = (INSTR_LIMIT as u32 / 4) + (rnd.rand32() % (INSTR_LIMIT as u32 / 2));
    let mut index = 0;
    let mut sp : usize = 0;

    // let data = [0, 0, 0, 0, 1, 2, 5, 10, 100, 1000];
    let data = [0, 0, 0, 0, rnd.rand64() % 10, rnd.rand64() % 100, rnd.rand64() % 1000, rnd.rand64() % 10000, rnd.rand64() % 100000, rnd.rand64()];
    let mut code = 0;

    while index < 128 {
        let opcode;
        if sp == 0 || (sp == 1 && (rnd.rand32() & 15 > 0)) {
            // Load
            let slot = rnd.rand32() % (LOAD_MAX_INDEX as u32);
            opcode = slot as u8;
            sp += 1;
        } else if sp == 1 {
            // Single operand instruction
            opcode = Not as u8;
        } else {
            opcode = BINARY_OPS[rnd.rand32() as usize % BINARY_OPS.len()] as u8;
            sp -= 1;
        }

        code |= (opcode as u128) << index;
        index += 4;
        if index > min_size && sp == 1 {
            break;
        }
    }

    GeneticProgram::new(code, data, if rnd.rand32() & 1 == 0 { 2 } else { -2 }, 0)
}

pub fn sanitize(program: &GeneticProgram) -> GeneticProgram {
    GeneticProgram::new(sanitize_code(program.code), program.data, program.score_increment, program.score_raise)
}

pub fn sanitize_code(mut code: u128) -> u128 {
    let mut new_code;

    loop {
        new_code = 0;
        let mut sp = 0;
        let mut pos = 0;

        while code != 0 {
            let opcode = (code & 0b1111) as u8;
            code >>= 4;
            let mut skip = false;

            match unsafe { transmute(opcode) } {
                Not => {
                    if sp == 0 {
                        skip = true;
                    }
                },

                And | Or | Xor | Mul | Add => {
                    if sp >= 2 {
                        sp -= 1;
                    } else {
                        skip = true;
                    }
                },

                _ => {
                    // Load data
                    sp += 1;
                },
            }

            if !skip {
                new_code |= (opcode as u128) << pos;
                pos += 4;
            }
        }

        if sp <= 1 {
            break;
        }

        code = new_code >> 4;
    }

    new_code
}



#[cfg(test)]
mod tests {
    use crate::genetic_eval::{Assembler, GeneticProgram, splice, sanitize, compile, generate_program, random_slice, slice, optimize};
    use crate::random::Random;

    #[test]
    pub fn test_empty() {
        let code = 0;

        let program = GeneticProgram::new(code, [3, 4, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result,  0);
    }

    #[test]
    pub fn test_mul() {
        let code = Assembler::new()
            .load(0)
            .load(1)
            .mul()
            .build_code();

        let program = GeneticProgram::new(code, [3, 4, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result,  12);
    }

    #[test]
    pub fn test_or() {
        let code = Assembler::new()
            .load(0)
            .load(1)
            .or()
            .build_code();

        let program = GeneticProgram::new(code, [0b1000, 0b0010, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result, 0b1010);
    }

    #[test]
    pub fn test_and() {
        let code = Assembler::new()
            .load(0)
            .load(1)
            .and()
            .build_code();

        let program = GeneticProgram::new(code, [0b1010, 0b1101, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result, 0b1000);
    }

    #[test]
    pub fn test_not() {
        let code = Assembler::new()
            .load(0)
            .not()
            .build_code();

        let program = GeneticProgram::new(code, [0xF0F0F0F0F0F0F0F0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result, 0x0F0F0F0F0F0F0F0F);
    }

    #[test]
    pub fn test_xor() {
        let code = Assembler::new()
            .load(0)
            .load(1)
            .xor()
            .build_code();

        let program = GeneticProgram::new(code, [0b1110, 0b1010, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);

        let result = program.run();
        assert_eq!(result, 0b0100);
    }

    #[test]
    pub fn test_slice() {
        let program = GeneticProgram::new(12046499, [0, 0, 0, 0, 0, 1, 2, 3, 4, 5], 1, 0);
        let slices = slice(&program);

        assert_eq!(vec!((0, 1),
                        (0, 2),
                        (2, 1),
                        (0, 4),
                        (4, 1),
                        (0, 6)), slices);
    }

    #[test]
    pub fn test_splice() {
        let code1 = 260836001;
        let code2 = 3573469134174395599636;

        let program1 = GeneticProgram::new(code1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        let program2 = GeneticProgram::new(code2, [0, 0, 0, 0, 0, 0, 0, 9, 0, 0], 1, 0);

        let result = splice(0, 3, 14, 1, &program1, &program2);

        assert_eq!(result.code, Assembler::new().load(4).load(0).or().load(8).add().build_code());
        assert_eq!(result.data[4], 9);
    }

    #[test]
    pub fn test_sanitize() {
        let program = GeneticProgram::new(158237916293065189983371, [0, 0, 0, 0, 0, 13859172126465860881, 12387944325771311803, 16208407326110241713, 0, 0], 1, 0);
        let opt_program = sanitize(&program);

        opt_program.disassemble();
    }

    #[test]
    pub fn test_sanitize2() {
        let program = GeneticProgram::new(241184359275410648409880091269505, [0, 0, 0, 0, 35, 5, 95, 5, 1472, 8], 1, 0);
        let opt_program = sanitize(&program);

        opt_program.disassemble();
        println!("-------------------------");
        compile(&opt_program);
    }

    #[test]
    pub fn test_generate_program() {
        let mut rnd = Random::new();
        let program = generate_program(&mut rnd);

        let slice = random_slice(&mut rnd, &slice(&program), 7);
        println!("{:?}", slice);

        program.disassemble();
        compile(&program);
        println!("-------------------------");

        let without_slice = splice(slice.0, slice.1, 0, 1, &program, &program);

        without_slice.disassemble();
        compile(&without_slice);
    }

    #[test]
    pub fn test_compile() {
        let code = Assembler::new()
            .load(4)
            .load(5)
            .mul()
            .load(0)
            .mul()
            .build_code();

        let program = GeneticProgram::new(code, [0, 0, 0, 0, 7, 3, 0, 0, 0, 0], 1, 0);

        compile(&program);
    }

    #[test]
    pub fn test_optimize() {
        let code = Assembler::new()
            .load(0)
            .load(4)
            .add()
            .load(5)
            .xor()
            .load(6)
            .or()
            .load(7)
            .mul()
            .load(8)
            .load(9)
            .mul()
            .and()
            .build_code();

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 2, 3, 4, 5, 6], 1, 0);

        let mut opt = optimize(&program);
        assert!(opt.instr_count() < program.instr_count());

        for i in 0..256 {
            program.update(i, 0, 0, 0);
            opt.update(i, 0, 0, 0);

            assert_eq!(program.run(), opt.run());
        }
    }

    #[test]
    pub fn test_scores() {
        let score_increment = 1;
        let score_raise = 128;
        for count in 0..64 {
            let score = count * score_increment * (count + 1) / 2 * score_raise / 32;
            println!("{:2} - {:4}", count, score);
        }

    }

}
