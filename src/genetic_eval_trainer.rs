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
use crate::random::Random;
use std::cmp::{min, max};
use crate::genetic_eval_trainer::Term::{Constant, Variable, UnaryOp, BinaryOp};
use std::collections::HashMap;
use wasmer::{Store, Module, Instance, Value, imports, Cranelift};
use crate::genetic_eval_trainer::Instruction::{Xor, And, Not, Or, Mul, Add};
use uint::construct_uint;

const CROSS_OVER_RATE: u32 = 90;
const MUTATION_RATE: u32 = 10;
const ELITE_COUNT: u32 = 1;

// const BINARY_OPS: [Instruction; 2] = [Mul, Add];
const BINARY_OPS: [Instruction; 4] = [Mul, Or, Xor, And];

pub const INSTR_SIZE: u64 = 6;
pub const INSTR_BITMASK: u64 = 0b111111;
pub const INSTR_LIMIT: usize = 512 / (INSTR_SIZE as usize);

construct_uint! {
	pub struct U512(8);
}

#[derive(Copy, Clone)]
pub struct GeneticProgram {
    pub code: U512,
    pub constants: [u64; 8],
    pub data: [u64; 24],
    pub score_increment: i32,
    pub score_raise: i32,
}

impl GeneticProgram {

    pub fn new(code: U512, constants: [u64; 8], score_increment: i32, score_raise: i32) -> GeneticProgram {
        GeneticProgram {
            code,
            data: [0; 24],
            constants,
            score_increment,
            score_raise
        }
    }

    pub fn new_from_str(code: &str, constants: [u64; 8], score_increment: i32, score_raise: i32) -> GeneticProgram {
        let code = U512::from_str_radix(code, 10).expect("code must be a 512-Bit unsigned integer");
        GeneticProgram::new(code, constants, score_increment, score_raise)
    }

    pub fn update(&mut self, v: &[u64; 24]) {
        self.data = *v
    }

    pub fn instr_count(&self) -> usize {
        ((512 - self.code.leading_zeros()) as usize) / (INSTR_SIZE as usize)
    }

    pub fn run(&self) -> u64 {
        let mut data = [0; 32];
        for i in 0..24 {
            data[i] = self.data[i]
        }

        for i in 0..8 {
            data[i + 24] = self.constants[i]
        }

        let mut stack: [u64; 64] = [0; 64];
        let mut sp : usize = 0;

        let mut code = self.code;

        while code.low_u64() != 0 {
            let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
            code >>= INSTR_SIZE;
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
                        let v2 = *stack.get_unchecked(sp - 1);
                        let v = stack.get_unchecked_mut(sp - 2);
                        *v = v.wrapping_add(v2);
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
                        let v2 = *stack.get_unchecked(sp - 1);
                        let v = stack.get_unchecked_mut(sp - 2);
                        *v = v.wrapping_mul(v2);
                    }
                    sp -= 1;
                },

                _ => {
                    // Load data
                    unsafe {
                        let v = *data.get_unchecked(opcode as usize - 1);
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
        println!("debug GenEval Constants: [{}]", self.constants.iter()
            .map(|&n| format!("{:016x}", n))
            .collect::<Vec<String>>()
            .join(", "));

        println!("debug GenEval Disassembled Code:");

        let mut code = self.code;

        let mut num = 0;
        while code.low_u64() != 0 {
            let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
            code >>= INSTR_SIZE;
            if opcode >= INSTR {
                let instruction: Instruction = unsafe { transmute(opcode) };
                println!("{:03}: {:?}", num, instruction);
            } else {
                // Load data
                let slot = opcode - 1;
                println!("{:03}: Load{}", num, slot);
            }
            num += 1;
        }
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub enum Instruction {
    Load0 = 1,
    Load1,
    Load2,
    Load3,
    Load4,
    Load5,
    Load6,
    Load7,
    Load8,
    Load9,
    Load10,
    Load11,
    Load12,
    Load13,
    Load14,
    Load15,
    Load16,
    Load17,
    Load18,
    Load19,
    Load20,
    Load21,
    Load22,
    Load23,
    Load24,
    Load25,
    Load26,
    Load27,
    Load28,
    Load29,
    Load30,
    Load31,
    Not,
    And,
    Or,
    Xor,
    Mul,
    Add,
}

// First instruction opcode
pub const INSTR: u8 = Not as u8;

impl Instruction {
    pub fn run_two_op(&self, op1: u64, op2: u64) -> u64 {
        match self {
            And => op1 & op2,
            Add => op1 + op2,
            Or => op1 | op2,
            Xor => op1 ^ op2,
            Mul => op1.wrapping_mul(op2),
            _ => panic!("{:?} is not a 2-operand instruction", self)
        }
    }
}


pub struct CodeGen {
    instance: Instance,
}

impl CodeGen {
    pub fn new(programs: &[GeneticProgram]) -> Self {
        let store = Store::default();

        let wat = compile_to_wasm(programs);
        let module = Module::new(&store, &wat).unwrap();

        let import_object = imports! {};
        let instance = Instance::new(&module, &import_object).unwrap();

        CodeGen{
            instance,
        }
    }
}

fn compile_to_wasm(programs: &[GeneticProgram]) -> String {
    let header =
        r#"
(module
    (func $eval (param i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64 i64) (result i64) (local $tmp i64)
"#;

    let footer = r#")
    (export "eval" (func $eval)))"#;


    let mut body = String::new();
    body += header;
    let mut count = 0;
    for program in programs.iter() {
        let term = parse(program).optimized();
        body += compile_term(&term).as_str();

        body += "        i64.popcnt\n";

        if program.score_raise == 0 {
            body += mul(program.score_increment).as_str();

        }  else {
            body += "        local.set $tmp\n";
            body += "        local.get $tmp\n";
            body += mul(program.score_increment).as_str();
            body += "        local.get $tmp\n";
            body += format!("        i64.const {}\n", 1).as_str();
            body += "        i64.add\n";
            body += "        i64.mul\n";
            body += format!("        i64.const {}\n", 2).as_str();
            body += "        i64.div_s\n";
            body += mul(program.score_raise).as_str();
        };

        count += 1;
        if count > 1 {
            body += "        i64.add\n";
        }
    }

    body += footer;
    body
}

fn mul(value: i32) -> String {
    if value == 1 {
        return String::from("");
    }

    return format!("        i64.const {}\n        i64.mul\n", value);
}

fn compile_term(term: &Term) -> String {

    match term {
        Constant(value) => {
            format!("        i64.const 0x{:x}\n", *value)
        },

        Variable(slot) => {
            format!("        local.get {}\n", *slot)
        },

        UnaryOp(instr, op) => {
            let p = compile_term(op);

            match *instr {
                Not => format!("{}        i64.const 0x{:x}\n        i64.xor\n", p, u64::max_value()),
                _ => panic!("Unexpected unary operator: {:?}", *instr)
            }
        },

        BinaryOp(instr, op1, op2) => {
            let p1 = compile_term(op1);
            let p2 = compile_term(op2);

            let op = match *instr {
                And => "i64.and",
                Xor => "i64.xor",
                Or => "i64.or",
                Add => "i64.add",
                Mul => "i64.mul",
                _ => panic!("Unexpected binary operator: {:?}", *instr)
            };

            format!("{}{}        {}\n", p1, p2, op)
        }
    }
}

pub struct GeneticEvaluator {
    programs: Vec<GeneticProgram>,
    opt_programs: Vec<GeneticProgram>,
    code_gen: Option<CodeGen>,
}

impl GeneticEvaluator {

    pub fn new() -> Self {
        GeneticEvaluator{
            programs: Vec::new(),
            opt_programs: Vec::new(),
            code_gen: None,
        }
    }

    pub fn compile(&mut self) {
        if self.programs.len() > 0 {
            self.code_gen = Some(CodeGen::new(&self.opt_programs));
        } else {
            self.code_gen = None;
        }
    }

    pub fn eval(&mut self, own_pawns: u64, opp_pawns: u64, own_knights: u64, opp_knights: u64, own_bishops: u64, opp_bishops: u64,
                own_rooks: u64, opp_rooks: u64, own_queens: u64, opp_queens: u64, own_king_bb: u64, opp_king_bb: u64,
                own_pawn_attacks: u64, opp_pawn_attacks: u64, own_knight_attacks: u64, opp_knight_attacks: u64, own_bishop_attacks: u64, opp_bishop_attacks: u64,
                own_rook_attacks: u64, opp_rook_attacks: u64, own_queen_attacks: u64, opp_queen_attacks: u64,
                opp_king_half: u64, own_king_half: u64) -> i64 {

        if self.code_gen.is_none() {
            return 0;
        }

        let func = self.code_gen.as_ref().map(|c| c.instance.exports.get_function("eval")).unwrap().unwrap();

        let result = func.call(&[Value::I64(own_pawns as i64), Value::I64(opp_pawns as i64), Value::I64(own_knights as i64),
            Value::I64(opp_knights as i64), Value::I64(own_bishops as i64),  Value::I64(opp_bishops as i64),
            Value::I64(own_rooks as i64), Value::I64(opp_rooks as i64),  Value::I64(own_queens as i64),  Value::I64(opp_queens as i64), Value::I64(own_king_bb as i64), Value::I64(opp_king_bb as i64),
            Value::I64(own_pawn_attacks as i64), Value::I64(opp_pawn_attacks as i64), Value::I64(own_knight_attacks as i64), Value::I64(opp_knight_attacks as i64), Value::I64(own_bishop_attacks as i64),
            Value::I64(opp_bishop_attacks as i64), Value::I64(own_rook_attacks as i64), Value::I64(opp_rook_attacks as i64), Value::I64(own_queen_attacks as i64), Value::I64(opp_queen_attacks as i64),
            Value::I64(opp_king_half as i64), Value::I64(own_king_half as i64)]).unwrap();

        result[0].unwrap_i64()
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

    pub fn print_rust_code(&self) {
        println!("pub fn genetic_eval({}) -> i32 {{", VAR_NAMES.iter().map(|&var| format!("{}: u64", var)).collect::<Vec<String>>().join(", "));
        println!("    let mut score: i32 = 0;");

        for program in self.programs.iter() {
            print_rust_code(&program);
        }

        println!("    score");
        println!("}}")
    }
}

fn print_generation(programs: &[GeneticProgram]) {
    print!("result ");
    for program in programs.iter() {
        let solution_size = program.instr_count();
        print!("{},{},{},{},{};",
               program.code,
               program.constants.iter()
                   .map(|&n| format!("{}", n))
                   .collect::<Vec<String>>()
                   .join(","),
               program.score_increment,
               program.score_raise,
               solution_size);
    }
    println!();
}


#[derive(Clone)]
pub enum Term {
    Constant(u64),
    Variable(usize),
    UnaryOp(Instruction, Box<Term>),
    BinaryOp(Instruction, Box<Term>, Box<Term>),
}

const VAR_NAMES: [&str; 24] = ["own_pawns", "opp_pawns", "own_knights", "opp_knights", "own_bishops", "opp_bishops",
    "own_rooks", "opp_rooks", "own_queens", "opp_queens", "own_king_bb", "opp_king_bb",
    "own_pawn_attacks", "opp_pawn_attacks", "own_knight_attacks", "opp_knight_attacks", "own_bishop_attacks", "opp_bishop_attacks",
    "own_rook_attacks", "opp_rook_attacks", "own_queen_attacks", "opp_queen_attacks",
    "opp_king_half", "own_king_half"];

impl Term {
    pub fn to_expr_str(&self) -> String {
        match self {
            Variable(slot) => String::from(VAR_NAMES[*slot]),
            // Constant(value) => format!("{:016x}u64", &value),
            Constant(value) => format!("{}u64", &value),
            UnaryOp(instr, op) => {
                match *instr {
                    Not => format!("!{}", op.to_expr_str()),
                    _ => panic!("{:?} is not a one operand instruction", *instr),
                }
            }
            BinaryOp(instr, op1, op2) => {
                match *instr {
                    And => format!("({} & {})", op1.to_expr_str(), op2.to_expr_str()),
                    Add => format!("{}.wrapping_add({})", op1.to_expr_str(), op2.to_expr_str()),
                    Or => format!("({} | {})", op1.to_expr_str(), op2.to_expr_str()),
                    Xor => format!("({} ^ {})", op1.to_expr_str(), op2.to_expr_str()),
                    Mul => format!("{}.wrapping_mul({})", op1.to_expr_str(), op2.to_expr_str()),
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

pub fn print_rust_code(program: &GeneticProgram) {
    let term = parse(&program).optimized();

    println!("    let tmp_score = {}.count_ones() as i32;", term.to_expr_str());
    if program.score_raise == 0 {
        println!("    score += tmp_score * {};", program.score_increment);
    } else {
        println!("    score += tmp_score * {} * (tmp_score + 1) / 2 * {};", program.score_increment, program.score_raise);
    }
}

fn parse(program: &GeneticProgram) -> Term {
    let mut stack: Vec<Term> = Vec::new();
    for _ in 0..64 {
        stack.push(Constant(0));
    }

    let mut sp : usize = 0;

    let mut code = sanitize_code(program.code);

    while code.low_u64() != 0 {
        let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
        code >>= INSTR_SIZE;
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
                let slot = opcode as usize - 1;
                if slot < 24 {
                    stack[sp] = Variable(slot);
                } else {
                    stack[sp] = Constant(program.constants[slot - 24]);
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
    code: U512,
    index: usize,
    valid: bool,
}

impl OpcodeGenerator {
    pub fn new() -> Self {
        OpcodeGenerator{
            slot_by_constant: HashMap::new(),
            code: U512::zero(),
            index: 0,
            valid: true
        }
    }

    pub fn generate(&mut self, term: &Term, score_increment: i32, score_raise: i32) -> Option<GeneticProgram> {
        self.slot_by_constant.clear();
        self.index = 0;
        self.valid = true;

        self.process(term);

        let mut constants = [0; 8];
        self.slot_by_constant.iter().for_each(|(&value, &slot)| constants[slot] = value);

        if self.valid {
            Some(GeneticProgram::new(self.code, constants, score_increment, score_raise))
        } else {
            None
        }
    }

    fn process(&mut self, term: &Term) {
        match term {
            Constant(value) => {
                let slot = self.register_constant(*value);

                self.emit_opcode(slot as u8 + 1 + 24);
            },

            Variable(slot) => {
                self.emit_opcode(*slot as u8 + 1);
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
        self.code = self.code | (U512::from(opcode) << U512::from(self.index));
        self.index += INSTR_SIZE as usize;
    }

    fn register_constant(&mut self, value: u64) -> usize {
        match self.slot_by_constant.get(&value) {
            Some(&slot) => slot,
            None => {
                let slot = self.slot_by_constant.len();
                if slot >= 8 as usize {
                    self.valid = false;
                    return slot - 1;
                }
                self.slot_by_constant.insert(value, slot);
                slot
            }
        }
    }
}

pub struct Assembler(U512, usize);

impl Assembler {
    pub fn new() -> Self {
        Assembler(U512::zero(), 0)
    }

    pub fn load(&mut self, slot: u8) -> &mut Self {
        if slot >= 32 as u8 {
            panic!("Slot must be in the range 0-31")
        }

        self.add_opcode(slot + 1)
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
        self.0 = self.0 | (U512::from(opcode) << bitpos);
        self.1 += INSTR_SIZE as usize;
        self
    }

    #[allow(dead_code)]
    fn build_code(&self) -> U512 {
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

fn cross(rnd: &mut Random, parent1: &GeneticProgram, parent2: &GeneticProgram) -> GeneticProgram {
    let p1_slices = slice(&parent1);
    let p2_slices = slice(&parent2);

    let (p1_start, p1_len) = random_slice(rnd, &p1_slices, INSTR_LIMIT / 4);

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

    while code.low_u64() != 0 {
        let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
        code >>= INSTR_SIZE;
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
    p1_pos *= INSTR_SIZE as usize;
    p1_len *= INSTR_SIZE as usize;
    p2_pos *= INSTR_SIZE as usize;
    p2_len *= INSTR_SIZE as usize;

    let p2_segment = parent2.code >> p2_pos & ((U512::one() << p2_len) - 1);

    let p1_pre_segment = parent1.code & ((U512::one() << p1_pos) - 1);
    let p1_post_segment = parent1.code >> (p1_pos + p1_len);
    let p1_without_segment = p1_pre_segment | p1_post_segment << (p1_pos + p2_len);

    // Find used slots
    let mut p1_used_slots = [false; 8];
    let mut code = p1_without_segment;
    while code.low_u64() != 0 {
        let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
        code >>= INSTR_SIZE;
        if opcode == 0 { // skip parts without opcodes within p1_without_segment
            continue;
        }
        if opcode < INSTR {
            // Data
            let slot = opcode - 1;
            if slot >= 24 {
                p1_used_slots[slot as usize - 24] = true;
            }
        }
    }

    let mut child_data = parent1.constants;
    let mut p2_used_slots = [false; 8];
    let mut p2_target_slot = [0; 8];

    // Find used slots for new code segment
    let mut code = p2_segment;
    let mut new_code = U512::zero();
    let mut pos: usize = 0;
    while code.low_u64() != 0 {
        let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
        code >>= INSTR_SIZE;
        if (25..33).contains(&opcode) {
            // Constant
            let slot = opcode - 1;
            if p2_used_slots[slot as usize - 24] {
                new_code = new_code | (U512::from(p2_target_slot[slot as usize - 24] + 1) << pos);

            } else {
                let target_value = parent2.constants[slot as usize - 24];
                let target_slot = find_target_slot(p1_used_slots, child_data, target_value);

                child_data[target_slot - 24] = target_value;
                p1_used_slots[target_slot - 24] = true;
                p2_used_slots[slot as usize - 24] = true;
                p2_target_slot[slot as usize - 24] = target_slot;

                new_code = new_code | (U512::from(target_slot + 1) << pos);
            }
        } else {
            new_code = new_code | (U512::from(opcode) << pos);
        }

        pos += INSTR_SIZE as usize;
    }

    let child_code = p1_without_segment | (new_code << p1_pos);

    GeneticProgram::new(child_code, child_data, parent1.score_increment, parent1.score_raise)
}

fn find_target_slot(used_slots: [bool; 8], data: [u64; 8], value: u64) -> usize {
    // Check if value exists
    for i in 0..8 {
        if data[i] == value {
            return i + 24;
        }
    }

    // Check for free slots
    for i in 0..8 {
        if !used_slots[i] {
            return i + 24;
        }
    }

    // Return last slot
    7 + 24
}

fn mutate(rnd: &mut Random, program: &GeneticProgram) -> GeneticProgram {
    let code;
    let mut data;

    if rnd.rand32() & 3 == 0 {

        let random_program = generate_program(rnd);

        let child = cross(rnd, &program, &random_program);
        code = child.code;
        data = child.constants;

    } else {
        // Mutate constants
        code = program.code;
        data = program.constants;

        let index = (rnd.rand32() % 8) as usize;

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
    score_increment = min(1500, max(-1500, score_increment));

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
    let min_size = ((INSTR_LIMIT as u32 / 8) + (rnd.rand32() % (INSTR_LIMIT as u32 / 4))) as usize;
    let mut index = 0;
    let mut sp : usize = 0;

    let mut constants = [0; 8];
    let mut range = 10;
    for i in 0..7 {
        constants[i] = rnd.rand64() % range;
        range *= 10;
    }
    constants[7] = rnd.rand64();

    let mut code = U512::zero();

    while index < INSTR_LIMIT {
        let opcode;
        if sp == 0 || (sp == 1 && (rnd.rand32() & 15 > 0)) {
            // Load
            let slot = rnd.rand32() % 32;
            opcode = slot as u8 + 1;
            sp += 1;
        } else if sp == 1 {
            // Single operand instruction
            opcode = Not as u8;
        } else {
            opcode = BINARY_OPS[rnd.rand32() as usize % BINARY_OPS.len()] as u8;
            sp -= 1;
        }

        code = code | (U512::from(opcode) << index);
        index += INSTR_SIZE as usize;
        if index > min_size && sp == 1 {
            break;
        }
    }

    GeneticProgram::new(code, constants, if rnd.rand32() & 1 == 0 { 2 } else { -2 }, 0)
}

pub fn sanitize(program: &GeneticProgram) -> GeneticProgram {
    GeneticProgram::new(sanitize_code(program.code), program.constants, program.score_increment, program.score_raise)
}

pub fn sanitize_code(mut code: U512) -> U512 {
    let mut new_code;

    loop {
        new_code = U512::zero();
        let mut sp = 0;
        let mut pos = 0;

        while code.low_u64() != 0 {
            let opcode = (code.low_u64() & INSTR_BITMASK) as u8;
            code >>= INSTR_SIZE;
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
                new_code = new_code | (U512::from(opcode) << pos);
                pos += INSTR_SIZE;
            }
        }

        if sp <= 1 {
            break;
        }

        code = new_code >> INSTR_SIZE;
    }

    new_code
}


#[cfg(test)]
mod tests {
    use crate::genetic_eval_trainer::{U512, Assembler, GeneticProgram, splice, print_rust_code, generate_program, random_slice, slice, optimize, GeneticEvaluator, parse, compile_to_wasm};
    use crate::random::Random;
    use std::time::Instant;
    use wasmer::{Store, Module, Instance, Value, imports, JITEngine, JIT};

    #[test]
    pub fn test_empty() {
        let code = U512::zero();

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

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

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

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

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[0b1000, 0b0010, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

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

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[0b1010, 0b1101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let result = program.run();
        assert_eq!(result, 0b1000);
    }

    #[test]
    pub fn test_not() {
        let code = Assembler::new()
            .load(0)
            .not()
            .build_code();

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[0xF0F0F0F0F0F0F0F0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

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

        let mut program = GeneticProgram::new(code, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        program.update(&[0b1110, 0b1010, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);

        let result = program.run();
        assert_eq!(result, 0b0100);
    }

    #[test]
    pub fn test_splice() {
        let code1 = U512::from(260836001);
        let code2 = U512::from(3573469134174395599636u128);

        let program1 = GeneticProgram::new(code1, [0, 0, 0, 0, 0, 0, 0, 0], 1, 0);
        let program2 = GeneticProgram::new(code2, [0, 0, 0, 0, 0, 9, 0, 0], 1, 0);

        let result = splice(0, 3, 14, 1, &program1, &program2);

        assert_eq!(result.code, Assembler::new().load(4).load(0).or().load(8).add().build_code());
        assert_eq!(result.data[4], 9);
    }

    #[test]
    pub fn test_generate_program() {
        let mut rnd = Random::new();
        let program = generate_program(&mut rnd);
        program.disassemble();

        let slice = random_slice(&mut rnd, &slice(&program), 7);

        print_rust_code(&program);

        let without_slice = splice(slice.0, slice.1, 0, 1, &program, &program);

        without_slice.disassemble();
        print_rust_code(&without_slice);
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

        let program = GeneticProgram::new(code, [0, 0, 7, 3, 0, 0, 0, 0], 1, 0);

        print_rust_code(&program);
    }

    #[test]
    pub fn test_optimize() {
        let code = Assembler::new()
            .load(0)
            .load(24)
            .add()
            .load(25)
            .xor()
            .load(26)
            .or()
            .load(27)
            .mul()
            .load(28)
            .load(29)
            .mul()
            .and()
            .build_code();

        let mut program = GeneticProgram::new(code, [0, 0, 0, 2, 3, 4, 5, 6], 1, 0);

        let mut opt = optimize(&program);
        opt.disassemble();
        assert!(opt.instr_count() < program.instr_count());

        for i in 0..256 {
            let mut data = [0; 24];
            data[0] = i;
            program.update(&data);
            opt.update(&data);

            assert_eq!(program.run(), opt.run());
        }
    }

    #[test]
    pub fn test_print_rust() {
        let code1 = Assembler::new()
            .load(0)
            .load(5)
            .add()
            .build_code();
        let program1 = GeneticProgram::new(code1, [1, 2, 3, 4, 0, 0, 0, 0], 1, 0);

        let code2 = Assembler::new()
            .load(1)
            .load(4)
            .add()
            .build_code();
        let program2 = GeneticProgram::new(code2, [1, 2, 3, 4, 0, 0, 0, 0], 1, 0);

        println!("Compiling ...");

        let mut eval = GeneticEvaluator::new();
        eval.add_program(program1);
        eval.add_program(program2);
        eval.compile();


        eval.print_rust_code();
    }

}