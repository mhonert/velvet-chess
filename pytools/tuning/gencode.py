# Velvet Chess Engine
# Copyright (C) 2020 mhonert (https://github.com/mhonert)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Generates Rust source code from the tuning options configured in config.yml
# python gencode.py tuning
# - generates the options in tuning mode, i.e. all options can be set with 'setoption name ... value ...'
#
# python gencode.py prod
# - generates the options in prod mode, i.e. all options are immutable consts and cannot be changed

from dataclasses import dataclass
import logging as log
import sys
import re
from pathlib import Path
from typing import List, Optional
from zobrist import PIECE_RNG_NUMBERS, PLAYER_RNG_NUMBER, CASTLING_RNG_NUMBERS, EN_PASSANT_RNG_NUMBERS

from common import Config


def main(mode: str = "tuning"):
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = Config("config.yml", False)

    log.info("Generating code from config.yml for %s mode ...", mode)

    # Generate options.rs
    out = open("../../engine/src/options.rs", "w")

    out.write(Path("./codegen_snippets/header.rs").read_text())

    if mode == "tuning":
        gen_for_tuning_mode(config, out)
    else:
        gen_for_prod_mode(config, out)

    out.close()

    # Generate zobrist.rs
    out = open("../../engine/src/zobrist.rs", "w")

    out.write(Path("./codegen_snippets/header.rs").read_text())

    gen_zobrist_keys(out)


    out.close()


def to_snake_case(camel_case):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_case).lower()


@dataclass
class MultiOption:
    name: str
    size: int
    values: List[int]
    min_value: Optional[int]


def gen_for_prod_mode(config, out):
    out.write('''
use std::sync::mpsc::Sender;
use crate::engine::Message;
''')
    multi_options = {}
    single_options = []
    for option in config.tuning_options:
        if option.is_part:
            name = option.orig_name
            if name in multi_options:
                multi_options[name].size += 1
                multi_options[name].values.append(option.value)
                continue

            multi_options[option.orig_name] = MultiOption(name, 1, [option.value], option.minimum)

        else:
            single_options.append(option)
            out.write(f'''
const {to_snake_case(option.name).upper()}: i32 = {option.value};''')
    for option in multi_options.values():
        values = ", ".join([str(v) for v in option.values])
        out.write(f'''
const {to_snake_case(option.name).upper()}: [i32; {option.size}] = [{values}];''')

    out.write('''
    
pub struct Options {}
    
impl Options {
    pub fn new() -> Self {
        Options{}
    }
    ''')

    # Generic setter
    out.write('''
    pub fn set_option(&mut self, name: String, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    
    pub fn set_array_option(&mut self, name: String, _: usize, _: i32) {
        eprintln!("Unknown option {}", name);
    }
    ''')

    # Getter and setter
    for option in single_options:
        out.write(f'''

    #[inline]
    pub fn get_{to_snake_case(option.name)}(&self) -> i32 {{
        {to_snake_case(option.name).upper()}
    }}
''')
    for option in multi_options.values():
        if option.name.lower().endswith("pst"):
            # Special case: piece square table values

            out.write(f'''
    #[inline]
    pub const fn get_{to_snake_case(option.name)}(&self) -> [i32; {option.size}] {{
        {to_snake_case(option.name).upper()}
    }}
            ''')

        else:
            out.write(f'''
    #[inline]
    pub fn get_{to_snake_case(option.name)}(&self, index: usize) -> i32 {{
        unsafe {{ *{to_snake_case(option.name).upper()}.get_unchecked(index) }}
    }}
    ''')
    out.write(f'''
}}
''')

    # UCI option parsing (noop in prod mode)
    out.write('''
pub fn parse_set_option(_: &Sender<Message>, _: &str, _: &str) {}
    ''')


def gen_for_tuning_mode(config, out):
    out.write('''
use std::cmp::max;
use std::sync::mpsc::Sender;
use crate::engine::Message;
use std::str::FromStr;

pub struct Options {''')
    multi_options = {}
    single_options = []
    for option in config.tuning_options:
        if option.is_part:
            name = option.orig_name
            if name in multi_options:
                multi_options[name].size += 1
                multi_options[name].values.append(option.value)
                continue

            multi_options[option.orig_name] = MultiOption(name, 1, [option.value], option.minimum)

        else:
            single_options.append(option)
            out.write(f'''
    {to_snake_case(option.name)}: i32,''')
    for option in multi_options.values():
        out.write(f'''
    {to_snake_case(option.name)}: [i32; {option.size}],''')
    out.write('''
}

impl Options {
    pub fn new() -> Self {
        Options{''')
    for option in single_options:
        out.write(f'''
            {to_snake_case(option.name)}: {option.value},''')
    for option in multi_options.values():
        values = ", ".join([str(v) for v in option.values])
        out.write(f'''
            {to_snake_case(option.name)}: [{values}],''')
    out.write('''
        }
    }
    ''')
    # Generic single setter
    out.write('''
    pub fn set_option(&mut self, name: String, value: i32) {
        match name.as_str() {''')
    for option in single_options:
        out.write(f'''
            "{option.name.lower()}" => self.set_{to_snake_case(option.name)}(value),''')
    out.write('''
            _ => println!("Unknown option {}", name)
        }
    }
    ''')
    # Generic array setter
    out.write('''
    pub fn set_array_option(&mut self, name: String, index: usize, value: i32) {
        match name.as_str() {''')
    for option in multi_options.values():
        out.write(f'''
            "{option.name.lower()}" => self.set_{to_snake_case(option.name)}(index, value),''')
    out.write('''
            _ => println!("Unknown option {}", name)
        }
    }
    ''')
    # Getter and setter
    for option in single_options:
        out.write(f'''
    fn set_{to_snake_case(option.name)}(&mut self, value: i32) {{
        self.{to_snake_case(option.name)} = value;
    }}

    #[inline]
    pub fn get_{to_snake_case(option.name)}(&self) -> i32 {{
        self.{to_snake_case(option.name)}
    }}
''')
    for option in multi_options.values():
        if option.min_value is not None:
            out.write(f'''
    fn set_{to_snake_case(option.name)}(&mut self, index: usize, value: i32) {{
        self.{to_snake_case(option.name)}[index] = max({option.min_value}, value);
    }}
                ''')

        else:
            out.write(f'''
    fn set_{to_snake_case(option.name)}(&mut self, index: usize, value: i32) {{
        self.{to_snake_case(option.name)}[index] = value;
    }}
        ''')

        if option.name.lower().endswith("pst"):
            # Special case: piece square table values

            out.write(f'''
    #[inline]
    pub fn get_{to_snake_case(option.name)}(&self) -> [i32; {option.size}] {{
        self.{to_snake_case(option.name)}
    }}
            ''')

        else:
            out.write(f'''
    #[inline]
    pub fn get_{to_snake_case(option.name)}(&self, index: usize) -> i32 {{
        self.{to_snake_case(option.name)}[index]
    }}
    ''')
    out.write(f'''
}}
''')
    # UCI option parsing
    single_option_names = ['"' + o.name.lower() + '"' for o in single_options]
    joined_names = ", ".join(single_option_names)
    out.write(f'''
const SINGLE_VALUE_OPTION_NAMES: [&'static str; {len(single_option_names)}] = [{joined_names}];''')
    multi_option_names = ['"' + o.name.lower() + '"' for o in multi_options.values()]
    joined_names = ", ".join(multi_option_names)
    out.write(f'''
const MULTI_VALUE_OPTION_NAMES: [&'static str; {len(multi_option_names)}] = [{joined_names}];
''')
    out.write(Path("./codegen_snippets/options_parse.rs").read_text())


def gen_zobrist_keys(out):
    out.write(f'''const PLAYER_ZOBRIST_KEY: u64 = {PLAYER_RNG_NUMBER};\n''')
    out.write(f'''const EN_PASSANT_ZOBRIST_KEYS: [u64; 16] = {EN_PASSANT_RNG_NUMBERS};\n''')
    out.write(f'''const CASTLING_ZOBRIST_KEYS: [u64; 16] = {CASTLING_RNG_NUMBERS};\n''')
    out.write(f'''const PIECE_ZOBRIST_KEYS: [u64; 13 * 64] = {PIECE_RNG_NUMBERS};\n''')

    out.write('''
    
#[inline]
pub fn player_zobrist_key() -> u64 {
    PLAYER_ZOBRIST_KEY
}

#[inline]
pub fn enpassant_zobrist_key(en_passant_state: u16) -> u64 {
    unsafe { *EN_PASSANT_ZOBRIST_KEYS.get_unchecked(en_passant_state.trailing_zeros() as usize) }
}

#[inline]
pub fn castling_zobrist_key(castling_state: u8) -> u64 {
    unsafe { *CASTLING_ZOBRIST_KEYS.get_unchecked(castling_state as usize & 0xf) }
}

#[inline]
pub fn piece_zobrist_key(piece: i8, pos: usize) -> u64 {
    unsafe { *PIECE_ZOBRIST_KEYS.get_unchecked(((piece + 6) as usize) * 64 + pos) }
}
''')


# Main
if __name__ == "__main__":
    main(*(sys.argv[1:]))
