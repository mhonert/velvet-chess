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

extern crate bindgen;
extern crate cc;

use std::env;
use std::path::PathBuf;

fn main() {
    let mut build = cc::Build::default();

    build.include("fathom/src")
        .file("fathom/src/tbprobe.c")
        .static_flag(true)
        .static_crt(true);

    if env::consts::OS == "windows" {
        // Workaround for error: <stdatomic.h> is not yet supported when compiling as C
        build.compiler("clang");
        build.define("_CRT_SECURE_NO_WARNINGS", None);
        build.flag(env::var("CC_MARCH").unwrap().as_str());
    }

    build.compile("fathom");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=fathom/src/tbprobe.h");

    let bindings = bindgen::Builder::default()
        .header("fathom/src/tbprobe.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");
    //
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
