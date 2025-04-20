/*
 * Velvet Chess Engine
 * Copyright (C) 2025 mhonert (https://github.com/mhonert)
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

use velvet::engine;
use velvet::engine::Message;
use velvet::init::init;
use velvet::uci;

fn main() {
    init();
    let tx = engine::spawn_engine_thread();

    if let Some(arg) = std::env::args().nth(1) {
        match arg.as_str() {
            "bench" | "profile" => {
                tx.send(Message::Profile).expect("Failed to send message");
            }
            "multibench" => {
                tx.send(Message::SetThreadCount(2)).expect("Failed to send message");
                tx.send(Message::SetTranspositionTableSize(64)).expect("Failed to send message");
                tx.send(Message::IsReady).expect("Failed to send message");
                tx.send(Message::Profile).expect("Failed to send message");
            }
            _ => {}
        }
    }

    uci::start_uci_loop(&tx);
}
