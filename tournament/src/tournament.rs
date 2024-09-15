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
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::{Arc, RwLock};
use std::sync::atomic::AtomicBool;
use velvet::nn::io::FastHasher;
use crate::config::{EngineConfig, EngineConfigs, TournamentConfig};

pub struct TournamentState {
    stopped: AtomicBool,
    shared: RwLock<SharedState>,
}

impl TournamentState {
    pub fn new(tournament_config: TournamentConfig, engine_configs: EngineConfigs) -> anyhow::Result<Arc<TournamentState>> {
        
        TournamentState {
            stopped: AtomicBool::new(false),
            shared: RwLock::new(SharedState::default()),
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn next_opponent(&self) -> Option<EngineConfig> {
        self.shared.write().expect("Could not acquire write lock on shared state").next_opponent()
    }
    
    pub fn add_opponent(&self, config: EngineConfig) {
        self.shared.write().expect("Could not acquire write lock on shared state").opponents.insert(config.id, Opponent {
            config,
            wins: 0,
            draws: 0,
            losses: 0,
            matches: 0,
        });
    }
}

type OpponentMap = HashMap<usize, Opponent, BuildHasherDefault<FastHasher>>;

struct SharedState {
    opponents: OpponentMap,
}

impl Default for SharedState {
    fn default() -> SharedState {
        SharedState {
            opponents: OpponentMap::default(),
        }
    }
}

impl SharedState {
    pub fn next_opponent(&mut self) -> Option<EngineConfig> {
        if let Some(opponent) = self.opponents.values_mut().min_by_key(|opponent| opponent.matches()) {
            opponent.matches += 1;
            return Some(opponent.config.clone());
        }
        
        None
    }
}

#[derive(Clone)]
struct Opponent {
    config: EngineConfig,
    wins: usize,
    draws: usize,
    losses: usize,
    matches: usize,
}

impl Opponent {
    // matches returns the number of finished and ongoing matches
    pub fn matches(&self) -> usize {
        self.matches
    }
}