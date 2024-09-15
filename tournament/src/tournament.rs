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
use selfplay::pentanomial::{PentanomialCount, PentanomialModel};
use velvet::nn::io::FastHasher;
use crate::config::{EngineConfig, EngineConfigs, TournamentConfig};

pub struct TournamentState {
    stopped: AtomicBool,
    shared: RwLock<SharedState>,
}

impl TournamentState {
    pub fn new(tournament_config: &TournamentConfig, engine_configs: &EngineConfigs) -> anyhow::Result<Arc<TournamentState>> {
        let mut shared_state = SharedState::default();
        for opponent in tournament_config.opponents.iter() {
            if let Some(config) = engine_configs.0.get(opponent) {
                shared_state.add_opponent(config.clone())?;
            } else {
                anyhow::bail!("Could not find engine config for opponent '{}'", opponent);
            }
        }

        anyhow::Ok(Arc::new(TournamentState {
            stopped: AtomicBool::new(false),
            shared: RwLock::new(shared_state),
        }))
    }

    pub fn stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::Relaxed)
    }
    
    pub fn next_opponent(&self) -> Option<(EngineConfig, usize)> {
        self.shared.write().expect("Could not acquire write lock on shared state").next_opponent()
    }

    pub fn update(&self, opponent_id: u32, result: PentanomialCount) {
        self.shared.write().expect("Could not acquire write lock on shared state").update(opponent_id, result);
    }
}

type OpponentMap = HashMap<u32, Opponent, BuildHasherDefault<FastHasher>>;

#[derive(Default)]
struct SharedState {
    finished_games: usize,
    started_pairs: usize,
    opponents: OpponentMap,
}

const STD_NORM_DEV_95: f64 = 1.959963984540054;

impl SharedState {
    pub fn add_opponent(&mut self, config: EngineConfig) -> anyhow::Result<()> {
        let name = config.name.clone();
        if self.opponents.insert(config.id, Opponent::new(config)).is_some() {
            anyhow::bail!("Opponent {} already exists", name);
        }

        Ok(())
    }

    pub fn next_opponent(&mut self) -> Option<(EngineConfig, usize)> {
        if let Some(opponent) = self.opponents.values_mut().min_by_key(|opponent| opponent.matches()) {
            opponent.matches += 1;
            self.started_pairs += 1;
            return Some((opponent.config.clone(), self.started_pairs));
        }
        
        None
    }

    pub fn update(&mut self, opponent_id: u32, result: PentanomialCount) {
        let opponent = self.opponents.get_mut(&opponent_id).expect("Could not find opponent");
        opponent.results.add_all(result);
        self.finished_games += 2;
        if self.finished_games % 100 == 0 {
            self.print_results();
        }
    }

    fn print_results(&self) {
        println!("_______________________________________________________________________________________");
        println!("Results after {} games:", self.finished_games);
        
        let longest_name = self.opponents.values().map(|opponent| opponent.config.name.len()).max().unwrap_or(0);

        for (_, opponent) in self.opponents.iter() {
            let p = PentanomialModel::from(opponent.results);
            let total = opponent.results.total() * 2;
            if total == 0 {
                println!(" - {} No games finished yet", opponent.config.name);
                continue;
            }
            let score = p.score();
            let deviation = p.deviation(score);
            let variance = deviation.total();
            let pair_variance = variance / total as f64;

            let upper_bound = score + STD_NORM_DEV_95 * pair_variance.sqrt();
            let lower_bound = score - STD_NORM_DEV_95 * pair_variance.sqrt();

            let draw_ratio = p.d2;

            let elo_diff = score_to_norm_elo(score, variance);
            let elo_error = (score_to_norm_elo(upper_bound, variance) - score_to_norm_elo(lower_bound, variance)) / 2.0;
            
            let name_with_padding = format!("{:width$}", opponent.config.name, width = longest_name);

            println!(" - {}: Norm. Elo: {:>6.2} (+/- {:>5.2}) / Draw ratio: {:>5.2}% / Games: {} ({})",
                     name_with_padding, elo_diff, elo_error, draw_ratio * 100.0, total, opponent.matches);
        }
    }
}

fn score_to_norm_elo(score: f64, variance: f64) -> f64 {
    (score - 0.5) / (variance * 2.0).sqrt() * (800.0 / 10.0f64.ln())
}

#[derive(Clone)]
struct Opponent {
    config: EngineConfig,
    matches: usize,
    results: PentanomialCount,
}

impl Opponent {
    pub fn new(config: EngineConfig) -> Opponent {
        Opponent {
            config,
            matches: 0,
            results: PentanomialCount::default(),
        }
    }

    // matches returns the number of finished and ongoing matches
    pub fn matches(&self) -> usize {
        self.matches
    }
}