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
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct EngineConfigs(HashMap<String, EngineConfig>);

#[derive(Clone, Debug, Deserialize)]
pub struct EngineConfig {
    #[serde(skip)]
    pub id: usize,

    cmd: String,

    #[serde(default)]
    options: Vec<(String, String)>,

    #[serde(default)]
    init_commands: Vec<String>,
}

pub fn read_engine_configs(file_path: String) -> anyhow::Result<EngineConfigs> {
    let config_str = std::fs::read_to_string(file_path)?;
    let mut config: EngineConfigs = toml::from_str(&config_str)?;

    config.0.iter_mut().enumerate().for_each(|(i, (_, v))| v.id = i + 1);
    Ok(config)
}

#[derive(Clone, Debug, Deserialize)]
pub struct TournamentConfig {
    pub tc: f32,
    
    #[serde(skip)]
    pub inc: f32,
    
    pub book: String,
    pub engines: String,
    pub challenger: String,
    pub opponents: Vec<String>,
}

pub fn read_tournament_config(file_path: String) -> anyhow::Result<TournamentConfig> {
    let config_str = std::fs::read_to_string(file_path)?;
    let mut config: TournamentConfig = toml::from_str(&config_str)?;
    
    config.inc = config.tc / 100.0;
    Ok(config)
}