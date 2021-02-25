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
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging as log
import yaml


@dataclass
class TuningOption:
    name: str
    value: int
    prev_value: int
    is_part: bool = False
    orig_name: str = ""
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    skip_early: int = 4
    steps: int = 1
    iteration: int = -1
    fails_in_row: int = 0
    improvements: int = 0
    improvement: float = .0
    skip: bool = False
    is_tuning = False

    def improved(self, err: float, improvement):
        self.improvements += 1
        self.is_tuning = False
        self.improvement = improvement
        self.fails_in_row = 0

        log.info("%.8f > Change %s from %d to %d", err, self.name, self.prev_value, self.value)

    def not_improved(self):
        self.fails_in_row += 1
        self.value = self.prev_value
        self.is_tuning = False
        self.improvement *= 0.5

    def should_skip(self):
        return self.fails_in_row >= self.skip_early


def get_config(cfg: Dict, key: str, msg: str):
    value = cfg.get(key)
    if value is None:
        sys.exit(msg)
    return value


@dataclass
class Config:
    engine_cmd: str
    debug_log: bool
    test_positions_file: str
    concurrent_workers: int
    starting_resolution: int
    k: float
    tuning_options: List[TuningOption]

    def __init__(self, config_file: str, skip_excluded_options = True):
        log.info("Reading configuration ...")

        cfg_stream = open(config_file, "r")

        cfg = yaml.safe_load(cfg_stream)

        options = get_config(cfg, "options", "Missing 'options' configuration")

        included_options = set(options.get("tune", []))

        tuning_cfg = cfg.get('tuning')
        self.tuning_options = []
        if tuning_cfg is not None:
            for t in tuning_cfg:
                if not skip_excluded_options or t["name"] in included_options:
                    if skip_excluded_options:
                        included_options.remove(t["name"])
                    value = t["value"]
                    if type(value) is list:
                        for index, v in enumerate(value):
                            option = TuningOption(t["name"] + str(index), int(v), [], True, t["name"], t.get("min", None), t.get("max", None), t.get("skip_early", 64))
                            self.tuning_options.append(option)

                    else:
                        option = TuningOption(t["name"], int(value), [], False, t["name"], t.get("min", None), t.get("max", None), t.get("skip_early", 64))
                        self.tuning_options.append(option)

            if skip_excluded_options and len(included_options) > 0:
                log.warning("Options %s not found", included_options)
        cfg_stream.close()

