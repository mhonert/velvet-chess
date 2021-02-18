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
    history: List[float]
    is_part: bool = False
    orig_name: str = ""
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    skip_early: int = 64
    steps: int = 1
    direction: int = 1
    improvement: float = .0
    has_improved: bool = True
    iteration: int = -1
    fails: int = 0
    prev_value: int = 0
    rel_improvement: float = .0
    improvements: int = 0
    skip: bool = False
    ever_improved: bool = False
    is_tuning = False

    def improved(self, diff):
        self.ever_improved = True
        self.fails = 0
        self.improvement = diff

        self.history.append(diff)
        self.adjust_steps()
        self.improvements += 1

        self.has_improved = True

        log.info(" > Change %s from %d to %d (%d): %.8f", self.name, self.prev_value, self.value, self.steps * self.direction, self.improvement)

    def not_improved(self, reset):
        self.fails += 1
        if not self.ever_improved and self.fails >= self.skip_early:
            self.skip = True

        self.value = self.prev_value  # Restore previous value
        self.has_improved = False
        if not reset:
            return

        if self.fails == 1:
            self.improvement /= 100
        else:
            self.improvement = 0

        self.history = self.history[-1:]
        self.rel_improvement = 0
        self.steps = 1

    def adjust_steps(self):
        if len(self.history) < 2:
            return

        log.debug("%s: %d - %.8f", self.name, self.value, self.improvement)

        self.rel_improvement = self.history[-1] / self.history[-2]
        steps = self.history[-1] * (len(self.history) - 1) / self.history[0]
        self.steps = min(10, max(1, int(steps)))
        # log.debug("%s: %f", self.name, self.steps)


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

