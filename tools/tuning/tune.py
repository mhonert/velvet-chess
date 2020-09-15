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

from dataclasses import dataclass
import logging as log
import yaml
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import sys
from typing import List, Callable
import copy
from random import shuffle, randint
from common import TuningOption, Config


# Uses "Texel's Tuning Method" for tuning evaluation parameters
# see https://www.chessprogramming.org/Texel%27s_Tuning_Method for a detailed description of the method


# Scaling factor (calculated for Velvet Chess engine)
K = 1.342224

@dataclass
class TestPosition:
    fen: str
    result: float
    score: int = 0


# Read test positions in format: FEN result
# result may be "1-0" for a white win, "0-1" for a black win or "1/2" for a draw
def read_fens(fen_file) -> List[TestPosition]:
    test_positions = []
    with open(fen_file, 'r') as file:

        # Sample line:
        # rnbqkb1r/1p2ppp1/p2p1n2/2pP3p/4P3/5N2/PPP1QPPP/RNB1KB1R w KQkq - 0 1 1-0
        for line in file:
            fen = line[:-5]
            result_str = line[-4:].strip()
            result = 1 if result_str == "1-0" else 0 if result_str == "0-1" else 0.5
            test_positions.append(TestPosition(fen, result))

    return test_positions


class Engine:
    def __init__(self, engine_cmd):
        self.process = subprocess.Popen([engine_cmd], bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, universal_newlines=True)

    def stop(self):
        log.debug("Stopping engine instance")
        self.process.communicate("quit\n", timeout=2)

        self.process.kill()
        self.process.communicate()

    def send_command(self, cmd):
        log.debug(">>> " + cmd)
        self.process.stdin.write(cmd + "\n")

    def wait_for_command(self, cmd):
        for line in self.process.stdout:
            line = line.rstrip()
            log.debug("<<< " + line)
            if cmd in line:
                return line


def run_engine(engine: Engine, tuning_options: List[TuningOption], test_positions: List[TestPosition]):
    results = []

    try:
        engine.send_command("uci")
        engine.wait_for_command("uciok")

        for option in tuning_options:
            engine.send_command("setoption name {} value {}".format(option.name, option.value))

        engine.send_command("isready")
        engine.wait_for_command("readyok")

        for chunk in make_chunks(test_positions, 256):
            fens = "eval "
            is_first = True
            for pos in chunk:
                if not is_first:
                    fens += ";"
                fens += pos.fen
                is_first = False

            engine.send_command(fens)

            result = engine.wait_for_command("scores")

            scores = [int(score) for score in result[len("scores "):].split(";")]
            assert len(scores) == len(chunk)

            for i in range(len(scores)):
                chunk[i].score = scores[i]

    except subprocess.TimeoutExpired as error:
        engine.stop()
        raise error

    except Exception as error:
        log.error(str(error))

    return results


# Split list of test positions into "batch_count" batches
def make_batches(positions: List[TestPosition], batch_count: int) -> List[List[TestPosition]]:
    max_length = len(positions)
    batch_size = max(1, max_length // batch_count)
    return make_chunks(positions, batch_size)


# Split list of test positions into chunks of size "chunk_size"
def make_chunks(positions: List[TestPosition], chunk_size: int) -> List[List[TestPosition]]:
    max_length = len(positions)
    for i in range(0, max_length, chunk_size):
        yield positions[i:min(i + chunk_size, max_length)]


def run_pass(config: Config, tuning_options: List[TuningOption], k: float, engines: List[Engine], test_positions: List[TestPosition]) -> float:
    futures = []

    log.debug("Starting pass")

    with ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
        worker_id = 1
        for batch in make_batches(test_positions, config.concurrent_workers):
            engine = engines[worker_id - 1]
            futures.append(executor.submit(run_engine, engine, tuning_options, batch))
            worker_id += 1

        for future in as_completed(futures):
            if future.exception():
                log.exception("Worker was cancelled", future.exception())
                sys.exit("Worker was cancelled")

            if future.cancelled():
                sys.exit("Worker was cancelled - possible engine bug? try enabling the debug_log output and re-run the tuner")

    log.debug("Pass completed")

    e = calc_avg_error(k, test_positions)

    return e


def calc_avg_error(k: float, positions: List[TestPosition]) -> float:
    errors = .0
    for pos in positions:
        win_probability = 1.0 / (1.0 + 10.0 ** (-pos.score * k / 400.0))
        error = pos.result - win_probability
        errors += error * error
    return errors / float(len(positions))


def write_options(options: List[TuningOption]):
    results = []
    result_by_name = {}
    for option in options:
        if option.is_part:
            if option.orig_name in result_by_name:
                result_by_name[option.orig_name]["value"].append(option.value)

            else:
                result = {"name": option.orig_name}
                if option.minimum is not None:
                    result["min"] = option.minimum

                result["value"] = [option.value]
                results.append(result)
                result_by_name[option.orig_name] = result

        else:
            results.append({"name": option.name, "value": option.value})

    with open("tuning_result.yml", "w") as file:
        yaml.dump(results, file, default_flow_style=None, indent=2, sort_keys=False)


def roll_testpositions(all: List[TestPosition], start: int, window_size: int) -> (int, List[TestPosition]):
    sub_set = all[start:(start + window_size)]
    missing_len = window_size - len(sub_set)
    if missing_len > 0:
        sub_set += all[0:missing_len]
        return missing_len // 4, sub_set
    return (start + window_size // 5 + randint(0, window_size // 10)) % len(all), sub_set


def create_pass_runner(config: Config, tuning_options: List[TuningOption], k: float, engines: List[Engine]):
    def run(test_positions: List[TestPosition]):
        return run_pass(config, tuning_options, k, engines, test_positions)

    return run


def create_option_tuner(run_test: Callable[[List[TestPosition]], float]):
    def run(best_err: float, option: TuningOption, test_positions: List[TestPosition]):
        prev_value = option.value

        for _ in range(2):
            option.value = prev_value + option.steps * option.direction
            err = run_test(test_positions)
            diff = best_err - err

            if diff > 0:
                option.improved(prev_value, diff)
                return err
            elif diff < 0:
                option.not_improved(prev_value)
            else:
                option.not_improved(prev_value)
                option.improvement = -1.0
                break

            option.direction *= -1

        return best_err

    return run


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = Config("config.yml")
    if config.debug_log:
        log.getLogger().setLevel(log.DEBUG)
    log.info("- use %i concurrent engine processes", config.concurrent_workers)

    log.info("Reading test positions ...")
    all_test_positions = read_fens(config.test_positions_file)

    log.info("Read %i test positions", len(all_test_positions))

    log.info("Shuffling test positions ...")
    shuffle(all_test_positions)

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []
    for i in range(config.concurrent_workers + 1):
        engine = Engine(config.engine_cmd)
        engines.append(engine)

    index_by_name = {}
    for index, option in enumerate(config.tuning_options):
        index_by_name[option.name] = index

    try:
        best_options = copy.deepcopy(config.tuning_options)
        run_test = create_pass_runner(config, best_options, K, engines)
        tune_option = create_option_tuner(run_test)

        tick = time()

        index = 0
        improvements = []

        iterations = 0

        # window_size = 720 * 1000

        low_improvements = 0
        is_first_iteration = True

        # (index, test_positions) = roll_testpositions(all_test_positions, index, window_size)
        test_positions = all_test_positions
        local_best_err = run_test(test_positions)
        init_err = local_best_err

        option_count = len(config.tuning_options)
        log.info("Starting error: %f", init_err)
        log.info("Tuning %d options", option_count)

        while low_improvements < option_count:
            iterations += 1
            # if iterations % 20 == 0:
            #     # shuffle(all_test_positions)
            #     (index, test_positions) = roll_testpositions(all_test_positions, index, window_size)
            #     local_best_err = run_test(test_positions)

            write_options(config.tuning_options)

            prev_err = local_best_err
            for i, option in enumerate(best_options):

                option.iteration = iterations
                option.has_improved = False
                new_local_best_err = tune_option(local_best_err, option, test_positions)
                if new_local_best_err < local_best_err:
                    local_best_err = new_local_best_err
                    if not is_first_iteration:
                        break

            is_first_iteration = False

            best_options.sort(key=lambda o: (o.improvement, -o.iteration), reverse=True)

            improvement = prev_err - local_best_err

            # update values in tuning config
            for _, option_update in enumerate(best_options):
                existing_option = config.tuning_options[index_by_name[option_update.name]]
                existing_option.value = option_update.value
                existing_option.steps = option_update.steps
                existing_option.direction = option_update.direction
            write_options(config.tuning_options)

            improvements.append(improvement)

            avg_improvement = sum(improvements) / len(improvements)
            log.info("%d. / Err.: %.8f / Avg. improvement: %.8f / Last improvement: %.8f", iterations, prev_err, avg_improvement, improvement)

            if len(improvements) > 10:
                improvements = improvements[1:]

                if improvement < 0.0000000005 and avg_improvement < 0.00000000005:
                    low_improvements += 1
                else:
                    low_improvements = 0

        log.info("Avg. error before tuning: %f", init_err)
        log.info("Avg. error after tuning : %f", local_best_err)
        log.info("Tuning duration         : %.2fs", time() - tick)

    finally:
        for engine in engines:
            engine.stop()


# Main
if __name__ == "__main__":
    main()
