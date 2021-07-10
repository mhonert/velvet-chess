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

from argparse import ArgumentParser
from engine import Engine
from dataclasses import dataclass
import logging as log
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import sys
from typing import List
import copy
from common import TuningOption, Config


# Uses "Texel's Tuning Method" for tuning evaluation parameters
# see https://www.chessprogramming.org/Texel%27s_Tuning_Method for a detailed description of the method


@dataclass
class TestPosition:
    fen: str
    result: float
    score: int = 0


# Read test positions in format: FEN result
# result encodes the game result as a float between 0.0 and 1.0 ("1.0" for a white win, "0.0" for a black win, "0.5" for a draw)
def read_fens(fen_file) -> List[TestPosition]:
    test_positions = []
    with open(fen_file, 'r') as file:

        # Sample line:
        # rnbqkb1r/1p2ppp1/p2p1n2/2pP3p/4P3/5N2/PPP1QPPP/RNB1KB1R w KQkq - 0 1 1.0
        for line in file:
            fen = line[:line.rfind(" ")]
            result_str = line[line.rfind(" "):].strip()
            result = float(result_str)
            test_positions.append(TestPosition(fen, result))

    return test_positions


def run_engine(k: float, engine: Engine, tuning_options: List[TuningOption]) -> (int, float):
    try:

        if not engine.is_prepared:
            engine.is_prepared = True
            is_first = True
            for chunk in make_chunks(engine.test_positions, 512):
                fens = "prepare_eval "
                for pos in chunk:
                    if not is_first:
                        fens += ";"
                    else:
                        is_first = False
                    fens += pos.fen + ":" + str(pos.result)

                engine.send_command(fens)
                engine.wait_for_command("prepared")

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.value))

        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engine.send_command("eval " + str(k))
        result = engine.wait_for_command("result")

        [pos_str, error_str] = result[len("result "):].split(":")

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.prev_value))

        return int(pos_str), float(error_str)

    except Exception as error:
        engine.stop()
        log.error("Error occured during eval calculation", error)
        raise error


def apply_options(engine: Engine, tuning_options: List[TuningOption]) -> (int, float):
    try:

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.value))
                option.is_tuning = False

    except Exception as error:
        engine.stop()
        log.error("Error occured during apply_options", error)
        raise error


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


def run_pass(config: Config, tuning_options: List[TuningOption], engines: List[Engine]) -> float:
    futures = []

    log.debug("Starting pass")

    tick1 = time()
    with ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
        for engine in engines:
            futures.append(executor.submit(run_engine, config.k, engine, tuning_options))

        errors = .0
        positions = 0
        for future in as_completed(futures):
            if future.exception():
                log.exception("Worker was cancelled", future.exception())
                sys.exit("Worker was cancelled")

            if future.cancelled():
                sys.exit("Worker was cancelled - possible engine bug? try enabling the debug_log output and re-run the tuner")

            (p, e) = future.result()
            errors += e
            positions += p

    log.debug("Calc evals duration: %.2fs", time() - tick1)

    return errors / float(positions)


def run_apply_options(config: Config, tuning_options: List[TuningOption], engines: List[Engine]):
    futures = []

    with ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
        for engine in engines:
            futures.append(executor.submit(apply_options, engine, tuning_options))

        for future in as_completed(futures):
            if future.exception():
                log.exception("Worker was cancelled", future.exception())
                sys.exit("Worker was cancelled")

            if future.cancelled():
                sys.exit("Worker was cancelled - possible engine bug? try enabling the debug_log output and re-run the tuner")

            future.result()


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

                if option.maximum is not None:
                    result["max"] = option.maximum

                result["value"] = [option.value]
                results.append(result)
                result_by_name[option.orig_name] = result

        else:
            results.append({"name": option.name, "value": option.value})

    with open("tuning_result.yml", "w") as file:
        yaml.dump(results, file, default_flow_style=None, indent=2, sort_keys=False)


def tune_option(config: Config, tuning_options: List[TuningOption], engines: List[Engine], best_err: float, option: TuningOption):
    while True:
        option.is_tuning = True
        option.prev_value = option.value

        option.value += 1
        err1 = run_pass(config, tuning_options, engines)

        diff1 = best_err - err1
        if diff1 == .0:
            option.not_improved()
            return best_err

        option.value += 1
        err2 = run_pass(config, tuning_options, engines)
        diff2 = err1 - err2

        if diff1 - diff2 == .0:
            option.not_improved()
            return best_err

        steps = max(1, abs(int((diff1 / (diff1 - diff2)))))

        if diff1 < .0:
            steps = -steps

        option.value = option.prev_value + steps
        if option.minimum is not None:
            option.value = max(option.minimum, option.value)

        if option.maximum is not None:
            option.value = min(option.maximum, option.value)

        if option.value == option.prev_value:
            option.not_improved()
            return best_err

        new_err = run_pass(config, tuning_options, engines)
        if new_err < best_err:
            run_apply_options(config, tuning_options, engines)
            option.improved(new_err, best_err - new_err)
            best_err = new_err
        else:
            option.not_improved()
            return best_err


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = Config("config.yml")

    parser = ArgumentParser()
    parser.add_argument("-e", "--engine", dest="engine_cmd", help="Engine command", required=True)
    parser.add_argument("-t", "--testfile", dest="test_positions_file", help="Test position file", required=True)
    parser.add_argument("-c", "--concurrency", dest="concurrency", help="Concurrency - should be <= physical CPU core count", default=8)
    parser.add_argument("-d", "--debug", dest="debug", help="Enable debug logging", default=False)

    args = parser.parse_args()
    config.engine_cmd = args.engine_cmd
    config.test_positions_file = args.test_positions_file
    config.concurrent_workers = int(args.concurrency)
    config.debug_log = args.debug

    if config.debug_log:
        log.getLogger().setLevel(log.DEBUG)

    log.info("- use %i concurrent engine processes", config.concurrent_workers)

    # Scaling factor (calculated for Velvet Chess engine)
    config.k = 1.603

    log.info("Reading test positions ...")
    all_test_positions = read_fens(config.test_positions_file)

    log.info("Read %i test positions", len(all_test_positions))

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []

    for batch in make_batches(all_test_positions, config.concurrent_workers):
        engine = Engine(config.engine_cmd)

        engine.send_command("uci")
        engine.wait_for_command("uciok")

        for option in config.tuning_options:
            engine.send_command("setoption name {} value {}".format(option.name, option.value))

        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engines.append(engine)
        engine.test_positions = batch

    index_by_name = {}
    for index, option in enumerate(config.tuning_options):
        index_by_name[option.name] = index

    try:
        best_options = copy.deepcopy(config.tuning_options)

        start_time = time()

        for _, option in enumerate(best_options):
            option.is_tuning = True

        run_apply_options(config, best_options, engines)
        init_err = run_pass(config, best_options, engines)

        option_count = len(config.tuning_options)

        log.info("%.8f > Start tuning of %d options", init_err, option_count)

        prev_err = init_err
        last_iteration_with_improvement = 0
        for iteration in range(1, 100000000):
            for i, option in enumerate(best_options):
                if i - last_iteration_with_improvement == 1 and option.should_skip():
                    continue

                new_err = tune_option(config, best_options, engines, prev_err, option)

                improvement = prev_err - new_err
                if improvement > .0:
                    last_iteration_with_improvement = iteration
                    # update values in tuning config
                    for _, option_update in enumerate(best_options):
                        existing_option = config.tuning_options[index_by_name[option_update.name]]
                        existing_option.value = option_update.value
                    write_options(config.tuning_options)
                prev_err = new_err

            log.info("%.8f > Finished iteration %d - %ds", prev_err, iteration, time() - start_time)
            best_options.sort(key=lambda o: (o.improvement, o.improvements, -o.fails_in_row), reverse=True)

            if iteration - last_iteration_with_improvement > 8:
                log.info("%.8f > No more improvements during the last couple iterations", prev_err)
                break

    finally:
        for engine in engines:
            engine.stop()

    # Calculate K (scaling factor)
    # last_e = 10000000.0
    # k = 0.5
    # for step in [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]:
    #     log.info("Next step: %f", step)
    #     while True:
    #         previous_k = k
    #         k += step
    #         config.k = k
    #         e = run_pass(config, best_options, engines)
    #         log.info("Check k = %.8f -> e = %.8f, last_e = %.8f", k, e, last_e)
    #         if e > last_e:
    #             k = previous_k
    #             break
    #         last_e = e
    #
    # log.info("=> k = %.8f", k)
    # return

# Main
if __name__ == "__main__":
    main()
