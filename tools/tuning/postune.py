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
from random import shuffle

from engine import Engine
from dataclasses import dataclass
import logging as log
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import sys
from typing import List, Dict
import copy
from common import TuningOption, Config
import chess


def parse_move_score(board, entry) -> (str, int):
    # Sample entry: Bb5=9
    parts = entry.strip().rsplit("=", maxsplit=1)
    if len(parts) != 2:
        return None

    san_move = parts[0]
    score = int(parts[1])

    try:
        move = board.parse_san(san_move)
    except ValueError:
        log.warning("Invalid entry: %s", entry)
        return None

    return move.uci(), score


@dataclass
class TestPosition:
    fen: str
    best_score: int
    solutions: Dict[str, int]


def read_epd(epd_file) -> List[TestPosition]:
    test_positions = []
    i = 0
    with open(epd_file, 'r') as file:
        # Sample line:
        # r2qk2r/pp1n1p2/2n1p2p/2bp3P/7p/2N1PQ1P/PPPB1P2/R3KB1R w KQkq - bm O-O-O; bm O-O-O; c0 "O-O-O=10, Bb5=9, Be2=9, Rg1=8";
        for line in file:
            i += 1
            if i % 2 != 0:
                continue
            parts = line.rstrip().split(";")
            if len(parts) == 5:
                del parts[1]
            if len(parts) != 4:
                log.warning("Invalid line with %i parts: %s", len(parts), line)
                continue

            fen = parts[0]
            if "bm " in fen:
                fen = fen[:fen.rindex("bm ")]  # remove best move part 'bm ...'

            if "id " in fen:
                fen = fen[:fen.rindex("id ")]  # remove id part 'id ...'

            fen += "0 1"

            board = chess.Board(fen)
            solution_str = parts[2]
            if not solution_str.startswith(' c0 "'):
                log.warning("Missing ' c0 \"' comment in line %s", line)

            solution_str = solution_str[5:len(solution_str) - 1]

            solutions = list(filter(None.__ne__, [parse_move_score(board, entry) for entry in solution_str.split(",")]))

            if len(solutions) > 0:
                test_positions.append(TestPosition(fen, solutions[0][1], dict(solutions)))
            else:
                log.warning("Skipped position, because it has no valid solution comment: %s", line)

    return test_positions


@dataclass
class TuningResult:
    value: int
    score: int


def run_engine(engine: Engine, tuning_options: List[TuningOption]) -> (int, int):
    try:

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.value))

        score = 0
        max_score = 0

        # shuffle(engine.test_positions)

        for pos in engine.test_positions[:250]:
            engine.send_command("ucinewgame")

            engine.send_command("isready")
            engine.wait_for_command("readyok")

            engine.send_command("position fen " + pos.fen)

            engine.send_command("go movetime 500")
            result = engine.wait_for_command("bestmove")

            engine_solution = result[len("bestmove "):].split(" ")[0]
            solution_score = pos.solutions.get(engine_solution, 0)

            score += solution_score
            max_score += pos.best_score

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.prev_value))

        return (score, max_score)

    except Exception as error:
        engine.stop()
        log.error("Error occured during eval calculation", error)
        raise error


def apply_options(engine: Engine, tuning_options: List[TuningOption]) -> (int, float):
    try:

        for option in tuning_options:
            if option.is_tuning:
                engine.send_command("setoption name {} value {}".format(option.name, option.value))

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

    total_max_score = 0
    total_score = 0

    tick1 = time()

    with ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
        for engine in engines:
            futures.append(executor.submit(run_engine, engine, tuning_options))

        for future in as_completed(futures):
            if future.exception():
                log.exception("Worker was cancelled", future.exception())
                sys.exit("Worker was cancelled")

            if future.cancelled():
                sys.exit("Worker was cancelled - possible engine bug? try enabling the debug_log output and re-run the tuner")

            (score, max_score) = future.result()
            total_score += score
            total_max_score += max_score

    log.info("Calc evals duration: %.2fs", time() - tick1)

    return 1.0 - (float(total_score) / float(total_max_score))


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


def tune_option(config: Config, tuning_options: List[TuningOption], engines: List[Engine], best_err: float, resolution: int, option: TuningOption):
    option.is_tuning = True
    option.prev_value = option.value

    for _ in range(2):
        option.value = (option.prev_value // resolution + option.steps * option.direction) * resolution

        if option.minimum is not None:
            if option.value < option.minimum:
                option.value = option.minimum
        if option.maximum is not None:
            if option.value > option.maximum:
                option.value = option.maximum

        err = run_pass(config, tuning_options, engines)
        diff = best_err - err

        if diff > .0:
            run_apply_options(config, tuning_options, engines)
            option.is_tuning = False

            option.improved(diff)
            return err
        elif diff < .0:
            option.not_improved(True)
        else:
            option.not_improved(True)
            # break

        option.direction *= -1

    option.is_tuning = False
    return best_err


def tune_options(config: Config, tuning_options: List[TuningOption], engines: List[Engine], best_err: float,
                 resolution: int, summed_improvements: float, options: List[TuningOption]):

    for option in options:
        option.is_tuning = True
        option.prev_value = option.value
        option.value = (option.prev_value // resolution + option.steps * option.direction) * resolution

        if option.minimum is not None:
            if option.value < option.minimum:
                option.value = option.minimum
        if option.maximum is not None:
            if option.value > option.maximum:
                option.value = option.maximum

    err = run_pass(config, tuning_options, engines)
    diff = best_err - err

    if diff > .0:
        run_apply_options(config, tuning_options, engines)
        for option in options:
            option.is_tuning = False
            option.improved(diff * option.improvement / summed_improvements)
            option.steps = 1
        return err
    elif diff < .0:
        for option in options:
            option.is_tuning = False
            option.not_improved(False)
    else:
        for option in options:
            option.is_tuning = False
            option.not_improved(False)
            option.improvement = .0

    return best_err


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = Config("config.yml")
    if config.debug_log:
        log.getLogger().setLevel(log.DEBUG)
    log.info("- use %i concurrent engine processes", config.concurrent_workers)

    log.info("Reading test positions ...")
    all_test_positions = read_epd(config.test_positions_file)

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

        resolution = config.starting_resolution

        adjust_values = True

        while adjust_values:
            adjust_values = False
            for _, option in enumerate(best_options):
                new_value = option.value // resolution
                if new_value * resolution != option.value:
                    log.info("Increase starting resolution")
                    resolution //= 10
                    if resolution < 10:
                        resolution = 1
                    adjust_values = True
                    break

        tick = time()

        improvements = []

        iterations = 0

        low_improvements = 0

        local_best_err = run_pass(config, best_options, engines)

        init_err = local_best_err

        option_count = len(config.tuning_options)
        log.info("Starting error: %f", init_err)
        log.info("Tuning %d options", option_count)
        log.info("Start with resolution %d", resolution)

        init_phase = -2

        any_options_checked = False

        is_first = True

        while low_improvements <= 2 or resolution > 1:
            iterations += 1

            write_options(config.tuning_options)

            prev_err = local_best_err
            if init_phase < 0:
                any_options_checked = False
                init_phase += 1
                improvement_count = 0
                log.info("Re-init phase %d", low_improvements)
                for i, option in enumerate(best_options):
                    if option.skip:
                        continue
                    log.info("Option %s, %f, %d", option.name, option.improvement, option.improvements)
                    option.iteration = iterations
                    option.has_improved = False
                    new_local_best_err = tune_option(config, best_options, engines, local_best_err, resolution, option)
                    if new_local_best_err < local_best_err:
                        local_best_err = new_local_best_err
                        improvement_count += 1

                    if not is_first and low_improvements == 0 and improvement_count >= 4 and option.improvements == 0:
                        log.info("Leave re-init phase")
                        break

                    if is_first and init_phase == 0 and option.improvements == 0:
                        log.info("Leave initial init phase")
                        break;

                for i, option in enumerate(best_options):
                    if option.improvements > 1 and not option.has_improved:
                        option.improvements -= 1

            else:
                is_first = False
                count = 0
                options = []
                start_rel = .0
                summed_improvements = .0

                for i, option in enumerate(best_options):
                    # log.info("%s: %f", option.name, option.rel_improvement)
                    if not option.has_improved or option.improvement <= .0 or option.skip:
                        continue
                    if count == 0:
                        option.iteration = iterations
                        option.has_improved = False
                        start_rel = option.rel_improvement
                        summed_improvements += option.improvement
                        options.append(option)
                        count += 1
                        continue

                    if option.steps > 1:
                        break

                    if start_rel - option.rel_improvement >= 1.0:
                        break

                    if option.rel_improvement < 1.0 and start_rel - option.rel_improvement >= .05:
                        break

                    if option.rel_improvement < .5 and start_rel - option.rel_improvement >= .025:
                        break

                    if start_rel <= 0.1 or start_rel - option.rel_improvement >= .1 or option.rel_improvement < 0.00000001:
                        break

                    option.iteration = iterations
                    option.has_improved = False
                    summed_improvements += option.improvement
                    options.append(option)
                    count += 1

                    if count >= 128:
                        break

                if count > 0:
                    any_options_checked = True
                    if count > 1:
                        log.info("Tuning %d options [%f, %f]", count, options[0].rel_improvement, options[-1].rel_improvement)
                        new_local_best_err = tune_options(config, best_options, engines, local_best_err, resolution, summed_improvements, options)
                        if new_local_best_err < local_best_err:
                            local_best_err = new_local_best_err
                        else:
                            for option in options:
                                new_local_best_err = tune_option(config, best_options, engines, local_best_err, resolution, option)
                                if new_local_best_err < local_best_err:
                                    local_best_err = new_local_best_err

                    else:
                        new_local_best_err = tune_option(config, best_options, engines, local_best_err, resolution, options[0])
                        if new_local_best_err < local_best_err:
                            local_best_err = new_local_best_err

                else:
                    any_options_checked = False

            best_options.sort(key=lambda o: (o.rel_improvement, o.improvement, o.improvements, -o.iteration), reverse=True)

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

            if improvement == 0 and not any_options_checked:
                if resolution > 1:
                    resolution //= 10
                    if resolution < 10:
                        resolution = 1

                    log.info("Continue with resolution %d", resolution)

                low_improvements += 1
                if init_phase >= 0:
                    init_phase = -1

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
