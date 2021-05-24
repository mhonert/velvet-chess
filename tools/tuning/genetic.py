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
import pickle
from argparse import ArgumentParser
from engine import Engine
from dataclasses import dataclass
import logging as log
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import sys
from typing import List
from common import Config
import random
import os.path


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


@dataclass
class GeneticProgram:
    code: int  # 128-bit
    data: List[int]  # fixed length: 6x 64-bit values
    score_increment: int  # 32-bit signed integer
    score_raise: int  # 32-bit signed integer
    result: float
    solution_size: int = 0


@dataclass
class Team:
    programs: List[GeneticProgram]
    result: float
    solution_size: int = 0


@dataclass
class Community:
    teams: List[Team]
    generation: 1


def run_engine(k: float, engine: Engine, programs: List[GeneticProgram]) -> (int, float):
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

        engine.send_command("clear_genetic_programs")
        for program in programs:
            engine.send_command("add_genetic_program {} {} {} {} {} {} {} {} {} {} {}".format(program.code, *program.data, program.score_increment, program.score_raise))
        engine.send_command("compile_genetic_programs")

        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engine.send_command("eval " + str(k))
        result = engine.wait_for_command("result")

        [pos_str, error_str] = result[len("result "):].split(":")

        return int(pos_str), float(error_str)

    except Exception as error:
        engine.stop()
        log.error("Error occured during eval calculation", error)
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


def run_pass(config: Config, programs: List[GeneticProgram], engines: List[Engine]) -> float:
    futures = []

    log.debug("Starting pass")

    tick1 = time()
    with ThreadPoolExecutor(max_workers=config.concurrent_workers) as executor:
        for engine in engines:
            futures.append(executor.submit(run_engine, config.k, engine, programs))

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


def write_programs(programs: List[GeneticProgram]):
    results = []
    for program in programs:
        result = {"code": program.code,
                  "data": program.data,
                  "score_increment": program.score_increment,
                  "score_raise": program.score_raise,
                  "result": program.result}
        results.append(result)

    with open("genetic_evals.yml", "w") as file:
        yaml.dump(results, file, default_flow_style=None, indent=2, sort_keys=False)


def create_new_generation(engine: Engine, curr_gen: List[GeneticProgram]):
    engine.send_command("clear_genetic_programs")
    for program in curr_gen:
        engine.send_command("add_genetic_program {} {} {} {} {} {} {} {} {} {} {}".format(program.code, *program.data, program.score_increment, program.score_raise))

    engine.send_command("new_genetic_generation")
    return read_generation_response(engine)


def init_generation(engine: Engine, pop_size: int):
    engine.send_command("init_genetic_generation " + str(pop_size))
    return read_generation_response(engine)


def read_generation_response(engine: Engine):
    result = engine.wait_for_command("result")

    new_gen = []

    for r in result[len("result "):].split(";"):
        tmp = r.split(",")
        if len(tmp) != 12:
            break

        program_result = [int(x) for x in tmp]
        new_gen.append(GeneticProgram(code=program_result[0], data=program_result[1:9], result=.0, score_increment=program_result[9], score_raise=program_result[10], solution_size=program_result[11]))

    return new_gen


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
    all_test_positions = random.sample(all_test_positions, 200000)

    log.info("Read %i test positions", len(all_test_positions))

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []

    for batch in make_batches(all_test_positions, config.concurrent_workers):
        engine = Engine(config.engine_cmd)

        engine.send_command("uci")
        engine.wait_for_command("uciok")

        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engines.append(engine)
        engine.test_positions = batch

    try:
        init_err = run_pass(config, [], engines)
        log.info("Reference error: %.8f", init_err)

        best_err = 1.0

        random.seed()

        if os.path.isfile("community.bin"):
            # Continue with existing community file
            with open("community.bin", "rb") as file:
                community = pickle.load(file)

            team_size = len(community.teams[0].programs)
            log.info("Continue evolving existing community (gen. %d) of %d teams with %d members", community.generation, len(community.teams), team_size)

        else:
            # Generate new community
            community_size = 256
            team_size = 64
            log.info("Generate initial community of %d teams with %d members", community_size, team_size)

            teams = []
            for i in range(community_size):
                members = init_generation(engines[0], team_size)
                team = Team(programs=members, solution_size=0, result=.0)
                teams.append(team)

            community = Community(teams=teams, generation=1)

        improved = False

        while True:

            eval_start_time = time()
            for team in community.teams:
                new_err = run_pass(config, team.programs, engines)
                team.result = new_err
                team.solution_size = 0
                for program in team.programs:
                    program.result = new_err
                    team.solution_size += program.solution_size

                if new_err < best_err:
                    best_err = new_err
                    improved = True
            eval_duration = time() - eval_start_time

            community.teams.sort(key=lambda t: (t.result, t.solution_size))

            for program in community.teams[0].programs:
                log.info("- %d (%d Instr.), %d/%d", program.code, program.solution_size, program.score_increment, program.score_raise)

            if improved:
                write_programs(community.teams[0].programs)
                improved = False

            # Next generation
            gen_start_time = time()
            for i in range(team_size):
                members = []
                for team in community.teams:
                    members.append(team.programs[i])

                members = create_new_generation(engines[0], members)

                j = 0
                for team in community.teams:
                    team.programs[i] = members[j]
                    j += 1

            community.generation += 1

            with open("community.bin", "wb") as file:
                pickle.dump(community, file, pickle.HIGHEST_PROTOCOL)

            gen_duration = time() - gen_start_time
            log.info("%.8f > Finished generation %d: (%d Instr.) - %ds / %ds / %ds", best_err, community.generation, community.teams[0].solution_size, eval_duration + gen_duration, eval_duration, gen_duration)


    finally:
        for engine in engines:
            engine.stop()


# Main
if __name__ == "__main__":
    main()
