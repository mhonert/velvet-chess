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
import argparse
import logging as log
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from concurrent.futures._base import FIRST_COMPLETED
from random import randint
from time import time
import sys
from typing import List
import chess


# Read test FEN positions
def read_fens(fen_file) -> List[str]:
    test_positions = []
    with open(fen_file, 'r') as file:

        for line in file:
            fen = line.strip()
            if fen != "":
                test_positions.append(fen)

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


def run_engine(engine: Engine, test_positions: List[str]):
    results = []

    try:

        start_nodes = 20000

        for pos in test_positions:
            pos_results = .0
            pos_result_count = 0
            remaining_iterations = 4
            previous_result = .0
            extended = False

            mirrored_fen = None

            while remaining_iterations > 0:
                remaining_iterations -= 1
                moves = []
                board = chess.Board(pos)

                if mirrored_fen is None:
                    mirrored_fen = board.mirror().fen()

                if board.is_game_over():
                    continue

                engine.send_command("ucinewgame")

                engine.send_command("isready")
                engine.wait_for_command("readyok")

                nodes = randint(start_nodes, start_nodes + start_nodes // 10) * (20 + pos_result_count) // 20

                claim_draw = True

                while claim_draw or len(moves) < 400:
                    if len(moves) > 0:
                        engine.send_command("position fen " + pos + " moves " + " ".join(moves))
                    else:
                        engine.send_command("position fen " + pos)

                    engine.send_command("go nodes " + str(nodes))

                    response = engine.wait_for_command("bestmove").split(' ')
                    best_move = response[1]

                    board.push_uci(best_move)
                    if board.is_game_over(claim_draw=claim_draw):
                        break

                    moves.append(best_move)

                    response_move = response[3] if len(response) == 4 else None
                    if response_move is None:
                        continue

                    board.push_uci(response_move)
                    if board.is_game_over(claim_draw=claim_draw):
                        break
                    moves.append(response_move)

                result = 0.5
                if board.is_checkmate():
                    result = to_result_value(board.result())

                pos_results += result

                if not extended and pos_result_count > 0:
                    if previous_result != result:
                        remaining_iterations += 4
                        extended = True

                previous_result = result
                pos_result_count += 1

            if pos_result_count > 0:
                pos_result = pos_results / float(pos_result_count)
                results.append(pos + " " + str(pos_result) + "\n")
                results.append(mirrored_fen + " " + str(1.0 - pos_result) + "\n")

    except subprocess.TimeoutExpired as error:
        engine.stop()
        raise error

    except Exception as error:
        log.error(str(error))

    return results


def to_result_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return .0
    else:
        return 0.5


def mirror(result: str) -> str:
    if result == "1-0":
        return "0-1"
    elif result == "0-1":
        return "1-0"
    return result


# Split list of test positions into "batch_count" batches
def make_batches(positions: List[str], batch_count: int) -> List[List[str]]:
    max_length = len(positions)
    batch_size = max(1, max_length // batch_count)
    return make_chunks(positions, batch_size)


# Split list of test positions into chunks of size "chunk_size"
def make_chunks(positions: List[str], chunk_size: int) -> List[List[str]]:
    max_length = len(positions)
    for i in range(0, max_length, chunk_size):
        yield positions[i:min(i + chunk_size, max_length)]


def run_pass(engines: List[Engine], concurrency: int, test_positions: List[str]) -> List[str]:
    futures = []

    log.debug("Starting pass")

    results = []

    total_count = len(test_positions)
    batches = list(make_batches(test_positions, concurrency * 16))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i in range(concurrency):
            engine = engines[i]
            batch = batches.pop()
            futures.append(executor.submit(run_engine, engine, batch))

        while len(futures) > 0:
            wait(futures, return_when=FIRST_COMPLETED)
            for i, future in enumerate(futures):
                if future.done():
                    if future.exception():
                        log.exception("Worker was cancelled", future.exception())
                        sys.exit("Worker was cancelled")

                    if future.cancelled():
                        sys.exit("Worker was cancelled - possible engine bug? try enabling the debug_log output and re-run the tuner")

                    result = future.result()
                    results.extend(result)

                    log.info("%d%% completed", len(results) * 50 / total_count)

                    if len(batches) > 0:
                        engine = engines[i]
                        batch = batches.pop()
                        futures[i] = executor.submit(run_engine, engine, batch)
                    else:
                        del futures[i]

    log.debug("Pass completed")

    return results


def main():
    log.basicConfig(stream=sys.stdout, level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', dest='engine_cmd')
    parser.add_argument('--tb', dest='tb_path')
    args = parser.parse_args()

    log.info("Reading test positions ...")

    concurrency = 7

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []
    for i in range(concurrency + 1):
        engine = Engine(args.engine_cmd)
        engine.send_command("uci")
        engine.wait_for_command("uciok")

        engine.send_command("setoption name Hash value 512")
        if args.tb_path:
            engine.send_command("setoption name SyzygyPath value " + args.tb_path)
        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engines.append(engine)

    log.info("Engines started")

    try:

        for i in range(1, 5 + 1):
            test_positions = read_fens("fen/set_filtered_" + str(i) + ".fen")
            log.info("Evaluating test position set %d ...", i)
            start = time()

            results = run_pass(engines, concurrency, test_positions)

            with open("fen/set_eval_" + str(i) + ".fen", "w", encoding="utf-8") as result_file:
                result_file.writelines(results)

            log.info("Duration         : %.2fs", time() - start)

    finally:
        for engine in engines:
            engine.stop()


# Main
if __name__ == "__main__":
    main()
