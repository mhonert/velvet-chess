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

import logging as log
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from concurrent.futures._base import FIRST_COMPLETED
from time import time
import sys
from typing import List
import chess


# Uses "Texel's Tuning Method" for tuning evaluation parameters
# see https://www.chessprogramming.org/Texel%27s_Tuning_Method for a detailed description of the method


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

        start_depth = 14

        for pos in test_positions:
            moves = []
            board = chess.Board(pos)
            mirror_fen = board.mirror().fen()

            engine.send_command("ucinewgame")

            engine.send_command("isready")
            engine.wait_for_command("readyok")

            depth = start_depth

            claim_draw = True
            is_first_move = True
            play_ponder_move = False
            skip = False

            while claim_draw or len(moves) < 400:
                if len(moves) > 0:
                    engine.send_command("position fen " + pos + " moves " + " ".join(moves))
                else:
                    engine.send_command("position fen " + pos)

                engine.send_command("go depth " + str(depth))

                if is_first_move:
                    info = engine.wait_for_command("info depth " + str(depth))
                    if info is not None:
                        infos = info.split(" ")
                        try:
                            score_index = infos.index("score")
                            score_type = infos[score_index + 1]
                            if score_type == "cp":
                                score = abs(int(infos[score_index + 2]))
                                if score >= 500:
                                    claim_draw = False
                                if score >= 300:
                                    depth -= 10
                                    claim_draw = False
                                elif score >= 200:
                                    depth -= 8
                                    claim_draw = False
                                elif score >= 150:
                                    depth -= 5
                                    claim_draw = True
                                elif score >= 125:
                                    depth -= 4
                                    claim_draw = True
                                elif score >= 100:
                                    depth -= 3
                                    claim_draw = True
                                elif score >= 75:
                                    depth -= 2
                                    claim_draw = True
                                elif score >= 50:
                                    depth -= 1
                                    claim_draw = True
                                else:
                                    claim_draw = True

                            elif score_type.startswith("mate"):
                                play_ponder_move = True
                                claim_draw = False

                            depth = max(10, depth)

                        except ValueError:
                            log.debug("some info lines do not contain a score: %s", info)

                response = engine.wait_for_command("bestmove").split(' ')
                best_move = response[1]

                board.push_uci(best_move)
                if board.is_game_over(claim_draw=claim_draw):
                    break
                moves.append(best_move)
                is_first_move = False

                if play_ponder_move:

                    response_move = response[3] if len(response) == 4 else None
                    if response_move is None:
                        continue

                    board.push_uci(response_move)
                    if board.is_game_over(claim_draw=claim_draw):
                        break
                    moves.append(response_move)

            if skip:
                continue

            result = "1/2"
            if board.is_checkmate():
                result = board.result()

            results.append(pos + " " + result + "\n")
            results.append(mirror_fen + " " + mirror(result) + "\n")

    except subprocess.TimeoutExpired as error:
        engine.stop()
        raise error

    except Exception as error:
        log.error(str(error))

    return results


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

    log.info("Reading test positions ...")

    concurrency = 6

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []
    for i in range(concurrency + 1):
        #engine = Engine("/home/norebo/extsource/stockfish/stockfish")
        engine = Engine("/home/norebo/bin/stockfish12")
        # engine = Engine("/home/norebo/chess/velvet_beta")
        engine.send_command("uci")
        engine.wait_for_command("uciok")

        engine.send_command("setoption name Hash value 2048")
        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engines.append(engine)

    log.info("Engines started")

    try:

        for i in range(1, 2):
            test_positions = read_fens("fen/set_" + str(i) + ".fen")
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
