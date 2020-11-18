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
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures._base import FIRST_COMPLETED
from time import time
import sys
from typing import List


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
        try:
            self.process.communicate("quit\n", timeout=10)
        finally:
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
        engine.send_command("resettestpositions")
        engine.wait_for_command("reset completed")

        fens = "prepare_quiet "
        is_first = True
        for pos in test_positions:
            if not is_first:
                fens += ";"
            else:
                is_first = False
            fens += pos + ":" + str(0)

        engine.send_command(fens)
        engine.wait_for_command("prepared")

        engine.send_command("printtestpositions")
        result = engine.wait_for_command("testpositions")[len("testpositions "):]
        for fen in result.split(";"):
            results.append(fen + "\n")

    except subprocess.TimeoutExpired as error:
        engine.stop()
        raise error

    except Exception as error:
        log.error(str(error))

    return results


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
    args = parser.parse_args()

    #log.getLogger().setLevel(log.DEBUG)
    log.info("Reading test positions ...")

    concurrency = 7

    # Start multiple engines
    log.info("Starting engines ...")
    engines = []
    for i in range(concurrency + 1):
        engine = Engine(args.engine)
        engine.send_command("uci")
        engine.wait_for_command("uciok")

        engine.send_command("setoption name Hash value 1024")
        engine.send_command("isready")
        engine.wait_for_command("readyok")

        engines.append(engine)

    log.info("Engines started")

    try:

        for i in range(1, 848):
            test_positions = read_fens("fen/set_" + str(i) + ".fen")
            log.info("Filtering test position set %d ...", i)
            start = time()

            results = run_pass(engines, concurrency, test_positions)

            with open("fen/set_filtered_" + str(i) + ".fen", "w", encoding="utf-8") as result_file:
                result_file.writelines(results)

            log.info("Duration         : %.2fs", time() - start)
            log.info("Skipped          : %d%%", 100 - (len(results) * 100 / len(test_positions)))

    finally:
        for engine in engines:
            engine.stop()


# Main
if __name__ == "__main__":
    main()
