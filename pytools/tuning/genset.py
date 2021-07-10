# A free and open source chess game using AssemblyScript and React
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

import chess.pgn
from random import randint

# Reads a collection of PGN files and generates a set of test positions for tuning


def split_pgns():
    print("Parsing chess games from pgn file ...")
    game_num = 0

    pgnFiles = [
        #"pgn/ccrl_blitz.pgn",
        # "pgn/ccrl4040.pgn",
        # "pgn/velvet.pgn",
        "pgn/velvet.pgn"
    ]

    setnum = 1
    setcount = 0

    for file in pgnFiles:

        new_pgn = open("pgn/set_" + str(setnum) + ".pgn", "w", encoding="utf-8")
        exporter = chess.pgn.FileExporter(new_pgn)
        pgn = open(file)
        print("Reading games from ", file)
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break

            result = game.headers['Result']
            if result == "*":
                continue

            # white = game.headers['White']
            # black = game.headers['Black']
            plyCount = int(game.headers['PlyCount'])
            #if plyCount > 150:
                #continue
            # isWhiteLoss = result == '0-1'
            # isBlackLoss = result == '1-0'

            # white_elo = game.headers.get('WhiteElo')
            # black_elo = game.headers.get('BlackElo')
            # if white_elo is None:
            #     continue
            # if black_elo is None:
            #     continue
            #
            # if int(white_elo) < 3000:
            #     continue
            #
            # if int(black_elo) < 3000:
            #     continue

            game_num += 1
            setcount += 1

            print("- read game #", game_num, " for set #", setnum, " - ", setcount)
            game.accept(exporter)

            if setcount >= 10000:
                setcount = 0

                new_pgn.close()
                setnum += 1
                new_pgn = open("pgn/set_" + str(setnum) + ".pgn", "w", encoding="utf-8")
                exporter = chess.pgn.FileExporter(new_pgn)

        new_pgn.close()
        pgn.close()


def extract_positions():

    white_count = 0
    black_count = 0
    set_count = 1
    pos_count = 0

    fen_file = open("fen/set_" + str(set_count) + ".fen", "w", encoding="utf-8")
    for i in range(1, 96 + 1):
        print("Processing pgn set", i, "...")
        pgn = open("pgn/set_" + str(i) + ".pgn")
        while True:
            game = chess.pgn.read_game(pgn)
            if not game:
                break
            # white_elo = game.headers.get('WhiteElo')
            # black_elo = game.headers.get('BlackElo')
            # if white_elo is None:
            #     continue
            # if black_elo is None:
            #     continue
            #
            # if int(white_elo) < 1600:
            #     continue
            #
            # if int(black_elo) < 1600:
            #     continue

            # ply_count = int(game.headers['PlyCount'])
            ply = 0

            candidate_fen = None
            candidate_white_color = True

            board = game.board()
            ply_count = sum(1 for _ in game.mainline_moves())
            if ply_count <= 40:
                continue

            next_start = randint(20, max(26, ply_count // 3))

            for move in game.mainline_moves():
                if not board.is_legal(move):
                    break

                is_capture = board.is_capture(move)
                is_promotion = move.promotion is not None

                board.push(move)
                gives_check = board.is_check()

                ply += 1

                if ply > 200:
                    break

                plies_before_mate = ply_count - ply
                if plies_before_mate <= 6:
                    break

                if ply < 20:
                    continue

                if ply < next_start:
                    continue

                if board.fullmove_number < 10:
                    continue

                is_white_turn = board.turn == chess.WHITE

                if gives_check or (candidate_fen is not None and is_capture) or (candidate_fen is None and is_promotion):
                    candidate_fen = None
                    continue

                if candidate_fen is not None:

                    pos_count += 1
                    fen_file.write(candidate_fen)
                    fen_file.write("\n")

                    if pos_count >= 5000:
                        pos_count = 0
                        fen_file.close()
                        set_count += 1
                        fen_file = open("fen/set_" + str(set_count) + ".fen", "w", encoding="utf-8")

                    candidate_fen = None

                    if candidate_white_color:
                        white_count += 1
                    else:
                        black_count += 1

                    next_start = ply + randint(13, 27)
                    if next_start > ply_count:
                        break

                elif is_white_turn and white_count <= black_count:
                    candidate_fen = board.fen()
                    candidate_white_color = True
                elif not is_white_turn and black_count <= white_count:
                    candidate_fen = board.fen()
                    candidate_white_color = False

        pgn.close()
        print("- total positions: ", (white_count + black_count))

    print(white_count)
    print(black_count)
    fen_file.close()


# Main
if __name__ == "__main__":
    extract_positions()
    #split_pgns()
