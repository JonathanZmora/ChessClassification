import os
import sys
import csv
from pathlib import Path
from io import TextIOWrapper
from typing import Optional

import chess.pgn

# dynamic type for file paths
pathType = str | Path


class FenGenerator:
    """ handles the extraction of chess FEN codes from given PGN chess game file """

    @staticmethod
    def __create_result_directory(dir_name: Path = Path("fen_results")) -> None:
        """ creates the result directory, if already exists - it ignores """
        os.makedirs(dir_name, exist_ok=True)

    @staticmethod
    def __read_next_chess_game_from_pgn(pgn_file: TextIOWrapper) -> Optional[chess.pgn.Game]:
        """ extracts and returns the next chess game from a PGN file """
        return chess.pgn.read_game(pgn_file)

    @staticmethod
    def __write_single_game_to_file(game: chess.pgn.Game, result_file: TextIOWrapper, game_number: int,
                                    src_pgn: str) -> None:
        """
        writes to a file, all FEN codes from a single game inside the targeted PGN

        Params:
            game (chess.pgn.Game): the current chess game
            result_file (TextIOWrapper): the result file
            game_number (int): index of the current game
            src_pgn (str): name of the source PGN file
        """
        board: chess.Board = game.board()
        writer = csv.writer(result_file)

        for move_index, move in enumerate(game.mainline_moves()):
            board.push(move)
            current_fen_full: str = board.fen()

            fen_move_code, *_ = current_fen_full.split(" ")
            real_move_index: str = f"{move_index + 1:05}"

            writer.writerow([game_number, real_move_index, real_move_index, fen_move_code])

    @staticmethod
    def fen_dataset_from_pgn(pgn_path: pathType, result_dir_path: pathType, result_file_name: pathType,
                             first_game: int = 0, games_limit: int = 100, base_game_index: int = 0) -> None:
        """
        extract FEN codes from PGN and writes them to a file

        Params:
            pgn_path (pathType): path to a PGN file
            result_dir_path (pathType): destination directory where generate files to
            first_game (int): how many games to skip
            game_limit (int): maximal number of games would be extracted (if negative it means 'all'), default: 100
            base_game_index (int): the base reference game, default: 0
        """
        if first_game < 0:
            raise ValueError("first game cannot be negative, provided:", first_game)

        pgn_path: Path = Path(pgn_path)
        result_dir_path = Path(result_dir_path)
        result_file_name = Path(result_file_name)

        FenGenerator.__create_result_directory(result_dir_path)

        try:
            with open(pgn_path, newline="\n") as pgn:
                current_game_index: int = 0
                total_games_read: int = 0
                pgn_name: str = pgn_path.stem

                print("[INFO] extracting FENs from all games in provided PGN...")

                current_result_file_name: Path = result_dir_path / result_file_name

                with open(current_result_file_name, "w", newline="") as result_fens:
                    csv.writer(result_fens).writerow(["game", "from_frame", "to_frame", "fen"])

                    while games_limit < 0 or current_game_index < games_limit:
                        next_game: chess.pgn.Game = FenGenerator.__read_next_chess_game_from_pgn(pgn)
                        if next_game is None:
                            break

                        if total_games_read < first_game:
                            total_games_read += 1
                            continue

                        print(f"[INFO] extracting game: {current_game_index + 1}")
                        FenGenerator.__write_single_game_to_file(next_game, result_fens,
                                                                 base_game_index + current_game_index, pgn_name)

                        current_game_index += 1

                print("[INFO] success!")
        except (FileNotFoundError, PermissionError) as e:
            print(f"[FAIL] Cannot open the given PGN file, provided: {pgn_path}", file=sys.stderr)


if __name__ == "__main__":
    try:
        # Example usage:
        FenGenerator.fen_dataset_from_pgn(
            r"C:\Users\user\Downloads\Abdusattorov\Abdusattorov.pgn",
            "",
            "Abdusattorov_games.csv",
            0,
            10,
            8
        )
    except TypeError as e:
        print(str(e), file=sys.stderr)
