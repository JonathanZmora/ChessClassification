import csv
import random
import chess
import sys
from tqdm import tqdm

__MAX_RETRIES_PER_ITEM: int = 5


def random_legal_chess_board(max_plies: int = 100) -> chess.Board:
    """
    generates a random chess board, by running a game simulation

    Notes:
        assumes only legal boards

    Args:
        max_plies (int): maximal amount of moves to make in board, defaults to 100

    Returns:
        chess.Board: a random chess board
    """
    board: chess.Board = chess.Board()
    plies: int = random.randint(0, max_plies)
    for _ in range(plies):
        moves: list[chess.Move] = list(board.legal_moves)
        if not moves:  # checkmate/stalemate
            break
        board.push(random.choice(moves))
    return board


if __name__ == "__main__":
    """ script to generate a game file with random unique FENs for synthetic data generation  """
    argv: list[str] = sys.argv
    argc: int = len(argv)

    if argc != 4:
        print(f"Usage: python3 {argv[0]} <dst_file_name> <game_number> <num_of_fens>", file=sys.stderr)
        sys.exit(1)

    dst_file_name: str = argv[1]
    game_number: int = int(argv[2])
    num_of_fens: int = int(argv[3])

    seen: set[str] = set()

    with open(dst_file_name, "w", newline="") as dst_file:
        writer = csv.writer(dst_file)
        writer.writerow(["game", "from_frame", "to_frame", "fen"])

        for game_index in tqdm(range(num_of_fens), desc="Generating FENs", unit="fen"):
            for attempt in range(__MAX_RETRIES_PER_ITEM):
                random_board: chess.Board = random_legal_chess_board(max_plies=200)
                board_key: str = random_board.epd()

                if board_key not in seen:
                    seen.add(board_key)
                    fen: str = random_board.fen()
                    real_move_index: str = f"{game_index + 1:05}"
                    writer.writerow(
                        [game_number, real_move_index, real_move_index, fen]
                    )
                    break
