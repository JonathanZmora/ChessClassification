import torch
import numpy as np
from src.constants import IDX_TO_FEN, FEN_TO_IDX


def tensor_to_fen(board_tensor):
    """
    Converts the (8, 8) output tensor back to a FEN string
    so you can compare it against your gt.csv.
    """
    grid = board_tensor.numpy() if not isinstance(board_tensor, np.ndarray) else board_tensor
    fen_rows = []
    
    for r in range(8):
        current_row = ""
        empty_count = 0
        for f in range(8):
            val = grid[r, f]
            char = IDX_TO_FEN[val]
            
            if char == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    current_row += str(empty_count)
                    empty_count = 0
                current_row += char
        
        if empty_count > 0:
            current_row += str(empty_count)
        fen_rows.append(current_row)
        
    return "/".join(fen_rows)


def fen_to_grid(fen):
    """Parses a FEN string into an 8x8 numpy array of class indices."""
    grid = np.full((8, 8), 12, dtype=int)
    rows = fen.split(' ')[0].split('/')
    
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit():
                c += int(char)
            else:
                grid[r, c] = FEN_TO_IDX[char]
                c += 1
    return grid


def fen_to_labels(fen):
    """Converts FEN string to (64,) tensor of labels (0-12)"""
    labels = []
    rows = fen.split(' ')[0].split('/')
    for row in rows:
        for char in row:
            if char.isdigit():
                labels.extend([12] * int(char)) # 12 = Empty
            else:
                labels.append(FEN_TO_IDX[char])
    return torch.tensor(labels, dtype=torch.long)
