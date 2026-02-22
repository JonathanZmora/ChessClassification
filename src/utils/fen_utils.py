import torch
import numpy as np
from src.constants import IDX_TO_FEN, FEN_TO_IDX


def tensor_to_fen(board_tensor, view='white'):
    """
    Converts an 8x8 grid of predicted class indices into a FEN string.

    Args:
        board_tensor (torch.Tensor or np.ndarray): 
            An 8x8 array containing the piece class indices for each square.
        view (str, optional): 
            The camera perspective ('white' or 'black'). Defaults to 'white'.

    Returns:
        str: The generated FEN string representing the board state.
    """
    grid = board_tensor.numpy() if not isinstance(board_tensor, np.ndarray) else board_tensor
    if view == 'black':
        grid = np.flip(grid, axis=(0, 1))
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


def fen_to_grid(fen, view='white'):
    """
    Parses a FEN string into an 8x8 grid of integer class indices.

    Args:
        fen (str): The FEN string representing the board state.
        view (str, optional): The camera perspective ('white' or 'black'). Defaults to 'white'.

    Returns:
        np.ndarray: An 8x8 integer array containing the class indices for each square.
    """
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

    if view == 'black':
        grid = np.flip(grid, axis=(0, 1))
        
    return grid


def fen_to_labels(fen, view='white'):
    """
    Converts a FEN string into a 1D tensor of class labels.

    Args:
        fen (str): The FEN string representing the board state.
        view (str, optional): The camera perspective ('white' or 'black'). Defaults to 'white'.

    Returns:
        torch.Tensor: A 1D tensor of length 64 containing the integer class indices for each square.
    """
    labels = []
    rows = fen.split(' ')[0].split('/')
    for row in rows:
        for char in row:
            if char.isdigit():
                labels.extend([12] * int(char)) # 12 = Empty
            else:
                labels.append(FEN_TO_IDX[char])
                
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    if view == 'black':
        labels_tensor = torch.flip(labels_tensor, dims=[0])
        
    return labels_tensor
