
IDX_TO_FEN = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
    12: '1'
}

IDX_TO_UNICODE = {
    0: '♙', 1: '♖', 2: '♘', 3: '♗', 4: '♕', 5: '♔', 
    6: '♟', 7: '♜', 8: '♞', 9: '♝', 10: '♛', 11: '♚',
    12: ''
}

FEN_TO_IDX = {
    'P':0, 'R':1, 'N':2, 'B':3, 'Q':4, 'K':5,
    'p':6, 'r':7, 'n':8, 'b':9, 'q':10, 'k':11
}

CLASS_MAP = {
    'white_pawn': 0,
    'white_rook': 1,    
    'white_knight': 2, 
    'white_bishop': 3,  
    'white_queen': 4,
    'white_king': 5,
    
    'black_pawn': 6,
    'black_rook': 7,    
    'black_knight': 8,  
    'black_bishop': 9,  
    'black_queen': 10,
    'black_king': 11,
    
    'empty': 12
}

EMPTY_LIMIT = 5000 
