import argparse
import csv
import os
import cv2
import random
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Mapping for readable class names
# We use FEN characters as folder names for simplicity
# P=White Pawn, p=Black Pawn, etc.
CLASS_MAP = {
    'P': 'white_pawn',   'p': 'black_pawn',
    'N': 'white_knight', 'n': 'black_knight',
    'B': 'white_bishop', 'b': 'black_bishop',
    'R': 'white_rook',   'r': 'black_rook',
    'Q': 'white_queen',  'q': 'black_queen',
    'K': 'white_king',   'k': 'black_king',
    'empty': 'empty'
}

def parse_fen_to_grid(fen):
    """
    Parses a FEN string into an 8x8 list of lists.
    Rank 8 (top) is the first line in FEN, which matches the top of the image (y=0).
    """
    board_str = fen.split(' ')[0]  # Get piece placement part
    ranks = board_str.split('/')
    
    grid = []
    for rank in ranks:
        row = []
        for char in rank:
            if char.isdigit():
                # Empty squares
                num_empty = int(char)
                row.extend(['empty'] * num_empty)
            else:
                # Piece
                row.append(char)
        grid.append(row)
    return grid

def create_dataset(index_csv, output_dir, padding_pct=0.0):
    count_writes = 0

    index_csv = Path(index_csv)
    output_dir = Path(output_dir)
    images_dir = index_csv.parent / "blender_original/boards"
    
    # Create class directories
    classes = list(CLASS_MAP.values())
    for cls in classes:
        (output_dir / cls).mkdir(parents=True, exist_ok=True)
        
    print(f"Processing data from {index_csv}...")
    print(f"Outputting to {output_dir} with padding {padding_pct*100}%")
    
    with open(index_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    for row in tqdm(rows):
        fen = row['FEN']
        
        for col in ['warped_overhead_name', 'warped_east_name', 'warped_west_name']: 
            img_name = row[col]
            img_prefix = img_name.replace("warped.png", "")
            img_path = images_dir / img_name
            
            if not img_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            # Image dimensions
            H, W = img.shape[:2]
            
            # Calculate square size
            sq_h = H / 8.0
            sq_w = W / 8.0
            
            # Parse FEN into 8x8 grid
            grid = parse_fen_to_grid(fen)
            
            # Iterate over ranks (y) and files (x)
            # FEN Rank 0 is Top (Image Y=0)
            for r in range(8):
                for f in range(8):
                    piece_class = CLASS_MAP[grid[r][f]]

                    # Only save some percent of empty squares
                    if piece_class == 'empty':
                        if random.random() > 0.10: 
                            continue
                    
                    # Calculate coordinates
                    y_center = (r + 0.5) * sq_h
                    x_center = (f + 0.5) * sq_w
                    
                    # Base half-size
                    h_half = sq_h / 2.0
                    w_half = sq_w / 2.0
                    
                    # Apply padding (Context)
                    h_pad = h_half * (1 + padding_pct)
                    w_pad = w_half * (1 + padding_pct)
                    
                    y1 = int(max(0, y_center - h_pad))
                    y2 = int(min(H, y_center + h_pad))
                    x1 = int(max(0, x_center - w_pad))
                    x2 = int(min(W, x_center + w_pad))
                    
                    # Crop
                    crop = img[y1:y2, x1:x2]
                    
                    # Save
                    # Naming: game_frame_rank_file.png
                    filename = f"{img_prefix}{r}_{f}.png"
                    save_path = output_dir / piece_class / filename
                    
                    # Resize to standard size (optional, but good for saving disk space if original is 300x300)
                    # For baseline, let's save at 128x128
                    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite(str(save_path), crop_resized)
                    count_writes += 1

    print(f"Done. Dataset created with {count_writes} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to synthetic_index.csv")
    parser.add_argument("--out", required=True, help="Folder where images will be saved")
    parser.add_argument("--padding", type=float, default=0.0, help="Percentage of padding (0.2 = 20%)")
    args = parser.parse_args()
    
    create_dataset(args.index, args.out, args.padding)