import cv2
import torch
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.utils.fen_utils import fen_to_labels
from src.constants import CLASS_MAP, EMPTY_LIMIT


class ChessSquareDataset(Dataset):
    def __init__(self, board_dataset):
        self.squares = []
        self.labels = []
        
        print("Pre-loading dataset...")
        for imgs, lbls in tqdm(board_dataset):
            self.squares.extend([img.detach().clone() for img in imgs])
            self.labels.extend(lbls.tolist())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.squares[idx], self.labels[idx]
    

class ChessBoardDataset(Dataset):
    def __init__(self, root_dir, transform=None, padding=0.0, return_fen=False):
        """
        Args:
            root_dir (str): Path containing an 'images' folder and 'gt.csv'
            transform (callable): Transform for individual squares
            padding (float): Context padding (e.g. 0.2)
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.csv_path = self.root_dir / "gt.csv"
        self.transform = transform
        self.padding = padding
        self.return_fen = return_fen
        
        # Load CSV
        self.df = pd.read_csv(self.csv_path)
        
        # Ensure column names are stripped and lowercase
        self.df.columns = [c.lower().strip() for c in self.df.columns]

    
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_name = row['image_name']
            fen = row['fen']
            
            img_path = self.images_dir / img_name
            
            # Load full board image
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            squares = []
            labels = fen_to_labels(fen) # Get the 64 square labels
            
            H, W = img.shape[:2]
            sq_h, sq_w = H / 8.0, W / 8.0
            
            # Padding config
            h_pad = (sq_h / 2.0) * (1 + self.padding)
            w_pad = (sq_w / 2.0) * (1 + self.padding)

            # Slice squares
            for r in range(8):
                for f in range(8):
                    y_c = (r + 0.5) * sq_h
                    x_c = (f + 0.5) * sq_w
                    
                    y1 = int(max(0, y_c - h_pad))
                    y2 = int(min(H, y_c + h_pad))
                    x1 = int(max(0, x_c - w_pad))
                    x2 = int(min(W, x_c + w_pad))
                    
                    crop = img[y1:y2, x1:x2]
                    
                    # Convert to PIL for Transforms
                    crop_pil = transforms.ToPILImage()(crop)
                    
                    if self.transform:
                        crop_tensor = self.transform(crop_pil)
                    else:
                        crop_tensor = transforms.ToTensor()(crop_pil)
                        
                    squares.append(crop_tensor)
            
            # Stack into a [64, 3, 64, 64] tensor
            board_tensor = torch.stack(squares)

            if self.return_fen:
                return board_tensor, labels, fen

            return board_tensor, labels
    