import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

# Defined limit for the 'empty' class 
EMPTY_LIMIT = 5000 


class ChessSquareDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder.
                            Expects subfolders like 'P', 'n', 'empty'.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Load all image paths and labels
        self._load_dataset()

    def _load_dataset(self):
        for class_name, label_idx in CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            
            # Get all image files
            files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            
            if class_name == 'empty' and EMPTY_LIMIT:
                # Randomly sample distinct files
                print(f"Undersampling 'empty' from {len(files)} to {EMPTY_LIMIT}...")
                files = random.sample(files, EMPTY_LIMIT)

            for img_path in files:
                self.samples.append((str(img_path), label_idx))

        print(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load Image (Convert to RGB to handle color randomization)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    

class ChessBoardDataset(Dataset):
    def __init__(self, root_dir, transform=None, padding=0.0):
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
        view = row['view'] 
        
        img_path = self.images_dir / img_name
        
        # Load Image
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Slice squares
        squares = []
        H, W = img.shape[:2]
        sq_h, sq_w = H / 8.0, W / 8.0
        
        h_pad = (sq_h / 2.0) * (1 + self.padding)
        w_pad = (sq_w / 2.0) * (1 + self.padding)

        for r in range(8):
            for f in range(8):
                y_c = (r + 0.5) * sq_h
                x_c = (f + 0.5) * sq_w
                
                y1 = int(max(0, y_c - h_pad))
                y2 = int(min(H, y_c + h_pad))
                x1 = int(max(0, x_c - w_pad))
                x2 = int(min(W, x_c + w_pad))
                
                crop = img[y1:y2, x1:x2]
                
                # Transform
                crop_pil = transforms.ToPILImage()(crop)
                if self.transform:
                    crop_tensor = self.transform(crop_pil)
                else:
                    crop_tensor = transforms.ToTensor()(crop_pil)
                    
                squares.append(crop_tensor)
        
        # Stack into a batch of 64 squares
        board_tensor = torch.stack(squares)
        
        return board_tensor, fen, view
    

class TransformWrapper(Dataset):
    """
    Wraps a subset to apply a transform on the fly.
    Necessary because random_split returns subsets of the original dataset.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        # Access the underlying dataset to get the raw PIL image
        original_ds = self.subset.dataset
        real_idx = self.subset.indices[idx]
        img_path, label = original_ds.samples[real_idx]
        
        # Load PIL image
        img = Image.open(img_path).convert('RGB')
        
        return self.transform(img), label
        
    def __len__(self):
        return len(self.subset)
    

def get_train_stats(dataset, batch_size=64, num_workers=4):
    """
    Computes mean and std of the dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    print("Computing mean and std for training data...")
    for images, _ in tqdm(loader):
        # images shape: [Batch, 3, H, W]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    
    return mean.tolist(), std.tolist()


def build_transforms(mean, std, mode='train', config=None):
    """
    Constructs a transform pipeline based on a config dict.
    
    config example:
    {
        'jitter': True,
        'blur': False,
        'noise': True,
        'geometry': True 
    }
    """
    if config is None: config = {}
    
    transform_list = []
    
    # TRAIN ONLY AUGMENTATIONS
    if mode == 'train':
        # Geometric (Scale/Flip)
        if config.get('geometry', False):
            transform_list.append(
                transforms.RandomResizedCrop(64, scale=(0.85, 1.0), ratio=(0.95, 1.05))
            )
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        else:
            # Always need at least Resize if not random cropping
            transform_list.append(transforms.Resize((64, 64)))

        # Photometric (Color/Lighting)
        if config.get('jitter', False):
            transform_list.append(
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
            )

        # Blur
        if config.get('blur', False):
            transform_list.append(
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2)
            )

        # Noise 
        if config.get('noise', False):
            transform_list.append(
                transforms.RandomApply([lambda x: x + torch.randn_like(x) * 0.05], p=0.2)
            )

    # COMMON TRANSFORMATIONS
    if mode == 'val':
        transform_list.append(transforms.Resize((64, 64)))
        
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)