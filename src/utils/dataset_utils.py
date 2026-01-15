import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataset_stats(dataset, batch_size=16, num_workers=4):
    """
    Computes mean/std for the given Dataset.
    Handles the 5D tensor structure [Batch, 64, 3, H, W]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    print("Computing mean and std for training data...")
    for boards, _ in tqdm(loader, desc="Calculating Stats"):
        # Flatten Batch and Squares dimensions to [Batch * 64, 3, 64 * 64]
        B, S, C, H, W = boards.shape
        flat_boards = boards.view(B * S, C, H * W)
        
        # Compute stats per channel
        mean += flat_boards.mean(2).sum(0)
        std += flat_boards.std(2).sum(0)
        total_samples += (B * S)

    mean /= total_samples
    std /= total_samples
    
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
    transform_list.append(transforms.Resize((64, 64)))
    
    # TRAIN ONLY TRANSFORMS
    if mode == 'train':
        # Geometric (Scale/Flip)
        if config.get('geometry', False):
            transform_list.append(
                transforms.RandomResizedCrop(64, scale=(0.85, 1.0), ratio=(0.95, 1.05))
            )
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

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

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)
