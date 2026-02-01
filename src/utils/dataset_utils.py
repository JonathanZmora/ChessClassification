import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from src.preprocessing.dataset import ChessSquareDataset, ChessBoardDataset


def get_dataset_stats(dataset, batch_size=16, num_workers=4):
    """
    Computes mean/std for the given Dataset.
    Handles the 5D tensor structure [Batch, 64, 3, H, W]
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
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


def calculate_stats(train_root, test_root, num_workers):
    temp_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    # Load raw datasets
    train_dataset = ChessBoardDataset(root_dir=train_root, transform=temp_transform)
    test_dataset = ChessBoardDataset(root_dir=test_root, transform=temp_transform)
    
    print(f"Training samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    
    # Compute statistics
    print("Computing mean and std for training data...")
    train_mean, train_std = get_dataset_stats(train_dataset, batch_size=256, num_workers=num_workers)
    print(f"\nCalculated training mean: {train_mean}")
    print(f"Calculated training std:  {train_std}\n")
    
    print("Computing mean and std for test data...")
    test_mean, test_std = get_dataset_stats(test_dataset, batch_size=256, num_workers=num_workers)
    print(f"\nCalculated test mean: {test_mean}")
    print(f"Calculated test std:  {test_std}\n")

    return train_mean, train_std, test_mean, test_std


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


def get_square_by_class(dataset, target_class_name, class_names):
    """
    Searches the dataset for the first instance of target_class_name.
    Returns the image tensor reshaped for the model: [1, 3, 64, 64]
    """
    # Get the integer index for the class name
    try:
        target_idx = class_names.index(target_class_name)
    except ValueError:
        print(f"Error: '{target_class_name}' not found in class_names list.")
        return None

    print(f"Searching for class: '{target_class_name}' (Index: {target_idx})...")

    # Iterate through boards to find the piece
    for board_image, board_labels in dataset:        
        # Find indices where the label matches the target
        matches = (board_labels == target_idx).nonzero(as_tuple=True)[0]
        
        if len(matches) > 0:
            # Take the first match on this board
            idx = matches[0].item()
                        
            # Extract the single square image
            single_square = board_image[idx]
            
            # Add a batch dimension because the model expects [1, 3, 64, 64]
            input_tensor = single_square.unsqueeze(0)
            
            return input_tensor

    print(f"Could not find any samples of {target_class_name} in the dataset.")
    return None
