import torch
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from src.preprocessing.dataset import ChessBoardDataset


def get_dataset_stats(dataset, batch_size=16, num_workers=4):
    """
    Computes mean and standard deviation for the given Dataset.
    Handles the 5D tensor structure [Batch, Squares, 3, H, W].

    Args:
        dataset (Dataset): The PyTorch dataset to compute the stats on.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 16.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 4.

    Returns:
        tuple[list[float], list[float]]: A tuple containing:
            - mean: A list of the calculated mean values for each channel.
            - std: A list of the calculated standard deviation values for each channel.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for boards, _ in tqdm(loader, desc="Calculating Stats"):
        # Flatten dimensions to [Batch * Squares, 3, H * W]
        B, S, C, H, W = boards.shape
        flat_boards = boards.view(B * S, C, H * W)
        
        # Compute stats per channel
        mean += flat_boards.mean(2).sum(0)
        std += flat_boards.std(2).sum(0)
        total_samples += (B * S)

    mean /= total_samples
    std /= total_samples
    
    return mean.tolist(), std.tolist()


def calculate_stats(train_root, test_root, train_size, test_size, num_workers):
    """
    Calculates mean and standard deviation for the given train and test datasets
    using get_dataset_stats.

    Args:
        train_root (str): Directory path to the training dataset.
        test_root (str): Directory path to the testing dataset.
        train_size (int): Target resolution to resize the training square crops.
        test_size (int): Target resolution to resize the testing square crops.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple[list[float], list[float], list[float], list[float]]: A tuple containing:
            - train_mean: Calculated RGB mean for the training set.
            - train_std: Calculated RGB standard deviation for the training set.
            - test_mean: Calculated RGB mean for the test set.
            - test_std: Calculated RGB standard deviation for the test set.
    """
    train_transform = transforms.Compose([transforms.Resize((train_size, train_size)), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((test_size, test_size)), transforms.ToTensor()])

    # Load raw datasets
    train_dataset = ChessBoardDataset(root_dir=train_root, transform=train_transform)
    test_dataset = ChessBoardDataset(root_dir=test_root, transform=test_transform)
    
    print(f"Training samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    
    # Compute statistics
    print("Computing mean and std for training data...")
    train_mean, train_std = get_dataset_stats(train_dataset, batch_size=16, num_workers=num_workers)
    print(f"\nCalculated training mean: {train_mean}")
    print(f"Calculated training std:  {train_std}\n")
    
    print("Computing mean and std for test data...")
    test_mean, test_std = get_dataset_stats(test_dataset, batch_size=16, num_workers=num_workers)
    print(f"\nCalculated test mean: {test_mean}")
    print(f"Calculated test std:  {test_std}\n")

    return train_mean, train_std, test_mean, test_std


def build_transforms(mean, std, config=None):
    """
    Constructs a composed torchvision transform pipeline based on a configuration dictionary.

    Args:
        mean (list[float]): RGB mean values for dataset normalization.
        std (list[float]): RGB standard deviation values for dataset normalization.
        config (dict, optional): Dictionary specifying augmentation parameters. 
            Supported keys include 'resize' (int, default 112), 'flip' (float probability), 
            'jitter' (tuple of brightness, hue), 'blur' (tuple of kernel_size, sigma), 
            and 'noise' (float noise factor). Defaults to None.

    Returns:
        torchvision.transforms.Compose: The composed transformation pipeline ready 
                                        to be applied to the dataset.
    """
    if config is None: 
        config = {}

    size = config.get('resize', 112)
    
    transform_list = []

    if size:
        transform_list.append(transforms.Resize((size, size)))
    transform_list.append(transforms.ToTensor())
    
    if config.get('flip', False):
        flip_prob = config['flip']
        transform_list.append(transforms.RandomVerticalFlip(p=flip_prob))

    if config.get('jitter', False):
        brightness, hue = config['jitter']
        transform_list.append(transforms.ColorJitter(brightness=brightness, hue=hue))

    if config.get('blur', False):
        kernel_size, sigma = config['blur']
        transform_list.append(transforms.GaussianBlur(kernel_size, sigma))
 
    if config.get('noise', False):
        noise = config['noise']
        transform_list.append(transforms.Lambda(lambda x: torch.clamp(x + noise * torch.randn_like(x), 0, 1)))

    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)
