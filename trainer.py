
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms, models
from src.network.training import train
from src.network.evaluation import evaluate, evaluate_with_solver
from src.preprocessing.dataset import ChessBoardDataset
from src.utils.dataset_utils import calculate_stats, build_transforms
from src.utils.plotting import plot_training_history
from models.models import init_model, ConvTransformer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="path to training dataset root (e.g. data/train/synthetic)")
    parser.add_argument("--val", required=True, help="path to validation dataset root (e.g. data/validation/synthetic)")
    parser.add_argument("--test", required=True, help="path to test dataset root (e.g. data/test/real)")
    parser.add_argument("--model-name", required=False, default="convnext_fine_tuned_final_stage", help="the chosen model name from the allowed model list")
    parser.add_argument("--model-path", required=False, help="path to saved model pth file (e.g. convnext_transformer.pth)")
    parser.add_argument("--save-path", required=True, help="path to save the trained model (e.g. trained_model.pth)")
    parser.add_argument("--lr", required=False, type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--epochs", required=False, type=int, default=15, help="number of epochs")
    parser.add_argument("--batch", required=False, type=int, default=2, help="batch size")
    parser.add_argument("--padding", required=False, type=float, default=1.0, help="crop padding value")
    parser.add_argument("--scheduler", action="store_true", help="use a learning rate scheduler")
    args = parser.parse_args()

    if "zero_shot" not in args.model_name and not args.model_path:
        raise ValueError("Missing input argument: path to saved model pth file") 

    NUM_WORKERS = 4
    SIZE = 112

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # Compute statistics
    train_mean, train_std, test_mean, test_std = calculate_stats(args.train, args.test, SIZE, SIZE, NUM_WORKERS)
    
    # Build final datasets and DataLoaders
    baseline_config = {} 
    train_transform = build_transforms(train_mean, train_std, config=baseline_config)
    test_transform = build_transforms(test_mean, test_std, config=baseline_config)
    
    train_dataset = ChessBoardDataset(root_dir=args.train, transform=train_transform, padding=args.padding)
    val_dataset = ChessBoardDataset(root_dir=args.val, transform=train_transform, padding=args.padding)
    test_dataset = ChessBoardDataset(root_dir=args.test, transform=test_transform, padding=args.padding)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model initialization
    model = init_model(args.model_name, args.model_path).to(device)

    # Training configuration
    lr = args.lr
    num_epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=lr)
    if args.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    else:
        scheduler = None
    
    # Training
    model, history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        save_path=args.save_path
    )

    # Testing without ILP solver
    square_acc, board_acc = evaluate(model, test_loader, device)
    print(f"Square accuracy: {square_acc:.2f}")
    print(f"Board accuracy: {board_acc:.2f}")

    # Testing with ILP solver
    square_acc, board_acc = evaluate_with_solver(model, test_loader, device)
    print(f"Square accuracy: {square_acc:.2f}")
    print(f"Board accuracy: {board_acc:.2f}")


if __name__ == "__main__":
    main()
    

