import torch
import torch.nn as nn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.network.inference import solve_chess_board_ilp
from src.network.evaluation import extract_cnn_feature_maps, pca_on_channels
from src.utils.fen_utils import tensor_to_fen, fen_to_grid
from src.constants import IDX_TO_UNICODE


def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss curves.

    Args:
        history (dict): A dictionary containing lists of epoch metrics. 
            Expected keys are 'train_acc', 'val_acc', 'train_loss', and 'val_loss'.
    """
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.show()


def plot_tensor_grid(ax, grid_data, title):
    """
    Renders a digital chessboard visualization on a matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib subplot axis to draw on.
        grid_data (torch.Tensor or np.ndarray): An 8x8 array of integer class indices.
        title (str): The title text to display above the plot.
    """
    board = np.zeros((8, 8))
    board[0::2, 0::2] = 1
    board[1::2, 1::2] = 1
    
    ax.imshow(board, cmap='binary', alpha=0.1)
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    
    if isinstance(grid_data, torch.Tensor):
        grid_data = grid_data.cpu().numpy()
        
    for r in range(8):
        for c in range(8):
            idx = grid_data[r, c]
            symbol = IDX_TO_UNICODE.get(idx, '?')
            color = 'blue' if 0 <= idx <= 5 else 'black'
            if symbol:
                ax.text(c, r, symbol, fontsize=20, ha='center', va='center', 
                        color=color, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight='bold', pad=10)


def visualize_test_samples(model, dataset, num_samples=3, device='cpu'):
    """
    Randomly samples and visualizes model predictions against ground truth.
    Selects random board configurations from the dataset, runs forward passes 
    through the model, and displays a three-panel matplotlib figure for each 
    sample containing the original full-board image, the ground truth digital 
    layout, and the model's predicted digital layout.

    Args:
        model (nn.Module): The trained PyTorch model to evaluate.
        dataset (Dataset): The dataset to sample from. Must yield tuples of 
            (board_tensor, target_tensor, true_fen) and contain 'df' and 'images_dir' attributes.
        num_samples (int, optional): The number of random samples. Defaults to 3.
        device (str, optional): Target computation device. Defaults to 'cpu'.
    """
    model.eval()
    # Randomly select num_samples indices of dataset samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        # Unpack board, labels, and FEN
        board_tensor, target_tensor, true_fen = dataset[idx]
        
        # Get image path and view
        row = dataset.df.iloc[idx]
        img_name = row['image_name']
        img_path = dataset.images_dir / img_name
        view = row['view'].lower()
        
        # Load image
        if not img_path.exists(): continue
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Inference
        input_batch = board_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            preds = torch.argmax(outputs, dim=1)
            pred_grid = preds.view(8, 8)
            
        # Convert prediction to FEN string for text comparison
        pred_fen = tensor_to_fen(pred_grid.cpu(), view)
        
        # Convert true FEN to grid for visual comparison
        true_grid = fen_to_grid(true_fen, view)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Input: {img_name}", fontweight='bold')
        axes[0].axis('off')
        
        plot_tensor_grid(axes[1], true_grid, "Ground Truth")
        plot_tensor_grid(axes[2], pred_grid, "Model Prediction")
        
        plt.tight_layout()
        plt.show()
        
        # Print text comparison
        print(f"True FEN: {true_fen}")
        print(f"Pred FEN: {pred_fen}")
        if true_fen == pred_fen:
            print("PERFECT MATCH")
        else:
            print("MISMATCH\n")


def visualize_test_samples_with_solver(model, dataset, num_samples=3, device='cpu'):
    """
    Visualizes and compares base model predictions against the refinements
    of the linear programming solver from src.network.inference.solve_chess_board_ilp.
    Randomly samples chess boards from the dataset and generates a four-panel 
    comparison for each: The original full-board image, the ground truth layout, 
    the raw model predictions, and the corrected predictions from the ILP solver. 

    Args:
        model (nn.Module): The trained PyTorch model to evaluate.
        dataset (Dataset): The dataset to sample from. Must yield tuples of 
            (board_tensor, target_tensor, true_fen) and contain 'df' and 'images_dir' attributes.
        num_samples (int, optional): The number of random samples. Defaults to 3.
        device (str, optional): Target computation device. Defaults to 'cpu'.
    """
    model.eval()
    # Randomly select num_samples indices of dataset samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        # Unpack board, labels, and FEN
        board_tensor, target_tensor, true_fen = dataset[idx]
        
        # Get image path and view
        row = dataset.df.iloc[idx]
        img_name = row['image_name']
        img_path = dataset.images_dir / img_name
        view = row.get('view', 'w').lower()
        
        # Load image
        if not img_path.exists(): continue
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Inference
        input_batch = board_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            
            # Get base model predictions
            base_preds = torch.argmax(outputs, dim=1)
            base_grid = base_preds.view(8, 8).cpu().numpy()
            
            # Get solver predictions
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            solver_preds = solve_chess_board_ilp(probs)
            solver_grid = solver_preds.reshape(8, 8)
            
        # Convert predictions to FEN strings for text comparison
        base_fen = tensor_to_fen(base_grid, view)
        solver_fen = tensor_to_fen(solver_grid, view)
        
        # Convert true FEN to grid for visual comparison
        true_grid = fen_to_grid(true_fen, view)
        
        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Input: {img_name} (View: {view.upper()})", fontweight='bold')
        axes[0].axis('off')
        
        plot_tensor_grid(axes[1], true_grid, "Ground Truth")
        plot_tensor_grid(axes[2], base_grid, "Base Model Prediction")
        plot_tensor_grid(axes[3], solver_grid, "ILP Solver Prediction")
        
        plt.tight_layout()
        plt.show()
        
        # Print text comparison
        print(f"True FEN:   {true_fen}")
        print(f"CNN FEN:    {base_fen}")
        print(f"Solver FEN: {solver_fen}")
        
        if true_fen == solver_fen:
            print("SOLVER PERFECT MATCH\n")
        elif true_fen == base_fen:
            print("CNN PERFECT MATCH (Solver messed it up)\n")
        else:
            print("MISMATCH ON BOTH\n")


def plot_feature_maps(model, images, classes_to_plot, labels_for_plot, samples_per_class):
    """
    Extracts, reduces, and visualizes CNN feature maps for specified classes.

    Args:
        model (nn.Module): The PyTorch model used for feature extraction.
        images (Iterable[torch.Tensor]): A collection of image tensors to process.
        classes_to_plot (list[str]): List of class names corresponding to the 
            rows in the visualization grid.
        labels_for_plot (Iterable): The target labels corresponding to the provided images.
        samples_per_class (int): The number of image samples to display per class.
    """
    print("Extracting features and running PCA...\n")
    raw_features = extract_cnn_feature_maps(model, images)
    pca_features = pca_on_channels(raw_features)
    
    # Resize to 64x64
    pca_features = nn.functional.interpolate(pca_features, size=(64, 64))
    
    # Plotting
    fig, axs = plt.subplots(len(classes_to_plot), samples_per_class, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.01, top=0.8)
    
    for i, (pca_map, label) in enumerate(zip(pca_features, labels_for_plot)):
        row = i // samples_per_class
        col = i % samples_per_class
        
        img_np = pca_map.permute(1, 2, 0).cpu().numpy() 
        axs[row, col].imshow(img_np)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])
        
        # Add row labels
        if col == 0:
            axs[row, col].set_ylabel(classes_to_plot[row], rotation='horizontal', ha='right', size='medium')
    
    plt.tight_layout()
    plt.show()
