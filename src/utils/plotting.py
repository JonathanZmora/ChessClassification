import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.fen_utils import tensor_to_fen, fen_to_grid
from src.constants import IDX_TO_UNICODE


def plot_training_history(history):
    """
    Helper function to plot training/validation curves.
    """
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
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
    Draws a digital chessboard representation.
    grid_data: 8x8 numpy array or tensor with class indices.
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
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx in indices:
        # Get Data
        board_tensor, target_tensor, true_fen = dataset[idx]
        
        # Get Image Path
        row = dataset.df.iloc[idx]
        img_name = row['image_name']
        img_path = dataset.images_dir / img_name
        
        # Load Image
        if not img_path.exists(): continue
        orig_img = cv2.imread(str(img_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Inference
        input_batch = board_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_batch)
            preds = torch.argmax(outputs, dim=1)
            pred_grid = preds.view(8, 8)
            
        # Convert Prediction to FEN string for text comparison
        pred_fen = tensor_to_fen(pred_grid.cpu())
        
        # Convert True FEN to Grid for visual comparison
        true_grid = fen_to_grid(true_fen)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original Image
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Input: {img_name}", fontweight='bold')
        axes[0].axis('off')
        
        # Ground Truth
        plot_tensor_grid(axes[1], true_grid, "Ground Truth")
        
        # Prediction
        plot_tensor_grid(axes[2], pred_grid, "Model Prediction")
        
        plt.tight_layout()
        plt.show()
        
        # Print Text Comparison
        print(f"True FEN: {true_fen}")
        print(f"Pred FEN: {pred_fen}")
        if true_fen == pred_fen:
            print("PERFECT MATCH")
        else:
            print("MISMATCH\n")
            