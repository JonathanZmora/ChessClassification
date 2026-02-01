import torch
import torch.nn as nn
import random
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from src.network.evaluation import extract_cnn_feature_maps, pca_on_channels
from src.utils.fen_utils import tensor_to_fen, fen_to_grid
from src.constants import IDX_TO_UNICODE
from tqdm import tqdm


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


def plot_class_distribution(dataset, class_names):
    all_labels = []
    
    print("Iterating through dataset to count classes (this might take a minute)...")
    
    for _, board_labels in tqdm(dataset):        
        all_labels.extend(board_labels.numpy().flatten())

    counts = Counter(all_labels)

    indices = range(len(class_names))
    
    # Get counts for each class index
    values = [counts.get(i, 0) for i in indices]
    names = class_names
    
    # Calculate percentages
    total = sum(values)
    percentages = [v / total * 100 if total > 0 else 0 for v in values]

    # Plot
    plt.figure(figsize=(14, 6))
    bars = sns.barplot(x=names, y=values, palette="viridis", hue=names, legend=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Class Distribution (Total Squares: {total})")
    plt.ylabel("Number of Squares")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of bars
    for i, p in enumerate(bars.patches):
        height = p.get_height()
        if height > 0: # Only label if bar exists
            bars.text(p.get_x() + p.get_width()/2., height + (max(values)*0.01),
                    f'{percentages[i]:.1f}%', ha="center", fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_feature_maps(model, images, classes_to_plot, labels_for_plot, samples_per_class):
    print("Extracting features and running PCA...\n")
    raw_features = extract_cnn_feature_maps(model, images)
    pca_features = pca_on_channels(raw_features)
    
    # Resize to match original image size (64x64)
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