import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from src.network.inference import predict, predict_with_solver


def evaluate(model, dataloader, device='cpu'):
    """
    Evaluates the model on the provided dataset and calculates accuracy metrics.

    Args:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (DataLoader): DataLoader providing batched sequence data.
        device (str, optional): Target computation device. Defaults to 'cpu'.

    Returns:
        tuple[float, float]: A tuple containing:
            - square_acc: Accuracy percentage across all individual squares.
            - board_acc: Accuracy percentage of perfectly predicted full boards.
    """
    model.eval()
    
    total_squares = 0
    correct_squares = 0
    total_boards = 0
    correct_boards = 0
    
    print(f"Starting Evaluation on {len(dataloader.dataset)} boards...")
    
    with torch.no_grad():
        for boards, targets in tqdm(dataloader, desc="Evaluating"):
            B, S, C, H, W = boards.shape
            inputs = boards.view(B * S, C, H, W).to(device)
            targets = targets.to(device).view(-1)
            
            # Inference
            outputs = model(inputs) 
            preds = torch.argmax(outputs, dim=1)
            
            # Square accuracy
            correct_squares += (preds == targets).sum().item()
            total_squares += targets.size(0)
            
            # Board accuracy
            pred_boards = preds.view(B, 64)
            target_boards = targets.view(B, 64)
            matches = (pred_boards == target_boards).all(dim=1)
            correct_boards += matches.sum().item()
            total_boards += B
            
    square_acc = 100.0 * correct_squares / total_squares
    board_acc = 100.0 * correct_boards / total_boards
    
    return square_acc, board_acc


def evaluate_with_solver(model, dataloader, device='cpu'):
    """
    Evaluates the model using the linear programming solver from 
    src.inference.solve.solve_chess_board_ilp and calculates accuracy metrics.

    Args:
        model (nn.Module): The PyTorch sequence model to evaluate.
        dataloader (DataLoader): DataLoader providing batched sequence data.
        device (str, optional): Target computation device. Defaults to 'cpu'.

    Returns:
        tuple[float, float]: A tuple containing:
            - square_acc: Accuracy percentage across all individual squares.
            - board_acc: Accuracy percentage of perfectly predicted full boards.
    """        
    print(f"Starting solver evaluation on {len(dataloader.dataset)} boards...")

    # Inference
    preds, targets = predict_with_solver(model, dataloader, device)
    
    # Square accuracy
    correct_squares = (preds == targets).sum()
    total_squares = len(targets)
    square_acc = 100.0 * correct_squares / total_squares
    
    # Board accuracy
    num_boards = len(preds) // 64
    pred_boards = preds.reshape(num_boards, 64)
    target_boards = targets.reshape(num_boards, 64)
    matches = (pred_boards == target_boards).all(axis=1)
    correct_boards = matches.sum()
    board_acc = 100.0 * correct_boards / num_boards
    
    return square_acc, board_acc


def analyze_model_performance(model, dataloader, class_names, device='cpu'):
    """
    Analyzes and visualizes model performance across all target classes.
    Computes and prints per-class accuracy and displays a heatmap of the 
    confusion matrix using the provided model and dataloader.

    Args:
        model (nn.Module): The PyTorch sequence model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        class_names (list[str]): List of target class names for labeling.
        device (str, optional): Target computation device. Defaults to 'cpu'.
    """
    # Inference
    all_preds, all_labels = predict(model, dataloader, device)

    # Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row to get accuracy per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle NaN if a class is missing from test set
    cm_normalized = np.nan_to_num(cm_normalized)

    # Print accuracy per class
    print("Accuracy per Class:\n")
    for i, class_name in enumerate(class_names):
        acc = cm_normalized[i, i] * 100
        count = cm[i, :].sum() # Total instances of this class
        print(f"{class_name:<10}: {acc:.2f}% ({int(cm[i, i])}/{count} correct)")

    print("\n" + "="*30 + "\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def analyze_model_performance_solver(model, dataloader, class_names, device='cpu'):
    """
    Analyzes and visualizes model performance using the linear programming solver 
    from src.inference.solve.solve_chess_board_ilp across all target classes.
    Computes and prints per-class accuracy and displays a heatmap of the 
    confusion matrix using the provided model and dataloader.

    Args:
        model (nn.Module): The PyTorch sequence model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        class_names (list[str]): List of target class names for labeling.
        device (str, optional): Target computation device. Defaults to 'cpu'.
    """
    # Inference
    all_preds, all_labels = predict_with_solver(model, dataloader, device)

    # Build confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row to get accuracy per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle NaN if a class is missing from test set
    cm_normalized = np.nan_to_num(cm_normalized)

    # Print accuracy per class
    print("Accuracy per Class:\n")
    for i, class_name in enumerate(class_names):
        acc = cm_normalized[i, i] * 100
        count = cm[i, :].sum() # Total instances of this class
        print(f"{class_name:<10}: {acc:.2f}% ({int(cm[i, i])}/{count} correct)")

    print("\n" + "="*30 + "\n")

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def extract_cnn_feature_maps(model, images, device='cpu'):
    """
    Extracts intermediate ResNet18 model feature maps.
    Truncates the provided model by removing its final 5 child modules 
    and then runs forward passes on the input images to collect the raw 
    feature activations given by the second residual block of the ResNet18 model.

    Args:
        model (nn.Module): The torchvision ResNet18 model to extract features from.
        images (Iterable[torch.Tensor]): A collection of image tensors.
        device (str, optional): Target computation device. Defaults to 'cpu'.

    Returns:
        list[torch.Tensor]: A list of extracted feature map tensors.
    """
    model.eval()
    feature_maps = []
    
    backbone = nn.Sequential(*list(model.children())[:-5]).to(device)
    
    with torch.no_grad():
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            
            img = img.to(device)
            features = backbone(img)
            feature_maps.append(features.squeeze(0)) # Remove batch dim

    return feature_maps


def pca_on_channels(feature_maps, n_components=3):
    """    
    Flattens spatial dimensions to apply PCA across the channel axis, 
    reducing high-dimensional activations into a specified number of 
    components. The results are Min-Max normalized to the range [0, 1].

    Args:
        feature_maps (Iterable[torch.Tensor]): A collection of feature map tensors of shape (C, H, W).
        n_components (int, optional): Number of principal components to retain. Defaults to 3 (RGB).

    Returns:
        torch.Tensor: A stacked tensor of shape (N, n_components, H, W) containing the PCA results.
    """
    pca_feature_maps = []

    for feature_map in feature_maps:
        C, H, W = feature_map.shape
        
        # Reshape to (H * W, C) for PCA
        reshaped_map = feature_map.permute(1, 2, 0).reshape(H * W, C).detach().cpu().numpy()

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(reshaped_map)

        # Reshape back to (H, W, 3) then transpose to (3, H, W) for future plotting
        pca_result_reshaped = pca_result.reshape(H, W, n_components).transpose(2, 0, 1)
        
        pca_tensor = torch.from_numpy(pca_result_reshaped)
        pca_tensor = (pca_tensor - pca_tensor.min()) / (pca_tensor.max() - pca_tensor.min())
        
        pca_feature_maps.append(pca_tensor)

    return torch.stack(pca_feature_maps)
