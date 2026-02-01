import torch
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from src.network.inference import predict


def evaluate(model, dataloader, device='cpu'):
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
            
            # Square Accuracy
            correct_squares += (preds == targets).sum().item()
            total_squares += targets.size(0)
            
            # Board Accuracy
            pred_boards = preds.view(B, 64)
            target_boards = targets.view(B, 64)
            matches = (pred_boards == target_boards).all(dim=1)
            correct_boards += matches.sum().item()
            total_boards += B
            
    square_acc = 100.0 * correct_squares / total_squares
    board_acc = 100.0 * correct_boards / total_boards
    
    return square_acc, board_acc


def analyze_model_performance(model, dataloader, class_names, device='cpu'):
    all_preds, all_labels = predict(model, dataloader, device)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize by row to get recall/accuracy per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Handle NaN if a class is missing from test set
    cm_normalized = np.nan_to_num(cm_normalized)

    # Print Accuracy per Class
    print("Accuracy per Class:\n")
    for i, class_name in enumerate(class_names):
        acc = cm_normalized[i, i] * 100
        count = cm[i, :].sum() # Total instances of this class
        print(f"{class_name:<10}: {acc:.2f}% ({int(cm[i, i])}/{count} correct)")

    print("\n" + "="*30 + "\n")

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def extract_cnn_feature_maps(model, images, device='cpu'):
    """
    Extracts features from the backbone.
    Args:
        model: The trained model.
        images: A list or tensor of images [N, 3, 64, 64].
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
    pca_feature_maps = []

    for feature_map in feature_maps:
        C, H, W = feature_map.shape
        
        # Reshape to (H*W, C) for PCA
        reshaped_map = feature_map.permute(1, 2, 0).reshape(H * W, C).detach().cpu().numpy()

        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(reshaped_map)

        # Reshape back to (H, W, 3) then transpose to (3, H, W) for plotting
        pca_result_reshaped = pca_result.reshape(H, W, n_components).transpose(2, 0, 1)
        
        pca_tensor = torch.from_numpy(pca_result_reshaped)
        pca_tensor = (pca_tensor - pca_tensor.min()) / (pca_tensor.max() - pca_tensor.min())
        
        pca_feature_maps.append(pca_tensor)

    return torch.stack(pca_feature_maps)
            