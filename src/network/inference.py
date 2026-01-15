import torch
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


_infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL = None # Global model placeholder


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    Output: (8, 8) int64 torch tensor.
    """
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model is not loaded. Set inference.MODEL = your_model")
        
    device = next(MODEL.parameters()).device
    
    # Slice
    squares = []
    H, W = image.shape[:2]
    sq_h, sq_w = H / 8.0, W / 8.0
    
    padding_pct = 0.0 
    h_pad = (sq_h / 2.0) * (1 + padding_pct)
    w_pad = (sq_w / 2.0) * (1 + padding_pct)

    for r in range(8):
        for f in range(8):
            y_c = (r + 0.5) * sq_h
            x_c = (f + 0.5) * sq_w
            
            y1 = int(max(0, y_c - h_pad))
            y2 = int(min(H, y_c + h_pad))
            x1 = int(max(0, x_c - w_pad))
            x2 = int(min(W, x_c + w_pad))
            
            crop = image[y1:y2, x1:x2]
            tensor = _infer_transform(crop)
            squares.append(tensor)
            
    # Batch Inference
    batch = torch.stack(squares).to(device)
    
    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(batch)
        preds = torch.argmax(outputs, dim=1)
        
    return preds.view(8, 8).cpu().to(torch.int64)
