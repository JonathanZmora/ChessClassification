import torch
import cv2
import numpy as np
from torchvision import transforms

# --- CONSTANTS ---
# Project 2 Spec Class Mapping
IDX_TO_FEN = {
    0: 'P', 1: 'R', 2: 'N', 3: 'B', 4: 'Q', 5: 'K',
    6: 'p', 7: 'r', 8: 'n', 9: 'b', 10: 'q', 11: 'k',
    12: '1' # Empty
}

# Transform (Must match training!)
_infer_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

MODEL = None # Global model placeholder

# --- SUBMISSION FUNCTION ---
def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.
    Output: (8, 8) int64 torch tensor.
    """
    global MODEL
    if MODEL is None:
        raise RuntimeError("Model is not loaded. Set inference.MODEL = your_model")
        
    device = next(MODEL.parameters()).device
    
    # 1. Slice
    squares = []
    H, W = image.shape[:2]
    sq_h, sq_w = H / 8.0, W / 8.0
    
    # Use 0.0 padding for submission to ensure strict grid alignment
    # (Or match your best training config if 0.0 fails)
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
            
    # 2. Batch Inference
    batch = torch.stack(squares).to(device)
    
    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(batch)
        preds = torch.argmax(outputs, dim=1)
        
    return preds.view(8, 8).cpu().to(torch.int64)

# --- HELPER FOR YOUR REPORT ---
def tensor_to_fen(board_tensor):
    """
    Converts the (8, 8) output tensor back to a FEN string
    so you can compare it against your gt.csv.
    """
    grid = board_tensor.numpy() if not isinstance(board_tensor, np.ndarray) else board_tensor
    fen_rows = []
    
    for r in range(8):
        current_row = ""
        empty_count = 0
        for f in range(8):
            val = grid[r, f]
            char = IDX_TO_FEN[val]
            
            if char == '1':
                empty_count += 1
            else:
                if empty_count > 0:
                    current_row += str(empty_count)
                    empty_count = 0
                current_row += char
        
        if empty_count > 0:
            current_row += str(empty_count)
        fen_rows.append(current_row)
        
    return "/".join(fen_rows)