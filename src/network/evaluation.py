import torch
from tqdm.notebook import tqdm


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