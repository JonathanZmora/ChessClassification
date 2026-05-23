import cv2
import torch
import pulp
import numpy as np
from torchvision import transforms
from tqdm import tqdm


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Parameters:
        image (np.ndarray): an array that represents a chessboard image

    Notes:
        * Please note that to use this function, you should manually update the path at the `model`
            this is done like that, because project's requirements explicitly asked for `predict_board()`
            receive only an image as an argument.

    Returns:
        torch.Tensor: a 8x8 int64 torch tensor, that classifies every cell in the original chessboard image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("path/to/model.pth", map_location=device, weights_only=False)

    infer_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.7058276534080505, 0.6708773374557495, 0.585018515586853],
            std=[0.1342628300189972, 0.1405027061700821, 0.128083735704422]
        )
    ])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    squares = []

    H, W = image.shape[:2]
    sq_h, sq_w = H / 8.0, W / 8.0

    # Padding config
    padding: float = 1.0
    h_pad = (sq_h / 2.0) * (1 + padding)
    w_pad = (sq_w / 2.0) * (1 + padding)

    # Slice squares
    for r in range(8):
        for f in range(8):
            y_c = (r + 0.5) * sq_h
            x_c = (f + 0.5) * sq_w

            y1 = int(max(0, y_c - h_pad))
            y2 = int(min(H, y_c + h_pad))
            x1 = int(max(0, x_c - w_pad))
            x2 = int(min(W, x_c + w_pad))

            crop = image[y1:y2, x1:x2]

            crop_tensor = infer_transform(crop)

            squares.append(crop_tensor)

    # Stack into a [64, 3, H, W] tensor
    board_tensor = torch.stack(squares).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(board_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        board_probs = probs.reshape(64, 13)
        preds = solve_chess_board_ilp(board_probs)

    return torch.from_numpy(preds.reshape(8, 8)).cpu().to(dtype=torch.int64)


def predict(model, dataloader, device='cpu'):
    """
    Runs model inference on a given dataset and aggregates predictions and labels.
    Processes batched 5D tensors by flattening the batch and square sequence dimensions, 
    computes the predicted class indices, and collects them alongside the ground 
    truth labels for downstream evaluation.

    Args:
        model (nn.Module): The PyTorch model used for inference.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        device (str, optional): Target computation device. Defaults to 'cpu'.

    Returns:
        tuple[list, list]: A tuple containing:
            - all_preds: A flat list of predicted class indices for all square inputs.
            - all_labels: A flat list of ground truth class indices for all square inputs.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            B, S, C, H, W = inputs.shape
            inputs = inputs.view(B * S, C, H, W).to(device)
            labels = labels.to(device).view(-1)
            
            outputs = model(inputs) 
            preds = torch.argmax(outputs, dim=1)

            labels = labels.view(-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_preds, all_labels


def predict_with_solver(model, dataloader, device='cpu'):
    """
    Generates model predictions and refines them using a linear programming solver.
    The solver used is from solve_chess_board_ilp.

    Args:
        model (nn.Module): The PyTorch model used for inference.
        dataloader (DataLoader): DataLoader providing the batched 5D data.
        device (str, optional): Target computation device. Defaults to 'cpu'.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - all_preds: 1D array of solver-corrected predicted class indices.
            - all_targets: 1D array of ground truth class indices.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for boards, targets in tqdm(dataloader, desc="Solver Inference"):
            B, S, C, H, W = boards.shape
            
            # Flatten to pass through the model
            inputs = boards.view(B * S, C, H, W).to(device)
            targets = targets.to(device).view(-1)
            
            # Get raw logits
            logits = model(inputs)
            
            # Convert to probabilities for the solver
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Reshape back into distinct boards
            probs_by_board = probs.reshape(B, 64, 13)
            
            # Apply the solver to each board individually
            batch_preds = []
            for board_probs in probs_by_board:
                corrected_board = solve_chess_board_ilp(board_probs)
                batch_preds.extend(corrected_board)
                
            all_preds.extend(batch_preds)
            all_targets.extend(targets.cpu().numpy())
            
    return np.array(all_preds), np.array(all_targets)


def solve_chess_board_ilp(probabilities):
    """
    Resolves chess board predictions using Integer Linear Programming (ILP).
    Maximizes the joint log-probability of piece placements across all 64 squares 
    while strictly enforcing chess board state constraints.

    Args:
        probabilities (np.ndarray): A 2D array of shape (64, 13) containing the 
            raw predicted class probabilities for each square.

    Returns:
        np.ndarray: A 1D array of length 64 containing the optimally corrected class indices.
    """
    # Use log probabilities to correctly maximize joint probability
    log_probs = np.log(np.clip(probabilities, 1e-7, 1.0))
    
    # Initialize linear programming problem
    prob = pulp.LpProblem("Chess_Board_Correction", pulp.LpMaximize)
    
    squares = range(64)
    classes = range(13)
    
    # Initialize variables dict (x[i][c] = 1 if square i is of class c)
    x = pulp.LpVariable.dicts("x", (squares, classes), cat='Binary')
    
    # Define objective function: Maximize total log-probability of the board state
    prob += pulp.lpSum([log_probs[i][c] * x[i][c] for i in squares for c in classes])
    
    # Define constraints
    
    # Exactly one piece (or empty) per square
    for i in squares:
        prob += pulp.lpSum([x[i][c] for c in classes]) == 1
        
    # Exactly one king per color
    prob += pulp.lpSum([x[i][5] for i in squares]) == 1   # 5 = white king
    prob += pulp.lpSum([x[i][11] for i in squares]) == 1  # 11 = black king
    
    # Maximum 8 pawns per color
    prob += pulp.lpSum([x[i][0] for i in squares]) <= 8   # 0 = white pawn
    prob += pulp.lpSum([x[i][6] for i in squares]) <= 8   # 6 = black pawn
    
    # Maximum 16 total pieces per color
    white_classes = [0, 1, 2, 3, 4, 5]
    black_classes = [6, 7, 8, 9, 10, 11]
    prob += pulp.lpSum([x[i][c] for i in squares for c in white_classes]) <= 16
    prob += pulp.lpSum([x[i][c] for i in squares for c in black_classes]) <= 16
    
    # No pawns on the 1st or 8th Ranks
    invalid_pawn_squares = list(range(0, 8)) + list(range(56, 64))
    for i in invalid_pawn_squares:
        prob += x[i][0] == 0  
        prob += x[i][6] == 0  

    # White pieces maximum count limits
    prob += pulp.lpSum([x[i][4] for i in squares]) <= 1  # Max 1 white queen
    prob += pulp.lpSum([x[i][1] for i in squares]) <= 2  # Max 2 white rooks
    prob += pulp.lpSum([x[i][2] for i in squares]) <= 2  # Max 2 white knights
    prob += pulp.lpSum([x[i][3] for i in squares]) <= 2  # Max 2 white bishops

    # Black pieces maximum count limits
    prob += pulp.lpSum([x[i][10] for i in squares]) <= 1 # Max 1 black queen
    prob += pulp.lpSum([x[i][7] for i in squares]) <= 2  # Max 2 black rooks
    prob += pulp.lpSum([x[i][8] for i in squares]) <= 2  # Max 2 black knights
    prob += pulp.lpSum([x[i][9] for i in squares]) <= 2  # Max 2 black bishops
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Build the final corrected 64-element array
    final_preds = np.zeros(64, dtype=int)
    for i in squares:
        for c in classes:
            if pulp.value(x[i][c]) == 1.0:
                final_preds[i] = c
                
    return final_preds
    