import torch
import time
import copy
from tqdm.notebook import tqdm


def train(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    scheduler=None, 
    num_epochs=25, 
    device='cpu', 
    save_path='best_model.pth'
):
    """
    Generic training loop.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function (e.g., nn.CrossEntropyLoss, nn.MSELoss).
        optimizer: Optimizer (e.g., optim.Adam, optim.SGD).
        scheduler: Learning rate scheduler (optional).
        num_epochs: Number of epochs to train.
        device: 'cuda' or 'cpu'.
        save_path: Path to save the best model weights.
        
    Returns:
        model: The model loaded with the best weights.
        history: Dictionary containing loss and accuracy history.
    """
    since = time.time()
    
    # Store history for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print(f"Training on {device} for {num_epochs} epochs...")

    epoch_bar = tqdm(range(num_epochs), desc="Training Progress", leave=True)

    for epoch in epoch_bar:
        log = f"Epoch {epoch+1}/{num_epochs} | "

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
                dataloader = train_loader
            else:
                model.eval()   
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            for inputs, labels in dataloader:
                B, S, C, H, W = inputs.shape
                inputs = inputs.view(B*S, C, H, W).to(device)  # [B, 64, 3, 64, 64] -> [B * 64, 3, 64, 64]
                labels = labels.view(-1).to(device)            # [B, 64] -> [B * 64]

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            # Store history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            log += f"{phase.capitalize()}: Loss={epoch_loss:.4f} Acc={epoch_acc:.4f}  "

            if phase == 'val':
                # Save the model if it's the best one so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model, save_path) 
                    log += " [Saved Best]"

                # Scheduler Step
                if scheduler is not None:
                    # Handle ReduceLROnPlateau if chosen
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(epoch_loss)
                    else:
                        scheduler.step()

        tqdm.write(log)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
