import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            # Get predicted class indices
            _, predicted = torch.max(y_pred_logits, 1)
            # Convert one-hot targets to indices
            _, target_indices = torch.max(y_batch, 1)
            total += y_batch.size(0)
            correct += (predicted == target_indices).sum().item()

    average_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy
