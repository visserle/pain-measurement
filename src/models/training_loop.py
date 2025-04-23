import logging

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.evaluation import evaluate_model
from src.models.utils import EarlyStopping


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    n_epochs: int,
    device: torch.device,
    is_test: bool = True,
    trial: optuna.Trial = None,
) -> dict[str, list[float]]:
    dataset = "test" if is_test else "validation"
    history = {
        "train_accuracy": [],
        "train_loss": [],
        f"{dataset}_accuracy": [],
        f"{dataset}_loss": [],
    }
    early_stopping = EarlyStopping()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        # for acc metric
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)  # raw logits
            loss = criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # from TSL
            optimizer.step()

            # Log metrics
            running_loss += loss.item() * X_batch.size(0)
            # Get predicted class indices
            _, predicted = torch.max(outputs, 1)
            # Convert one-hot to indices
            _, target_indices = torch.max(y_batch, 1)
            total += y_batch.size(0)
            correct += (predicted == target_indices).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        # Log progress
        max_digits = len(str(n_epochs))
        logging.debug(
            f"E[{+epoch + 1:>{max_digits}d}/{n_epochs}] "
            f"| train {epoch_loss:.4f} ({epoch_acc:.1%}) "
            f"· {dataset} {test_loss:.4f} ({test_accuracy:.1%})"
        )

        # Early stopping
        if epoch > 20:
            if early_stopping(test_accuracy).early_stop:
                logging.debug(f"Early stopping at epoch {epoch + 1}")
                break

        # Adjust learning rate
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step(test_accuracy)
        if learning_rate != scheduler.get_last_lr()[0]:
            logging.debug(f"Learning rate adjusted to: {scheduler.get_last_lr()[0]}")

        # Save history
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history[f"{dataset}_loss"].append(test_loss)
        history[f"{dataset}_accuracy"].append(test_accuracy)

        # Report intermediate value to Optuna for pruning
        # Pruning aborts unpromising trials early (=/= early stopping)
        if trial is not None and not is_test:
            val_accuracy = test_accuracy
            trial.report(val_accuracy, epoch)

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return history
