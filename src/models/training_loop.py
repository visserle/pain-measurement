import logging

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.evaluation import evaluate_model

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


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
    trial: optuna.Trial | None = None,
) -> dict[str, list[float]]:
    dataset = "test" if is_test else "validation"
    history = {
        "train_accuracy": [],
        "train_loss": [],
        f"{dataset}_accuracy": [],
        f"{dataset}_loss": [],
    }

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
        logger.debug(
            f"E[{+epoch + 1:>{max_digits}d}/{n_epochs}] "
            f"| train {epoch_loss:.4f} ({epoch_acc:.1%}) "
            f"· {dataset} {test_loss:.4f} ({test_accuracy:.1%})"
        )

        # Update scheduler at the end of each epoch (no validation dependency)
        scheduler.step()

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


# not used in the current codebase, as the final model training combines train and val
# data and we cannot evaluate on the validation set anymore
class EarlyStopping:
    """Early stopping based on score improvement (maximization)."""

    def __init__(
        self,
        patience: int = 20,
        delta: float = 0.0,
    ):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_accuracy = float("-inf")
        self.early_stop = False

    def __call__(self, accuracy: float):
        if accuracy > self.best_accuracy - self.delta:
            self.best_accuracy = accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self
