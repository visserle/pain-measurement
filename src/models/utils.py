import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models.models_config import MODELS

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def get_device() -> torch.device:
    """Return the device to be used by the model."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Set seed to {seed}")


def get_input_shape(
    model_name: str,
    X: np.ndarray | DataLoader,
) -> int:
    """Return input length and number of features (dimensions) for the model."""
    data_format = MODELS[model_name]["format"]
    if isinstance(X, DataLoader):
        X = next(iter(X))[0].numpy()
    match data_format:
        case "2D":
            input_shape = X.shape[2] * X.shape[1], 1
        case "3D":
            input_shape = X.shape[1], X.shape[2]
        case _:
            raise ValueError(f"Unknown data format: {data_format}")
    return input_shape


def initialize_model(
    model_name: str,
    input_shape: int,
    device: torch.device,
    **hyperparams,
) -> tuple[
    nn.Module,
    nn.modules.loss._Loss,
    optim.Optimizer,
    optim.lr_scheduler._LRScheduler,
]:
    # Extracting lr from hyperparams and not passing it to the model's constructor
    lr = hyperparams.pop("lr")
    model_class = MODELS[model_name]["class"]
    model = model_class(
        input_len=input_shape[0],
        input_dim=input_shape[1],
        **hyperparams,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # note that TSL uses RAdam
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
        cooldown=5,
    )
    return model, criterion, optimizer, scheduler


def save_model(
    model: nn.Module,
    accuracy: float,
    best_params: dict,
    best_model_name: str,
    X_train_val: np.ndarray | DataLoader,
):
    save_dict = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": best_params,
        "model_name": best_model_name,
        "test_accuracy": accuracy,
        "input_shape": get_input_shape(best_model_name, X_train_val),
    }
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(save_dict, f"models/{best_model_name}_{timestamp}.pt")
    logger.info(f"Model saved as {best_model_name}_{timestamp}.pt")


def load_model(model_path):
    # Load the saved dictionary
    save_dict = torch.load(model_path)

    # Extract components
    model_name = save_dict["model_name"]
    hyperparams = save_dict["hyperparameters"]
    state_dict = save_dict["model_state_dict"]
    test_accuracy = save_dict["test_accuracy"]
    input_shape = save_dict["input_shape"]

    # Initialize the model with the same architecture and hyperparameters
    model, _, _, _ = initialize_model(
        model_name,
        input_shape,
        **hyperparams,
    )

    # Load the weights
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    logger.info(
        f"Loaded {model_name} model (input_shape={input_shape}) with test accuracy {test_accuracy:.2f}%"
    )

    return model


class EarlyStopping:
    """Early stopping based on score improvement (maximization)."""

    def __init__(
        self,
        patience: int = 20,
        delta: float = 0.0,
    ) -> None:
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
