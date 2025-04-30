import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.models.models_config import MODELS

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def get_device(
    log_device: bool = True,
) -> torch.device:
    """Return the device to be used by the model."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if log_device:
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
    try:
        model_class = MODELS[model_name]["class"]
    except KeyError:
        raise ValueError(
            f"Unknown model: {model_name}. Make sure it is defined in models_config.py"
        )
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
    feature_list: list,
    model_path: str | Path,
) -> None:
    # Create directory if it doesn't exist
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": best_params,
        "model_name": best_model_name,
        "test_accuracy": accuracy,
        "input_shape": get_input_shape(best_model_name, X_train_val),
        "feature_list": feature_list,
    }

    # Use the path directly without modification
    torch.save(save_dict, model_path)
    logger.info(f"Final model saved as {model_path} with accuracy {accuracy:.2f}%")


def load_model(
    model_path: str | Path,
) -> tuple[nn.Module, list]:
    device = get_device()
    # Load the saved dictionary
    save_dict = torch.load(model_path, map_location=device)
    # Extract components
    model_name = save_dict["model_name"]
    hyperparams = save_dict["hyperparameters"]
    state_dict = save_dict["model_state_dict"]
    test_accuracy = save_dict["test_accuracy"]
    input_shape = save_dict["input_shape"]
    feature_list = save_dict["feature_list"]

    # Initialize the model with the same architecture and hyperparameters
    model, _, _, _ = initialize_model(
        model_name,
        input_shape,
        device,
        **hyperparams,
    )

    # Load the weights
    model.load_state_dict(state_dict)

    # Set model to evaluation mode
    model.eval()

    logger.info(f"Loaded {model_name} model with test accuracy {test_accuracy:.2f}%")
    logger.info(f"Input shape: {input_shape} | Features: {feature_list}")

    return model, feature_list
