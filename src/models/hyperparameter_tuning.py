import logging

import optuna
import torch
from torch.utils.data import DataLoader

from src.models.evaluation import evaluate_model
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def create_objective_function(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    model_info: dict,
    device: torch.device,
    n_epochs: int,
) -> callable:
    """
    Create an objective function for hyperparameter tuning using Optuna.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        The 'objective' function can access 'X', 'y', and 'input_size' even after
        'create_objective_function' has finished execution as they are captured by the
        function closure (similar to partial functions).
        """
        # try-except block to prevent memory leaks if optuna prunes the trial
        try:
            # Suggest hyperparameters
            hyperparams = suggest_hyperparameters(trial, model_info)

            # Train the model with the suggested hyperparameters
            model, criterion, optimizer, scheduler = initialize_model(
                model_name,
                get_input_shape(model_name, val_loader),
                device,
                **hyperparams,
            )
            train_model(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                n_epochs,
                device,
                is_test=False,
                trial=trial,
            )
            average_loss, accuracy = evaluate_model(
                model, val_loader, criterion, device
            )

            logger.info(
                f"Validation Loss: {average_loss:.2f} | Validation Accuracy: {accuracy:.2f}"
            )
            return accuracy
        finally:
            # Clean up GPU memory if using CUDA
            if device == "cuda" or (
                isinstance(device, torch.device) and device.type == "cuda"
            ):
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after training model")

    return objective  # returns a function


def suggest_hyperparameters(
    trial: optuna.Trial,
    model_info: dict,
) -> dict:
    # Dynamically suggest hyperparameters based on model_info
    hyperparams = {}
    exp_params = {}
    for param_name, param_config in model_info["hyperparameters"].items():
        param_type = param_config["type"]
        match param_type:
            case "int":
                hyperparams[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
            case "exp":
                # we need to store exp values separately as optuna does not support
                # suggesting power of 2 directly
                exp_value = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                )
                hyperparams[param_name] = 2**exp_value
                exp_params[param_name] = exp_value
            case "float":
                hyperparams[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            case "categorical":
                hyperparams[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"],
                )
            case _:
                raise ValueError(f"Unknown parameter type: {param_type}")

    return hyperparams
