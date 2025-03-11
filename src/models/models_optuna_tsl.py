# TODO:
# - improve model configuration
# - seperate hyperparameters from training parameters
# - also use different config format
# - add more models
# - parallelize optuna trials?

import logging
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from icecream import ic
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from src.data.database_manager import DatabaseManager
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders, transform_sample_df_to_arrays
from src.models.models_config import MODELS
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset
from src.models.utils import (
    EarlyStopping,
    get_device,
    get_input_shape,
    initialize_model,
    save_model,
    set_seed,
)

RANDOM_SEED = 42
BATCH_SIZE = 64
N_EPOCHS = 100
N_TRIALS = 3  # number of trials for hyperparameter optimization

configure_logging(stream_level=logging.DEBUG)
device = get_device()
set_seed(RANDOM_SEED)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    n_epochs: int,
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
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

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


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
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


def create_objective_function(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name,
    model_info,
) -> callable:
    def objective(trial: optuna.Trial) -> float:
        """
        The 'objective' function can access 'X', 'y', and 'input_size' even after
        'create_objective_function' has finished execution as they are captured by the
        function closure (similar to partial functions).
        """
        # Dynamically suggest hyperparameters based on model_info
        hyperparams = {}
        for param_name, param_config in model_info["hyperparameters"].items():
            param_type = param_config["type"]
            match param_type:
                case "int":
                    hyperparams[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                    )
                case "power2":
                    hyperparams[param_name] = 2 ** trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                    )
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
            N_EPOCHS,
            is_test=False,
            trial=trial,
        )
        average_loss, accuracy = evaluate_model(model, val_loader, criterion)

        logging.info(
            f"Validation Loss: {average_loss:.2f} | Validation Accuracy: {accuracy:.2f}"
        )
        return accuracy

    return objective  # returns a function


def main():
    db = DatabaseManager()
    with db:
        df = db.get_table(
            "Merged_and_Labeled_Data",
            exclude_trials_with_measurement_problems=True,
        )

    intervals = {
        # "decreases": "decreasing_intervals",
        "decreases": "major_decreasing_intervals",
        "increases": "strictly_increasing_intervals_without_plateaus",
        # "plateaus": "plateau_intervals",
    }
    label_mapping = {
        "decreases": 0,
        "increases": 1,
        # "plateaus": 1,
    }
    offsets_ms = {
        "decreases": 2000,
        "increases": 0,
    }
    sample_duration_ms = 5000

    samples = create_samples(
        df, intervals, label_mapping, sample_duration_ms, offsets_ms
    )
    samples = make_sample_set_balanced(samples)
    samples = samples.select(
        "sample_id",
        "participant_id",
        "rating",
        "temperature",
        "eda_raw",
        "eda_tonic",
        "eda_phasic",
        "pupil_mean",
        "pupil_mean_tonic",
        "label",
    )
    feature_list = [
        # "temperature",  # only for visualization
        # "rating"
        # "eda_raw",
        "eda_tonic",
        "eda_phasic",
        "pupil_mean",
    ]

    X, y, groups = transform_sample_df_to_arrays(samples, feature_columns=feature_list)

    print(X.shape, y.shape, groups.shape)

    # Split the data into training+validation set and test set
    # while respecting group structure in the data
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    idx_train_val, idx_test = next(splitter.split(X, y, groups=groups))
    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Split the training+validation set into training and validation sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    idx_train, idx_val = next(
        splitter.split(X_train_val, y_train_val, groups=groups[idx_train_val])
    )
    X_train, y_train = X_train_val[idx_train], y_train_val[idx_train]
    X_val, y_val = X_train_val[idx_val], y_train_val[idx_val]

    # Scale the data
    # TODO: fix scaler? not sure if correct
    X_train, X_val = scale_dataset(X_train, X_val)

    # Create dataloaders for training and validation sets
    train_loader, val_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=BATCH_SIZE,
        is_test=False,
    )

    best_value = float("-inf")
    best_params = None
    best_model_name = None

    for model_name, model_info in MODELS.items():
        logging.info(f"Training {model_name}...")

        objective_function = create_objective_function(
            train_loader, val_loader, model_name, model_info
        )
        study = optuna.create_study(
            direction="maximize",  # maximize accuracy
            storage="sqlite:///db.sqlite3",
            study_name=f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )
        study.optimize(objective_function, n_trials=N_TRIALS)

        if study.best_value > best_value:
            best_value = study.best_value
            best_params = study.best_params
            best_model_name = model_name

        logging.info(
            f"Best value for {model_name}: {study.best_value} (params: {study.best_params})"
        )

    logging.info(
        f"Overall Best Model: {best_model_name} with value: {best_value} (params: {best_params})"
    )

    # Retrain the model with the best parameters on the entire training+validation set
    # TODO: fix scaler
    X_train_val, X_test = scale_dataset(X_train_val, X_test)

    train_val_loader, test_loader = create_dataloaders(
        X_train_val,
        y_train_val,
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
    )
    model, criterion, optimizer, scheduler = initialize_model(
        best_model_name,
        get_input_shape(best_model_name, X_train_val),
        device,
        **best_params,
    )
    train_model(
        model,
        train_val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        N_EPOCHS,
        is_test=True,
    )
    average_loss, accuracy = evaluate_model(model, test_loader, criterion)
    logging.info(
        f"Final Model | Test Loss: {average_loss:.2f} | Test Accuracy: {accuracy:.2f}"
    )

    # Save model
    save_model(model, accuracy, best_params, best_model_name, X_train_val, feature_list)


if __name__ == "__main__":
    main()
