import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

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
N_EPOCHS = 10
N_TRIALS = 2

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models with different feature combinations"
    )

    # Default feature list
    default_features = [
        "eda_tonic",
        "eda_phasic",
        "pupil_mean",
        "pupil_mean_tonic",
        "heartrate",
    ]

    parser.add_argument(
        "--features",
        nargs="+",
        default=default_features,
        help="List of features to include in the model. Default: all available features",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="List of model architectures to train. Default: all available models",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials to run. Default: {N_TRIALS}",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results. Default: 'results'",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Define unique experiment ID for this run
    experiment_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"experiment_{experiment_id}"
    experiment_dir = output_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)

    # Set up results tracking
    results = {
        "experiment_id": experiment_id,
        "features": args.features,
        "models": {},
        "feature_string": "_".join(args.features),
    }

    feature_str = "_".join(args.features)
    logging.info(f"Starting experiment with features: {feature_str}")

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
        # "increases": 0,
    }
    sample_duration_ms = 5000

    samples = create_samples(
        df, intervals, label_mapping, sample_duration_ms, offsets_ms
    )
    samples = make_sample_set_balanced(samples, RANDOM_SEED)

    # Use the features specified in the command-line arguments
    feature_list = args.features
    X, y, groups = transform_sample_df_to_arrays(samples, feature_columns=feature_list)

    # Split the data into training+validation set and test set
    # while respecting group structure in the data
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    idx_train_val, idx_test = next(splitter.split(X, y, groups=groups))
    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Split the training+validation set into training and validation sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    idx_train, idx_val = next(
        splitter.split(X_train_val, y_train_val, groups=groups[idx_train_val])
    )
    X_train, y_train = X_train_val[idx_train], y_train_val[idx_train]
    X_val, y_val = X_train_val[idx_val], y_train_val[idx_val]

    # Count and print the number of unique groups in each set
    for name, group_indices in [
        ("training", groups[idx_train_val][idx_train]),
        ("validation", groups[idx_train_val][idx_val]),
        ("test", groups[idx_test]),
    ]:
        logging.info(
            f"Number of unique participants in {name} set: {len(np.unique(group_indices))}"
        )

    # Scale the data
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

    # Overall best model tracking
    best_value = float("-inf")
    best_params = None
    best_model_name = None

    # Iterate through the specified models
    for model_name in args.models:
        if model_name not in MODELS:
            logging.warning(f"Model {model_name} not found in config, skipping")
            continue

        model_info = MODELS[model_name]
        logging.info(f"Training {model_name} with features: {feature_str}")

        # Create a unique study name for this model and feature combination
        study_name = f"{model_name}_{feature_str}_{experiment_id}"

        objective_function = create_objective_function(
            train_loader, val_loader, model_name, model_info
        )
        study = optuna.create_study(
            direction="maximize",  # maximize accuracy
            storage="sqlite:///db.sqlite3",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
            ),
        )
        study.optimize(objective_function, n_trials=args.trials)

        # Get the best parameters, transforming exponential parameters if needed
        best_model_params = study.best_params.copy()

        # Transform exponential parameters for actual use (but keep original in tracking)
        for param_name in model_info["hyperparameters"]:
            if model_info["hyperparameters"][param_name]["type"] == "exp":
                exp_value = best_model_params[param_name]
                best_model_params[param_name] = 2**exp_value

        # Track the best model for each architecture
        model_best_value = study.best_value

        # Save results for this model
        results["models"][model_name] = {
            "validation_accuracy": model_best_value,
            "best_model_params": best_model_params,
            "study_name": study_name,
        }

        logging.info(
            f"Best value for {model_name}: {model_best_value:.4f} (params: {best_model_params})"
        )

        # Update overall best model if this one is better
        if model_best_value > best_value:
            best_value = model_best_value
            best_params = best_model_params
            best_model_name = model_name

    logging.info(
        f"Overall Best Model: {best_model_name} with value: {best_value:.4f} (params: {best_params})"
    )

    # Update results with overall best model
    results["overall_best"] = {
        "model_name": best_model_name,
        "validation_accuracy": best_value,
        "params": best_params,
    }

    # Save results to JSON file
    results_file = experiment_dir / f"results_{feature_str}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also save a simple text summary for quick reference
    summary_file = experiment_dir / f"summary_{feature_str}.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Features: {', '.join(args.features)}\n\n")
        f.write("MODEL PERFORMANCE SUMMARY:\n")
        f.write("=" * 60 + "\n")

        # Sort models by validation accuracy (descending)
        sorted_models = sorted(
            results["models"].items(),
            key=lambda x: x[1]["validation_accuracy"],
            reverse=True,
        )

        for model_name, model_data in sorted_models:
            f.write(f"Model: {model_name}\n")
            f.write(f"Validation Accuracy: {model_data['validation_accuracy']:.4f}\n")
            f.write(f"Best Parameters: {model_data['best_model_params']}\n")
            f.write("-" * 60 + "\n")

        f.write("\nOVERALL BEST MODEL:\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Validation Accuracy: {best_value:.4f}\n")
        f.write(f"Parameters: {best_params}\n")

    # Retrain the best model with the best parameters on the entire training+validation set
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
    history = train_model(
        model,
        train_val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        N_EPOCHS,
        is_test=True,
    )
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
    logging.info(
        f"Final Model | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}"
    )

    # Update results with test performance
    results["overall_best"]["test_accuracy"] = test_accuracy
    results["overall_best"]["test_loss"] = test_loss
    results["overall_best"]["history"] = history

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Update summary
    with open(summary_file, "a") as f:
        f.write("\nTEST PERFORMANCE:\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")

    # Save model
    save_model(
        model,
        test_accuracy,
        best_params,
        best_model_name,
        X_train_val,
        feature_list,
    )


if __name__ == "__main__":
    main()
