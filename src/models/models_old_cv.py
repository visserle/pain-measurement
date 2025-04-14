"""Nested cross-validation for model comparison for time series classification with 3D input

In research areas such as classification and regression, there are well-established stan-
dard practices for evaluation. Data partitioning is performed by using a standard k-fold
Cross-Validation (CV) to tune the model hyperparameters based on the error on a vali-
dation set, the model with the best hyperparameter combination is tested on the testing
set, standard error measures such as squared errors, absolute errors or precision, recall,
or area under the curve are computed and finally the best models are selected."""


# TODO:
# - improve model configuration
# - seperate hyperparameters from training parameters
# - also use different config format
# - add more models
# - add lr scheduler, early stopping (?), etc. pp.
# - parallelize optuna trials?
# ... print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
# for loggings with extra zeros at the beginning

import logging
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.database_manager import DatabaseManager
from src.log_config import configure_logging
from src.models.architectures import LongShortTermMemory, MultiLayerPerceptron
from src.models.data_loader import create_dataloaders, transform_sample_df_to_arrays
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import StandardScaler3D
from src.models.utils import get_device

EPOCHS = 10
K_FOLDS = 2
N_TRIALS = 3

configure_logging(stream_level=logging.DEBUG)

device = get_device()


# Define a dictionary of models with their respective hyperparameters and native data format
models_dict = {
    "MLP": {
        "class": MultiLayerPerceptron,
        "format": "2D",
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 4096},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
    # Define other models with their respective hyperparameters and input sizes
    "LSTM": {
        "class": LongShortTermMemory,
        "format": "3D",
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 1024},
            "num_layers": {"type": "int", "low": 1, "high": 2},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
}


def initialize_model(
    model_name: str,
    input_size: int,
    **hyperparams,
):
    model_class = models_dict[model_name]["class"]
    # Extracting lr and batch_size from hyperparams and not passing them to the model's constructor
    lr = hyperparams.pop("lr")
    batch_size = hyperparams.pop("batch_size")
    model = model_class(input_size=input_size, **hyperparams).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def get_input_size(
    model_name: str,
    X: np.ndarray,
):
    data_format = models_dict[model_name]["format"]
    match data_format:
        case "2D":
            return X.shape[2] * X.shape[1]
        case "3D":
            return X.shape[2]
        case _:
            raise ValueError(f"Unknown data format: {data_format}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    optimizer: optim.Optimizer,
    epochs: int,
    is_test: bool = True,
) -> dict[str, list[float]]:
    dataset = "test" if is_test else "validation"
    history = {
        "train_accuracy": [],
        "train_loss": [],
        f"{dataset}_accuracy": [],
        f"{dataset}_loss": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        # for acc metric only
        correct = 0
        total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

            # Calculate training accuracy
            y_pred_classes = (torch.sigmoid(outputs) >= 0.5).float()
            total += y_batch.size(0)
            correct += (y_pred_classes == y_batch).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)

        # Store the metrics in the history dictionary
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history[f"{dataset}_loss"].append(test_loss)
        history[f"{dataset}_accuracy"].append(test_accuracy)

        # Log progress
        max_digits = len(str(epochs))
        logging.debug(
            f"E[{+epoch + 1:>{max_digits}d}/{epochs}] "
            f"| train {epoch_loss:.4f} ({epoch_acc:.1%}) "
            f"· {dataset} {test_loss:.4f} ({test_accuracy:.1%})"
        )

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
            # Metric here is acc
            y_pred_classes = (torch.sigmoid(y_pred_logits) >= 0.5).float()
            total += y_batch.size(0)
            correct += (y_pred_classes == y_batch).sum().item()

    average_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    k_folds: int,
    **hyperparams,
):
    kfold = KFold(n_splits=k_folds, shuffle=True)  # TODO: change to GroupKFold
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # TODO: add scaler

        train_loader, val_loader = create_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=hyperparams["batch_size"],
            is_test=False,
        )
        model, criterion, optimizer = initialize_model(
            model_name, get_input_size(model_name, X_train), **hyperparams
        )
        train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)
        average_loss, accuracy = evaluate_model(model, val_loader, criterion)
        val_losses.append(average_loss)

        logging.info(
            f"Fold {fold} | Accuracy: {accuracy:.2f} | Validation Loss: {average_loss:.2f},"
        )
    return val_losses


def create_objective_function(
    X,
    y,
    model_name,
    model_info,
):
    def objective(trial):
        """
        The 'objective' function can access 'X', 'y', and 'input_size'
        even after 'create_objective_function' has finished execution.
        as they are captured by the closure.
        """
        # Dynamically suggest hyperparameters based on model_info
        hyperparams = {}
        for param_name, param_config in model_info["hyperparameters"].items():
            param_type = param_config["type"]
            match param_type:
                case "int":
                    hyperparams[param_name] = trial.suggest_int(
                        param_name, param_config["low"], param_config["high"]
                    )
                case "float":
                    hyperparams[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get(
                            "log", False
                        ),  # false by default if log is not specified
                    )
                case "categorical":
                    hyperparams[param_name] = trial.suggest_categorical(
                        param_name, param_config["choices"]
                    )
                case _:
                    raise ValueError(f"Unknown parameter type: {param_type}")

        # Perform cross-validation and return the mean validation loss of the inner loop as objective
        val_losses = cross_validate(X, y, model_name, K_FOLDS, **hyperparams)
        return np.mean(val_losses)

    return objective


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
    sample_duration_ms = 5000

    samples = create_samples(df, intervals, sample_duration_ms, label_mapping)
    # TODO: improve balance function, maybe use f1 or mcc as metric instead of accuracy
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
    X, y, groups = transform_sample_df_to_arrays(
        samples,
        feature_columns=[
            # "temperature",  # only for visualization
            # "rating"
            # "eda_raw",
            # "eda_tonic",
            "pupil_mean",
        ],
    )

    # Split the data into training+validation set and test set
    # while respecting group structure in the data
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # might not be exactly 50% due to group structure
    idx_train_val, idx_test = next(splitter.split(X, y, groups=groups))
    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    best_value = float("inf")
    best_params = None
    best_model_name = None

    for model_name, model_info in models_dict.items():
        logging.info(f"Training {model_name}...")

        objective_function = create_objective_function(
            X_train_val, y_train_val, model_name, model_info
        )
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///db.sqlite3",
            study_name=f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        study.optimize(objective_function, n_trials=N_TRIALS)

        if study.best_value < best_value:
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
    # TODO: add scaler beforehand

    train_loader, test_loader = create_dataloaders(
        X_train_val,
        y_train_val,
        X_test,
        y_test,
        batch_size=best_params["batch_size"],
    )
    model, criterion, optimizer = initialize_model(
        best_model_name,
        get_input_size(best_model_name, X_train_val),
        **best_params,
    )
    train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        EPOCHS,
        is_test=True,
    )
    average_loss, accuracy = evaluate_model(model, test_loader, criterion)
    logging.info(
        f"Final Model | Test Accuracy: {accuracy:.2f} | Test Loss: {average_loss:.2f}"
    )


if __name__ == "__main__":
    main()
