"""Nested cross-validation for model comparison for time series classification with 3D input"""

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
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.log_config import configure_logging
from src.models.utils import StandardScaler3D

LEVEL = logging.DEBUG
configure_logging(stream_level=LEVEL)

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
logging.debug(f"Using device: {device}")

# Define Constants
BATCH_SIZE = 32  # not used, optimized
EPOCHS = 10
K_FOLDS = 3
N_TRIALS = 10


def time_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


class MLP(nn.Module):
    """MLP class for time series classification with 3D input"""

    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to Fortran order
        x = x.flatten(start_dim=1)  # Flatten to 2D
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out  # BCEWithLogitsLoss will apply sigmoid


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# Define a dictionary of models with their respective hyperparameters and native data format
models_dict = {
    "MLP": {
        "class": MLP,
        "format": "2D",
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 4096},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
    "LSTM": {
        "class": LSTM,
        "format": "3D",
        "hyperparameters": {
            "hidden_size": {"type": "int", "low": 128, "high": 1024},
            "num_layers": {"type": "int", "low": 1, "high": 2},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]},
        },
    },
    # Define other models with their respective hyperparameters and input sizes
}


def initialize_model(
    model_name,
    input_size,
    **hyperparams,
):
    model_class = models_dict[model_name]["class"]
    # Extracting lr and batch_size from hyperparams and not passing them to the model's constructor
    lr = hyperparams.pop("lr")  # finally a use for pop
    batch_size = hyperparams.pop("batch_size")
    model = model_class(input_size=input_size, **hyperparams).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, criterion, optimizer


def get_input_size(
    model_name,
    X,
):
    data_format = models_dict[model_name]["format"]
    match data_format:
        case "2D":
            return X.shape[2] * X.shape[1]
        case "3D":
            return X.shape[2]
        case _:
            raise ValueError(f"Unknown data format: {data_format}")


def load_dataset():
    X = np.load("data/x.npy")
    y = np.load("data/y.npy")
    return X, y


def create_dataloaders(
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    is_validation=False,
):
    if not len(X_train.shape) == len(X_test.shape) == 3:
        raise ValueError(
            "X_train and X_test must have 3 dimensions: (samples, timesteps, features)"
        )

    scaler = StandardScaler3D()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_data = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).view(-1, 1),
    )
    test_data = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).view(-1, 1),
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size)

    data_set = "Validation" if is_validation else "Test"
    logging.debug(
        f"Train Data: {len(train_data)} samples, {data_set} Data: {len(test_data)} samples"
    )

    return train_loader, test_loader


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    is_test=False,
):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)

        phase = "Final Training" if is_test else "Training"
        data_set = "Test" if is_test else "Validation"
        logging.debug(
            f"{phase} | Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, {data_set} Loss: {val_loss:.4f}, {data_set} Accuracy: {val_accuracy:.4f}"
        )


def evaluate_model(
    model,
    val_loader,
    criterion,
):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch)
            loss = criterion(y_pred_logits, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            # Metric heer is acc
            y_pred_classes = (torch.sigmoid(y_pred_logits) >= 0.5).float()
            total += y_batch.size(0)
            correct += (y_pred_classes == y_batch).sum().item()

    average_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    return average_loss, accuracy


def cross_validate(
    X,
    y,
    model_name,
    input_size,
    k_folds,
    **hyperparams,
):
    kfold = KFold(n_splits=k_folds, shuffle=True)
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_loader, val_loader = create_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=hyperparams["batch_size"],
            is_validation=True,
        )
        model, criterion, optimizer = initialize_model(
            model_name, input_size, **hyperparams
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
    input_size,
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
        val_losses = cross_validate(
            X, y, model_name, input_size, K_FOLDS, **hyperparams
        )
        return np.mean(val_losses)

    return objective


def main():
    X, y = load_dataset()
    # Split the data into training+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_value = float("inf")
    best_params = None
    best_model_name = None

    for model_name, model_info in models_dict.items():
        logging.info(f"Training {model_name}...")

        input_size = get_input_size(model_name, X_train_val)

        objective_function = create_objective_function(
            X_train_val, y_train_val, model_name, input_size, model_info
        )
        study = optuna.create_study(
            direction="minimize",
            storage="sqlite:///db.sqlite3",
            study_name=f"{model_name}_{time_stamp()}",
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
    train_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=best_params["batch_size"]
    )
    input_size = get_input_size(best_model_name, X_train_val)
    model, criterion, optimizer = initialize_model(
        best_model_name, input_size, **best_params
    )
    train_model(
        model, train_loader, test_loader, criterion, optimizer, EPOCHS, is_test=True
    )
    average_loss, accuracy = evaluate_model(model, test_loader, criterion)
    logging.info(
        f"Final Model | Test Accuracy: {accuracy:.2f} | Test Loss: {average_loss:.2f}"
    )


if __name__ == "__main__":
    main()
