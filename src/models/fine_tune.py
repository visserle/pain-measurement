import json
import logging
from datetime import datetime
from pathlib import Path

import holoviews as hv
import hvplot.polars  # noqa
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl
import tomllib
import torch
from joblib import Memory
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.database_manager import DatabaseManager
from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.features.labels import add_labels
from src.features.resampling import (
    add_normalized_timestamp,
    interpolate_and_fill_nulls_in_trials,
)
from src.features.transforming import merge_dfs
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders, transform_sample_df_to_arrays
from src.models.data_preparation import (
    expand_feature_list,
    load_data_from_database,
    prepare_data,
)
from src.models.evaluation import evaluate_model
from src.models.hyperparameter_tuning import create_objective_function
from src.models.main_config import N_EPOCHS, RANDOM_SEED
from src.models.models_config import MODELS
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import StandardScaler3D
from src.models.training_loop import train_model
from src.models.utils import get_input_shape, initialize_model, load_model, save_model
from src.plots.make_model_plots import InferenceCache
from src.plots.model_inference import (
    _process_confidence_data,
    analyze_test_dataset_for_one_stimulus,
    plot_prediction_confidence_heatmap,
)
from src.plots.model_performance_per_participant import (
    analyze_per_participant,
    plot_feature_accuracy_comparison,
    plot_participant_accuracy_comparison,
)

pl.Config.set_tbl_rows(12)  # for the 12 trials
hv.output(widget_location="bottom", size=130)

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])

FACE_FEATURES = [
    "brow_furrow",
    "cheek_raise",
    "mouth_open",
    "nose_wrinkle",
    "upper_lip_raise",
]
EEG_FEATURES = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]


def prepare_data_for_finetuning(
    df: pl.DataFrame,
    feature_list: list,
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    random_seed: int = 42,
) -> tuple:
    """Prepare data for for single participant fine-tuning."""

    # Create and balance samples
    samples = create_samples(
        df, intervals, label_mapping, sample_duration_ms, offsets_ms
    )
    samples = make_sample_set_balanced(samples, random_seed)

    # Transform samples to arrays
    X, y, groups = transform_sample_df_to_arrays(samples, feature_columns=feature_list)

    # For single participant fine-tuning, use simple random split instead of group-based
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.50, random_state=random_seed
    )

    logger.debug("Using simple random split for single participant fine-tuning")

    # Scale the data
    train_scaler = StandardScaler3D()
    X_train = train_scaler.fit_transform(X_train)
    X_test = train_scaler.transform(X_test)

    # Log sample information
    logger.debug(
        f"Preparing data with sample duration {sample_duration_ms} ms and random seed {random_seed}"
    )
    logger.debug(f"Samples are based on intervals: {intervals}")
    logger.debug(f"Offsets for intervals: {offsets_ms}")
    logger.debug(f"Label mapping: {label_mapping}")

    logger.info("Using single participant fine-tuning mode with random splits")
    logger.info(f"Split sizes - Train: {len(X_train)}, Test: {len(X_test)}")

    return X_train, y_train, X_test, y_test


def finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    n_epochs: int,
    device: torch.device,
    freeze_encoder: bool = True,
    freeze_epochs: int = 0,
    lr: float = 1e-6,
    weight_decay: float = 0.1,
    dropout_rate: float = 0.5,
) -> dict[str, list[float]]:
    """
    Fine-tune a pre-trained iTransformer model for small datasets.

    Includes aggressive regularization strategies for tiny datasets.
    """
    dataset = "test"
    history = {
        "train_accuracy": [],
        "train_loss": [],
        f"{dataset}_accuracy": [],
        f"{dataset}_loss": [],
    }

    # Increase dropout for small datasets
    original_dropout = model.dropout
    model.dropout = dropout_rate
    if hasattr(model, "dropout_layer"):
        model.dropout_layer = nn.Dropout(dropout_rate)

    # Only train the final projection layer for small datasets
    for param in model.parameters():
        param.requires_grad = False

    # Only enable gradients for projection layer
    for param in model.projection.parameters():
        param.requires_grad = True
    logger.info("Training only projection layer for small dataset")

    # Simple SGD often works better for fine-tuning with few samples
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Use CosineAnnealingLR to avoid data leakage
    # T_max set to n_epochs for one complete cosine cycle
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-8
    )

    best_test_acc = 0
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Multiple passes over small dataset (data augmentation via dropout)
        num_passes = 3 if len(train_loader.dataset) < 100 else 1

        for pass_idx in range(num_passes):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Add noise for regularization with small datasets
                if pass_idx > 0:
                    noise = torch.randn_like(X_batch) * 0.01
                    X_batch = X_batch + noise

                optimizer.zero_grad()
                outputs = model(X_batch)

                # Add L2 regularization directly to loss
                l2_lambda = 0.01
                l2_norm = sum(p.pow(2.0).sum() for p in model.projection.parameters())
                loss = criterion(outputs, y_batch) + l2_lambda * l2_norm

                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  # Smaller clip
                optimizer.step()

                # Only count metrics from first pass
                if pass_idx == 0:
                    running_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    _, target_indices = torch.max(y_batch, 1)
                    total += y_batch.size(0)
                    correct += (predicted == target_indices).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        # Update best_test_acc when a better accuracy is achieved
        best_test_acc = max(best_test_acc, test_accuracy)

        # Update scheduler based on test accuracy
        scheduler.step()

        # Log progress
        max_digits = len(str(n_epochs))
        logger.debug(
            f"FT[{epoch + 1:>{max_digits}d}/{n_epochs}] "
            f"| train {epoch_loss:.4f} ({epoch_acc:.1%}) "
            f"· {dataset} {test_loss:.4f} ({test_accuracy:.1%}) "
            f"| best: {best_test_acc:.1%}"
        )

        # Save history
        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_acc)
        history[f"{dataset}_loss"].append(test_loss)
        history[f"{dataset}_accuracy"].append(test_accuracy)
        history["best_test_accuracy"] = best_test_acc

    return history


if __name__ == "__main__":
    configure_logging(
        stream_level=logging.DEBUG,
        ignore_libs=["matplotlib", "Comm", "bokeh", "tornado"],
    )

    feature_lists = [
        ["eeg"],
    ]
    feature_lists = expand_feature_list(feature_lists)
    feature_list = feature_lists[0]
    feature_list_str = "_".join(feature_list)

    device = "cpu"

    # Load model
    json_path = Path(f"results/experiment_{feature_list_str}/results.json")
    dictionary = json.loads(json_path.read_text())
    model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

    model = None
    model, feature_list, sample_duration_ms, intervals, label_mapping, offsets_ms = (
        load_model(model_path, device=device)
    )

    # Get some data only do get the participant ids
    df = load_data_from_database("eda")

    # Prepare data
    test_groups = prepare_data(
        df=df,
        feature_list=["eda_raw"],
        sample_duration_ms=sample_duration_ms,
        intervals=intervals,
        label_mapping=label_mapping,
        offsets_ms=offsets_ms,
        random_seed=RANDOM_SEED,
        only_return_test_groups=True,
    )

    test_set_ids = np.unique(test_groups)
    test_set_ids = [test_set_ids[3]]

    for participant_id in test_set_ids:
        logger.info(f"Fine-tuning and evaluating for participant {participant_id}")

        db = DatabaseManager()
        with db:
            eeg = (
                db.execute(
                    f"from feature_eeg select * where participant_id = {participant_id}"
                )
                .pl()
                .filter(pl.col("trial_id").is_not_null())
            )
            trials = db.execute(
                f"from trials select * where participant_id = {participant_id}"
            ).pl()
        eeg = add_normalized_timestamp(eeg)
        df = add_labels(eeg, trials)

        results = {}

        # Load model
        json_path = Path(f"results/experiment_{feature_list_str}/results.json")
        dictionary = json.loads(json_path.read_text())
        model_path = Path(dictionary["overall_best"]["model_path"].replace("\\", "/"))

        model = None
        (
            model,
            feature_list,
            sample_duration_ms,
            intervals,
            label_mapping,
            offsets_ms,
        ) = load_model(model_path, device="cpu")
        lr = dictionary["overall_best"]["params"]["lr"]
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  # note that TSL uses RAdam
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        # Prepare data
        X_train, y_train, X_test, y_test = prepare_data_for_finetuning(
            df=df,
            feature_list=feature_list,
            sample_duration_ms=sample_duration_ms,
            intervals=intervals,
            label_mapping=label_mapping,
            offsets_ms=offsets_ms,
            random_seed=RANDOM_SEED,
        )
        train_loader, test_loader = create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=64
        )

        result_df = analyze_per_participant(
            model,
            test_loader,
            test_groups=np.repeat(participant_id, len(test_loader.dataset)),
            threshold=0.50,
        )

        old_acc = result_df["accuracy"][0]

        # Use class weights for imbalanced small datasets
        train_labels = []
        for _, y_batch in train_loader:
            _, target_indices = torch.max(y_batch, 1)
            train_labels.extend(target_indices.numpy())

        # Weighted loss for imbalanced data
        classes = np.unique(train_labels)
        class_weights = compute_class_weight(
            "balanced", classes=classes, y=train_labels
        )
        class_weights = torch.FloatTensor(class_weights).to("cpu")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Fine-tune with specialized function
        history = finetune_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            n_epochs=N_EPOCHS + 10,
            device=device,
            lr=1e-3,  # Start with very small learning rate
            weight_decay=0.1,  # Strong L2 regularization
            dropout_rate=0.5,  # High dropout
        )
