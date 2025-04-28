import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.data.database_manager import DatabaseManager
from src.features.labels import add_labels
from src.features.resampling import add_normalized_timestamp
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders, transform_sample_df_to_arrays
from src.models.model_selection import (
    ExperimentTracker,
    run_model_selection,
    train_evaluate_and_save_best_model,
)
from src.models.models_config import MODELS
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset
from src.models.utils import (
    get_device,
    get_input_shape,
    initialize_model,
    save_model,
    set_seed,
)

RANDOM_SEED = 42
BATCH_SIZE = 64
N_EPOCHS = 100
N_TRIALS = 30

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
device = get_device()
set_seed(RANDOM_SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="Train models for EEG data")

    # Default feature list
    default_features = ["f3", "f4", "c3", "c4", "cz", "p3", "p4", "oz"]

    parser.add_argument(
        "--features",
        nargs="+",
        default=default_features,
        help="List of features to use for training the model.",
    )

    available_models = list(MODELS.keys())
    parser.add_argument(
        "--models",
        nargs="+",
        default=available_models,
        choices=available_models,
        type=lambda x: next(
            m for m in available_models if m.lower() == x.lower()
        ),  # case-insensitive matching
        help="List of model architectures to train. Default: all available models",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=N_TRIALS,
        help=f"Number of optimization trials to run. Default: {N_TRIALS}",
    )

    return parser.parse_args()


def prepare_eeg_data():
    db = DatabaseManager()
    with db:
        eeg = db.get_table("Preprocess_EEG")
        trials = db.get_table("Trials")
    eeg = add_normalized_timestamp(eeg)
    df = add_labels(eeg, trials)

    intervals = {
        "decreases": "major_decreasing_intervals",
        "increases": "strictly_increasing_intervals_without_plateaus",
    }
    label_mapping = {
        "decreases": 0,
        "increases": 1,
    }
    offsets_ms = {
        "decreases": 2000,
    }
    sample_duration_ms = 5000

    samples = create_samples(
        df, intervals, label_mapping, sample_duration_ms, offsets_ms
    )

    # Fix EEG samples to ensure consistent length
    new = []
    for sample in samples.group_by("sample_id", maintain_order=True):
        sample = sample[1]
        sample = sample.head(1250)
        while sample.height < 1250:
            sample = pl.concat([sample, sample.tail(1)])
        new.append(sample)

    samples = pl.concat(new)
    samples = make_sample_set_balanced(samples, RANDOM_SEED)

    # Select relevant columns for EEG analysis
    samples = samples.select(
        "sample_id",
        "participant_id",
        "label",
        "f3",
        "f4",
        "c3",
        "c4",
        "cz",
        "p3",
        "p4",
        "oz",
    )

    return samples


def main():
    args = parse_args()

    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        features=args.features,
        base_dir=RESULT_DIR,
    )

    # Prepare EEG data
    samples = prepare_eeg_data()
    X, y, groups = transform_sample_df_to_arrays(samples, feature_columns=args.features)

    # Split data into training, validation, and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    idx_train_val, idx_test = next(splitter.split(X, y, groups=groups))
    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Split training+validation set further
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    idx_train, idx_val = next(
        splitter.split(X_train_val, y_train_val, groups=groups[idx_train_val])
    )
    X_train, y_train = X_train_val[idx_train], y_train_val[idx_train]
    X_val, y_val = X_train_val[idx_val], y_train_val[idx_val]

    # Log information about the groups in each set
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
    X_train_val, X_test = scale_dataset(X_train_val, X_test)

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, is_test=False
    )

    train_val_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=BATCH_SIZE
    )
    # Run model selection and hyperparameter tuning
    results = run_model_selection(
        train_loader=train_loader,
        val_loader=val_loader,
        X_train_val=X_train_val,
        feature_list=args.features,
        model_names=args.models,
        models_config=MODELS,
        n_trials=args.trials,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
    )

    # Train final model on combined training+validation data
    experiment_tracker = results["overall_best"]
    train_evaluate_and_save_best_model(
        train_val_loader=train_val_loader,
        test_loader=test_loader,
        X_train_val=X_train_val,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
    )


if __name__ == "__main__":
    main()
