import argparse
import logging
from datetime import datetime
from pathlib import Path

import optuna.logging
import polars as pl

from src.data.database_manager import DatabaseManager
from src.features.labels import add_labels
from src.features.resampling import add_normalized_timestamp, interpolate_and_fill_nulls
from src.features.transforming import merge_dfs
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders
from src.models.data_preparation import prepare_data
from src.models.main_config import (
    BATCH_SIZE,
    INTERVALS,
    LABEL_MAPPING,
    N_EPOCHS,
    N_TRIALS,
    OFFSETS_MS,
    RANDOM_SEED,
    SAMPLE_DURATION_MS,
)
from src.models.model_selection import (
    ExperimentTracker,
    run_model_selection,
    train_evaluate_and_save_best_model,
)
from src.models.models_config import MODELS
from src.models.utils import get_device, set_seed

configure_logging(
    stream_level=logging.DEBUG,
    file_path=Path(f"runs/models/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
)
optuna.logging.disable_default_handler()
optuna.logging.enable_propagation()

device = get_device()
set_seed(RANDOM_SEED)

# Default feature list
# for sanity checks, we could train on temperature and rating features only
default_features = ["eda_raw", "pupil", "hear_rate"]
face_features = [
    "cheek_raise",
    "mouth_open",
    "upper_lip_raise",
    "nose_wrinkle",
    "brow_furrow",
]
# Note that EEG data cannot be merged with other features
eeg_features = ["f3", "f4", "c3", "c4", "cz", "p3", "p4", "oz"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train models with different feature combinations"
    )

    parser.add_argument(
        "--features",
        nargs="+",
        default=default_features,
        help="List of features to include in the model. Default: all available features",
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


def main():
    args = parse_args()
    args.features = sorted(args.features)
    if "face" in args.features:
        args.features = face_features + [f for f in args.features if f != "face"]
    if "eeg" in args.features:
        args.features = eeg_features + [f for f in args.features if f != "eeg"]
    # Separate EEG and non-EEG features
    eeg_features_in_list = [f for f in args.features if f in eeg_features]
    non_eeg_features = [f for f in args.features if f not in eeg_features]

    # Create experiment tracker
    experiment_tracker = ExperimentTracker(
        features=args.features,
        sample_duration_ms=SAMPLE_DURATION_MS,
        intervals=INTERVALS,
        label_mapping=LABEL_MAPPING,
        offsets_ms=OFFSETS_MS,
    )

    # Load data from database
    db = DatabaseManager()
    with db:
        if eeg_features_in_list:  # If any EEG features are requested
            if non_eeg_features:  # Mixed EEG and non-EEG features
                eeg = db.get_table(
                    "Feature_EEG", exclude_trials_with_measurement_problems=True
                ).with_columns(marker=1)  # add marker to keep eeg sampling unchanged

                data = (
                    db.get_table(
                        "Merged_and_Labeled_Data",
                        exclude_trials_with_measurement_problems=True,
                    ).with_columns(marker=0)
                ).select(
                    [
                        "participant_id",
                        "trial_id",
                        "trial_number",
                        "timestamp",
                        "marker",
                    ]
                    + non_eeg_features
                )

                trials = db.get_table(
                    "Trials", exclude_trials_with_measurement_problems=True
                )

                df = merge_dfs(
                    [eeg, data],
                    on=[
                        "participant_id",
                        "trial_id",
                        "trial_number",
                        "timestamp",
                        "marker",
                    ],
                )
                df = interpolate_and_fill_nulls(df, non_eeg_features).filter(
                    pl.col("marker") == 1  # keep only EEG data after interpolation
                )
                df = add_normalized_timestamp(df)
                df = add_labels(df, trials)
            else:  # Only EEG features
                eeg = db.get_table(
                    "Feature_EEG",
                    exclude_trials_with_measurement_problems=True,
                )
                trials = db.get_table(
                    "Trials",
                    exclude_trials_with_measurement_problems=True,
                )
                eeg = add_normalized_timestamp(eeg)
                df = add_labels(eeg, trials)
        else:  # No EEG features
            df = db.get_table(
                "Merged_and_Labeled_Data",
                exclude_trials_with_measurement_problems=True,
            )

    # Prepare data
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = (
        prepare_data(
            df=df,
            feature_list=args.features,
            sample_duration_ms=SAMPLE_DURATION_MS,
            intervals=INTERVALS,
            label_mapping=LABEL_MAPPING,
            offsets_ms=OFFSETS_MS,
            random_seed=RANDOM_SEED,
        )
    )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, is_test=False
    )

    train_val_loader, test_loader = create_dataloaders(
        X_train_val, y_train_val, X_test, y_test, batch_size=BATCH_SIZE
    )

    # Run model selection and hyperparameter tuning
    experiment_tracker = run_model_selection(
        train_loader=train_loader,
        val_loader=val_loader,
        model_names=args.models,
        models_config=MODELS,
        n_trials=args.trials,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
        random_seed=RANDOM_SEED,
    )

    # Train final model on combined training+validation data
    experiment_tracker = train_evaluate_and_save_best_model(
        train_val_loader=train_val_loader,
        test_loader=test_loader,
        n_epochs=N_EPOCHS,
        device=device,
        experiment_tracker=experiment_tracker,
    )

    logging.info(f"Experiment with features {args.features} completed successfully")


if __name__ == "__main__":
    main()
