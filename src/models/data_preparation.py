import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna.logging
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.data.database_manager import DatabaseManager
from src.features.labels import add_labels
from src.features.resampling import add_normalized_timestamp, interpolate_and_fill_nulls
from src.features.transforming import merge_dfs
from src.log_config import configure_logging
from src.models.data_loader import create_dataloaders, transform_sample_df_to_arrays
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
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset
from src.models.utils import get_device, set_seed

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def prepare_data(
    df: pl.DataFrame,
    feature_list: list,
    sample_duration_ms: int,
    intervals: dict,
    label_mapping: dict,
    offsets_ms: dict,
    random_seed: int = 42,
    only_return_test_groups: bool = False,  # for the participant_ids in the test set
) -> tuple:
    """Prepare data for model training, including creating and splitting samples."""

    # Create and balance samples
    samples = create_samples(
        df, intervals, label_mapping, sample_duration_ms, offsets_ms
    )
    samples = make_sample_set_balanced(samples, random_seed)

    # Transform samples to arrays
    X, y, groups = transform_sample_df_to_arrays(samples, feature_columns=feature_list)

    # Split into train+val and test sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_seed)
    idx_train_val, idx_test = next(splitter.split(X, y, groups=groups))

    if only_return_test_groups:
        return groups[idx_test]

    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Split train+val into train and val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
    idx_train, idx_val = next(
        splitter.split(X_train_val, y_train_val, groups=groups[idx_train_val])
    )
    X_train, y_train = X_train_val[idx_train], y_train_val[idx_train]
    X_val, y_val = X_train_val[idx_val], y_train_val[idx_val]

    # Scale the data
    X_train, X_val = scale_dataset(X_train, X_val)
    X_train_val, X_test = scale_dataset(X_train_val, X_test)

    # Log sample information
    logger.debug(
        f"Preparing data with sample duration {sample_duration_ms} ms and random seed {random_seed}"
    )
    logger.debug(f"Samples are based on intervals: {intervals}")
    logger.debug(f"Offsets for intervals: {offsets_ms}")
    logger.debug(f"Label mapping: {label_mapping}")

    # Log group information
    for name, group_indices in [
        ("training", groups[idx_train_val][idx_train]),
        ("validation", groups[idx_train_val][idx_val]),
        ("test", groups[idx_test]),
    ]:
        logger.info(
            f"Number of unique participants in {name} set: {len(np.unique(group_indices))}"
        )

    # Log test set participant IDs
    logger.debug(f"Participant IDs in test set: {np.unique(groups[idx_test])}")

    return X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test


def load_data_from_database(feature_list: list):
    face_features = [
        "brow_furrow",
        "cheek_raise",
        "mouth_open",
        "nose_wrinkle",
        "upper_lip_raise",
    ]
    eeg_features = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]

    if "face" in feature_list:
        feature_list = face_features + [f for f in feature_list if f != "face"]
    if "eeg" in feature_list:
        feature_list = eeg_features + [f for f in feature_list if f != "eeg"]
    # Separate EEG and non-EEG features
    eeg_features_in_list = [f for f in feature_list if f in eeg_features]
    non_eeg_features = [f for f in feature_list if f not in eeg_features]

    # Load data from database
    db = DatabaseManager()
    with db:
        if eeg_features_in_list:  # if any EEG features are requested
            if non_eeg_features:  # mixed EEG and non-EEG features
                eeg = db.get_table(
                    "Feature_EEG", exclude_trials_with_measurement_problems=True
                ).with_columns(marker_eeg=1)  #  to keep eeg sampling rate unchanged

                data = (
                    db.get_table(
                        "Merged_and_Labeled_Data",
                        exclude_trials_with_measurement_problems=True,
                    ).with_columns(marker_eeg=0)
                ).select(
                    [
                        "participant_id",
                        "trial_id",
                        "trial_number",
                        "timestamp",
                        "marker_eeg",
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
                        "marker_eeg",
                    ],
                )
                df = interpolate_and_fill_nulls(df, non_eeg_features).filter(
                    pl.col("marker_eeg") == 1  # keep only EEG data after interpolation
                )
                df = add_normalized_timestamp(df)
                df = add_labels(df, trials)
            else:  # only EEG features
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

    return df, feature_list
