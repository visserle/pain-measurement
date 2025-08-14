import logging

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.data.database_manager import DatabaseManager
from src.features.labels import add_labels
from src.features.resampling import add_normalized_timestamp, interpolate_and_fill_nulls
from src.features.transforming import merge_dfs
from src.models.data_loader import transform_sample_df_to_arrays
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset

FACE_FEATURES = [
    "brow_furrow",
    "cheek_raise",
    "mouth_open",
    "nose_wrinkle",
    "upper_lip_raise",
]

EEG_FEATURES = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def expand_feature_list(feature_list: list) -> list:
    """Expand shorthand feature names to full feature lists."""
    expanded_list = feature_list.copy()

    if "face" in expanded_list:
        expanded_list = FACE_FEATURES + [f for f in expanded_list if f != "face"]
    if "eeg" in expanded_list:
        expanded_list = EEG_FEATURES + [f for f in expanded_list if f != "eeg"]

    return expanded_list


def _categorize_features(feature_list: list) -> tuple:
    """Separate EEG and non-EEG features from the feature list."""
    eeg_features_in_list = [f for f in feature_list if f in EEG_FEATURES]
    non_eeg_features = [f for f in feature_list if f not in EEG_FEATURES]

    return eeg_features_in_list, non_eeg_features


def load_data_from_database(feature_list: list) -> pl.DataFrame:
    """Load data from the database. Allow merging EEG and non-EEG features.

    Used in the main script to load data based on the provided feature list."""
    # Categorize features
    eeg_features_in_list, non_eeg_features = _categorize_features(feature_list)

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

    return df


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


if __name__ == "__main__":
    feature_list = ["face"]
    feature_list = expand_feature_list(feature_list)
    print(feature_list)
