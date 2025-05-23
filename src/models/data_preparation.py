import logging

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.models.data_loader import transform_sample_df_to_arrays
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def prepare_data(
    df: pl.DataFrame,
    feature_list: list,
    sample_duration_ms: int = 5000,
    random_seed: int = 42,
    only_return_test_groups: bool = False,  # for the participant_ids in the test set
) -> tuple:
    """Prepare data for model training, including creating and splitting samples."""
    intervals = {
        "increases": "strictly_increasing_intervals",
        "decreases": "major_decreasing_intervals",
    }
    label_mapping = {
        "increases": 0,
        "decreases": 1,
    }
    offsets_ms = {
        "increases": 0,
        "decreases": 2000,
    }

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
