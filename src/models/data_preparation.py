import logging

import numpy as np
import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from src.features.labels import add_labels
from src.features.resampling import add_normalized_timestamp
from src.models.data_loader import transform_sample_df_to_arrays
from src.models.main_eeg import RANDOM_SEED
from src.models.sample_creation import create_samples, make_sample_set_balanced
from src.models.scalers import scale_dataset

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def prepare_data(
    df,
    feature_list,
    sample_duration_ms,
    random_seed,
):
    """Prepare data for model training, including creating and splitting samples."""
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
    X_train_val, y_train_val = X[idx_train_val], y[idx_train_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # Split train+val into train and val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed)
    idx_train, idx_val = next(
        splitter.split(X_train_val, y_train_val, groups=groups[idx_train_val])
    )
    X_train, y_train = X_train_val[idx_train], y_train_val[idx_train]
    X_val, y_val = X_train_val[idx_val], y_train_val[idx_val]

    # Log group information
    for name, group_indices in [
        ("training", groups[idx_train_val][idx_train]),
        ("validation", groups[idx_train_val][idx_val]),
        ("test", groups[idx_test]),
    ]:
        logger.info(
            f"Number of unique participants in {name} set: {len(np.unique(group_indices))}"
        )

    # Scale the data
    X_train, X_val = scale_dataset(X_train, X_val)
    X_train_val, X_test = scale_dataset(X_train_val, X_test)

    return X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test


def prepare_eeg_data(eeg, trials):
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
