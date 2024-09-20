# For labeling, a new table is created in the database.

# functions for labels based on derivatives and absolute values of the time series
# plus for temperature and rating

import operator
from functools import reduce

import polars as pl
from icecream import ic
from polars import col

from src.experiments.measurement.stimulus_generator import StimulusGenerator


def process_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Label the stimulus functions based on temperature intervals."""
    # Get label intervals for all stimulus seeds
    labels = _get_label_intervals(df)
    # Normalize timestamps for each trial
    df = df.with_columns(
        (col("timestamp") - col("timestamp").min().over("trial_id")).alias(
            "normalized_timestamp"
        )
    ).drop("duration", "timestamp_end", "timestamp_start")
    # Label the stimulus functions based on temperature intervals
    df = (
        df.group_by("stimulus_seed").map_groups(
            lambda group: label_intervals(group, labels)
        )
    ).sort("trial_id", "timestamp")
    return number_intervals(df, labels)


def _get_label_intervals(df):
    """Get label intervals for all stimulus seeds."""
    seeds = df.get_column("stimulus_seed").unique()
    return {seed: StimulusGenerator(seed=seed).labels for seed in seeds}


def _get_mask(
    group: pl.DataFrame,
    labels: dict[int, dict[str, list[tuple[float, float]]]],
    label_name: str,
) -> pl.Series:
    """Create a mask for each interval segment and combine them with an OR."""
    # without reduce and operator.or_, one would have to start with an already
    # initialized neutral element, which is not possible with lambda functions
    return reduce(
        operator.or_,
        [
            group["normalized_timestamp"].is_between(start, end)
            for start, end in labels[group.get_column("stimulus_seed").unique().item()][
                label_name
            ]
        ],
    )


def label_intervals(
    group: pl.DataFrame,
    labels: dict[int, dict[str, list[tuple[float, float]]]],
) -> pl.DataFrame:
    """Create a binary column for each label that is 1 if the timestamp is within."""
    # Determine the stimulus seed for the group
    stimulus_seed = group.get_column("stimulus_seed").unique().item()
    label_names = labels[stimulus_seed].keys()

    return group.with_columns(
        [
            pl.when(_get_mask(group, labels, label_name))
            .then(1)
            .otherwise(0)
            .alias(label_name)
            .cast(pl.UInt16)
            for label_name in label_names
        ]
    )


def number_intervals(
    df: pl.DataFrame,
    labels: dict[int, dict[str, list[tuple[float, float]]]],
) -> pl.DataFrame:
    """Give each interval a unique, consecutive number.

    E.g.:
    0-0-0-1-1-0-0-0-0-1-1-1-1-0-0-1-1-1
    ->
    0-0-0-1-1-0-0-0-0-2-2-2-2-0-0-3-3-3
    """
    label_names = labels[list(labels)[0]].keys()
    return (
        df.with_columns(
            [
                (
                    col(label_name)
                    * (
                        (col(label_name).diff().fill_null(0) == 1)
                        | (col(label_name).cum_count() == 0)
                    )
                )
                .cum_sum()
                .alias("temp_" + label_name)
                for label_name in label_names
            ]
        )
        .with_columns(
            [
                pl.when(col(label_name) == 1)
                .then(col("temp_" + label_name))
                .otherwise(0)
                .alias(label_name)
                for label_name in label_names
            ]
        )
        .drop(col(r"^temp_.*$"))
    )
