# For labeling, a new table is created in the database.

# functions for labels based on derivatives and absolute values of the time series
# plus for temperature and rating

import operator
from functools import reduce

import polars as pl
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
    df = number_intervals(df, labels)
    df = add_strictly_increasing_intervals(df)
    return df


def _get_label_intervals(
    df: pl.DataFrame,
) -> dict[int, dict[str, list[tuple[float, float]]]]:
    """Get label intervals for all stimulus seeds."""
    seeds = df.get_column("stimulus_seed").unique()
    return {seed: StimulusGenerator(seed=seed).labels for seed in seeds}


def _get_mask(
    group: pl.DataFrame,
    labels: dict[int, dict[str, list[tuple[int, int]]]],
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
    labels: dict[int, dict[str, list[tuple[int, int]]]],
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
    labels: dict[int, dict[str, list[tuple[int, int]]]],
) -> pl.DataFrame:
    """Give each interval a unique, consecutive number for each label.

    E.g.:
    0-0-0-1-1-0-0-0-0-1-1-1-1-0-0-1-1-1
    ->
    0-0-0-1-1-0-0-0-0-2-2-2-2-0-0-3-3-3
    """
    label_names = labels[list(labels)[0]].keys()

    # We need to temporarely insert a dummy line at the very beginning of the df to
    # avoid missing the very first interval (increasing interval)
    dummy_line = pl.DataFrame({column: 0 for column in df.columns}, schema=df.schema)
    df = pl.concat([dummy_line, df]).sort("trial_id", "normalized_timestamp")

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
                .cast(pl.UInt16)
                for label_name in label_names
            ]
        )
        .drop(col(r"^temp_.*$"))
    ).tail(-1)  # remove dummy line


def add_strictly_increasing_intervals(df: pl.DataFrame) -> pl.DataFrame:
    """Add a column for strictly increasing intervals (meaning no plateaus)."""
    return (
        # Add column for increasing interval segments
        df.with_columns(
            col("plateau_intervals")
            .max()
            .over("increasing_intervals")
            .alias("strictly_increasing_intervals")
        )
        .sort("trial_id", "timestamp")
        .with_columns(
            pl.when(col("strictly_increasing_intervals") > 0)
            .then(0)
            .otherwise(col("increasing_intervals"))
            .alias("strictly_increasing_intervals")
        )
        .sort("trial_id", "timestamp")
        # Recount segments to not have gaps in the numbering
        .with_columns(
            # Create a flag for the start of each new non-zero segment
            (
                (col("strictly_increasing_intervals") != 0)
                & (
                    col("strictly_increasing_intervals")
                    != col("strictly_increasing_intervals").shift(1)
                )
            ).alias("new_interval_flag")
        )
        .sort("trial_id", "timestamp")
        # Cumsum of the new segment flag to get unique identifiers for each non-zero
        # segment
        .with_columns(col("new_interval_flag").cum_sum().alias("interval_id"))
        # Replace non-zero values with their corresponding segment_id
        .with_columns(
            pl.when(col("strictly_increasing_intervals") != 0)
            .then(col("interval_id"))
            .otherwise(0)
            .alias("strictly_increasing_intervals")
            .cast(pl.UInt16)
        )
        .drop(["interval_id", "new_interval_flag"])
    ).sort("trial_id", "timestamp")
