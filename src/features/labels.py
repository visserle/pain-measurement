import operator
from functools import reduce

import polars as pl
from polars import col

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.features.transforming import merge_dfs


def add_labels(
    data_df: pl.DataFrame,
    trials_df: pl.DataFrame,
) -> pl.DataFrame:
    """Add labels to the data DataFrame. Note that labels are based on the temperarture
    intervals of the stimulus generator and are added as binary columns to the
    data DataFrame.

    Note: Needs normalized timestamps for each trial."""
    assert "normalized_timestamp" in data_df.columns, (
        "The data DataFrame needs to have a column 'normalized_timestamp' "
        "for each trial, see resampling.py."
    )
    # Add temporary markers so that we can remove rows from trials from the real data
    trials_df = trials_df.with_columns(label_marker=pl.lit(False))
    data_df = data_df.with_columns(label_marker=pl.lit(True))

    # Merge data and trials DataFrames
    df = (
        merge_dfs(
            [data_df, trials_df],
            on=["trial_id", "participant_id", "trial_number", "label_marker"],
        )
        .drop("duration", "timestamp_end", "timestamp_start")
        # Add stimulus seed info to all columns so that we can group after it later
        # Note that ffill is sufficient here, because entry from trials_df is always
        # the first for the respective trial (same for skin_patch)
        .with_columns(col(["stimulus_seed", "skin_patch"]).forward_fill())
    )
    # Process labels
    return process_labels(df).filter(label_marker=pl.lit(True)).drop("label_marker")


def process_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Label the stimulus functions based on stimulus intervals."""
    # Get label intervals for all stimulus seeds
    labels = _get_label_intervals(df)

    # Label the stimulus functions based on stimulus intervals
    df = (
        df.group_by("stimulus_seed", maintain_order=True).map_groups(
            lambda group: label_intervals(group, labels)
        )
    ).sort("trial_id", "normalized_timestamp")
    # Give each interval a unique, consecutive number for each label
    df = number_intervals(df, labels)
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
            for start, end in labels[
                group.get_column("stimulus_seed").unique().drop_nulls().item()
            ][label_name]
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
    dummy_values = {
        column: False
        if dtype == pl.Boolean  # false only for the marker, int =/= bool
        else 0
        for column, dtype in df.schema.items()
    }
    dummy_line = pl.DataFrame(dummy_values, schema=df.schema)
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
