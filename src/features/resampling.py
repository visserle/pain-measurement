"""
For our purposes, handling data with the duration data type in milliseconds would be
optimal. Unfortunately, Polars support of its duration data types is not fully
implemented yet: https://github.com/pola-rs/polars/issues/13560.
"""

import polars as pl
from polars import col

from src.features.transformations import map_trials


def downsample(df: pl.DataFrame, sampling_rate: int) -> pl.DataFrame:
    pass


@map_trials
def remove_duplicate_timestamps(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Remove duplicate timestamps from the DataFrame.

    For instance, the Shimmer3 GSR+ unit collects 128 samples per second but with
    only 100 unique timestamps. This function removes the duplicates.

    Note that timestamps can be duplicated across participants, so we need to group by
    trial_id (or participant_id) before removing duplicates.
    """
    return df.unique("timestamp").sort("timestamp")


def add_time_column(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a new column that contains the time from Timestamp in ms.

    Note: This datatype is not fully implemented in Polars and DuckDB yet and is not
    recommended for saving to a database.
    """
    df = df.with_columns(
        col("timestamp").cast(pl.Duration(time_unit="ms")).alias("time")
    )
    return df
