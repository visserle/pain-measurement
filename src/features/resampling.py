"""
For our purposes, handling data with the duration data type in milliseconds would be
optimal. Unfortunately, Polars support of its duration data types is not fully
implemented yet: https://github.com/pola-rs/polars/issues/13560.
"""

import polars as pl
from polars import col

from src.features.transformations import map_participants, map_trials


def downsample(df: pl.DataFrame, sampling_rate: int) -> pl.DataFrame:
    pass


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


def add_timestamp_µs_column(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a new column that contains the timestamp in microseconds (µs).

    Casts the datatype to Int64 which allow group_by_dynamic operations.
    """
    return df.with_columns(
        (col("timestamp") * 1000).cast(pl.Int64).alias("timestamp_µs"),
    )
