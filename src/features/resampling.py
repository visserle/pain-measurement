# TODO: add anti-aliasing downsampling option

"""
Note that the usage of the pl.Duration data type is not fully supported in Polars yet.

https://github.com/pola-rs/polars/issues/13560
"""

import logging

import polars as pl
import scipy.signal as signal
from polars import col

from src.features.transforming import map_participants, map_trials

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


@map_trials
def decimate(
    df: pl.DataFrame,
    factor: int = 10,
) -> pl.DataFrame:
    """Decimate all float columns by a factor of 10.

    This function applies scipy.signal.decimate to all float columns in the DataFrame
    (except the 'timestamp' column) and gathers every 10th row for all other columns.
    """
    if sum(s.count("time") for s in df.columns) > 0:
        logger.warning(
            "More than one time column found. The additional time columns will be "
            "low-pass filtered which may lead to unexpected results."
        )
    print(df.height)

    def decimate_column(col):
        if col.dtype in [pl.Float32, pl.Float64] and col.name != "timestamp":
            return pl.from_numpy(
                signal.decimate(
                    x=col.to_numpy(),
                    q=factor,
                    ftype="iir",
                    zero_phase=True,
                )
            ).to_series()
        else:
            return col.gather_every(factor)

    return df.select(pl.all().map_batches(decimate_column))


@map_trials
def downsample(
    df: pl.DataFrame,
    new_sample_rate: int = 10,
) -> pl.DataFrame:
    # Find out current sampling rate
    current_sample_rate = round(
        1000
        / (
            df.filter(col("trial_id") == df.select(pl.first("trial_id")))
            .get_column("timestamp")
            .diff()
            .mean()
        )
    )
    if new_sample_rate >= current_sample_rate:
        raise ValueError(
            f"New sample rate {new_sample_rate} must be smaller than current sample "
            f"rate {current_sample_rate}"
        )

    # Add time column in µs for integer data type so that group_by_dynamic works
    df = add_timestamp_µs_column(df)
    # Downsample using group_by_dynamic
    df = df.group_by_dynamic(
        "timestamp_µs", every=f"{int((1000 / new_sample_rate)*1000)}i"
    ).agg(pl.all().mean())
    # Reverse the timestamp_µs column to timestamp in ms
    df = df.with_columns(
        (col("timestamp_µs") / 1000).alias("timestamp"),
        col("trial_id").cast(pl.UInt16),
        col(["trial_number", "participant_id"]).cast(pl.UInt8),
        # we should also cast the other columns to the correct data type TODO
    ).drop("timestamp_µs")

    return df


def add_timestamp_µs_column(
    df: pl.DataFrame,
    time_column: str = "timestamp",
) -> pl.DataFrame:
    """
    Create a new column that contains the timestamp in microseconds (µs).

    Casts the datatype to Int64 which allow group_by_dynamic operations.
    """
    return df.with_columns(
        (col(time_column) * 1000).cast(pl.Int64).alias(time_column + "_µs")
    )


def add_time_column(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    time_unit: str = "ms",
    new_column_name: str = "time",
) -> pl.DataFrame:
    """
    Create a new column that contains the time from Timestamp in ms.

    Note: This datatype is not fully supported in Polars and DuckDB yet.
    Use with caution.
    """
    df = df.with_columns(
        col(time_column).cast(pl.Duration(time_unit=time_unit)).alias(new_column_name)
    )
    return df
