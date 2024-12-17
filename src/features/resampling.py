# TODO: add interpolation with zero-stuffing for up-sampling, polars does this by default using upsample
# TODO: use butterworth filter for low-pass filtering in the decimate function to avoid
# ripple in the passband


import logging

import polars as pl
import scipy.signal as signal
from polars import col

from src.features.transforming import map_trials

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


@map_trials
def decimate(
    df: pl.DataFrame,
    factor: int,
    time_column: str = "timestamp",
) -> pl.DataFrame:
    """Decimate float columns using scipy.signal.decimate (order 8 Chebyshev type I
    filter) and downsample integer columns by gathering every 'factor'.

    Note that the 'timestamp' column is not decimated, but only downsampled.
    """
    if sum(s.count("time") for s in df.columns) > 1:
        logger.warning(
            "More than one time column found. The additional time columns will be "
            "low-pass filtered via the decimate function which may lead to unexpected "
            "results."
        )

    def decimate_column(column: pl.Series) -> pl.Series:
        if column.dtype in [pl.Float32, pl.Float64] and column.name != time_column:
            return pl.from_numpy(
                signal.decimate(
                    x=column.to_numpy(),
                    q=factor,
                    ftype="iir",
                    zero_phase=True,
                )
            ).to_series()
        else:
            return column.gather_every(factor)

    return df.select(pl.all().map_batches(decimate_column))


@map_trials
def interpolate_and_fill_nulls(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    time_column: str = "timestamp",
) -> pl.DataFrame:
    """
    Linearly interpolate and fill null values of float columns in a DataFrame.
    The interpolation is done based on the time column.

    Args:
        df (pl.DataFrame): The DataFrame to interpolate
        columns (list[str], optional): The columns to interpolate.
            If None, all float columns are interpolated. Defaults to None.
        time_column (str, optional): The time column. Defaults to "timestamp".
    """
    # Note that maybe using NaN as null value is better as NaN is a float in Polars
    # and we wouldn't need a selector
    # However, this would need a rewrite and testing of some parts of the pipeline
    # Also refactoring this function into interpolate and fill_nulls would make sense
    selected_columns = columns or df.select(pl.selectors.by_dtype(pl.Float64)).columns

    return df.with_columns(
        [col(column).interpolate_by(col(time_column)) for column in selected_columns]
    ).with_columns(
        [
            col(column).fill_null(strategy="forward").fill_null(strategy="backward")
            for column in selected_columns
        ]
    )


@map_trials
def resample_to_equidistant_ms(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    group_by: str = "trial_id",
    gather_every: int | None = None,
):
    """
    Resample the DataFrame to equidistant time steps with a resolution of 1 ms.
    Note that this rounds every timestamp to the nearest millisecond.

    For a lower resolution, use the gather_every function to get back to the
    original sampling rate. Use decimate to downsample the data.

    For sanity checks, one could use the following code snippet:
    ```
    (
        df.with_columns(col("timestamp_").diff().over("trial_id").alias("diff"))
        .get_column("diff")
        .mean(),
        df.with_columns(col("timestamp").diff().over("trial_id").alias("diff"))
        .get_column("diff")
        .std(),
    )
    ```
    """
    df = df.with_columns(col("timestamp").round(0).alias("timestamp").cast(pl.Int64))
    df = df.upsample(
        time_column="timestamp",
        every="1i",  # otherwise we would lose data
        maintain_order=True,
        group_by="trial_id",
    ).with_columns(
        # do not lose crucial information
        pl.col(pl.selectors.INTEGER_DTYPES).forward_fill()
    )

    df = interpolate_and_fill_nulls(df, time_column="timestamp")
    if gather_every:
        df = df.gather_every(gather_every)
    return df


def add_time_column(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    time_unit: str = "ms",
    new_column_name: str = "time",
) -> pl.DataFrame:
    """
    Create a new column that contains the time from Timestamp in ms.

    Note: This datatype is not fully supported in Polars and DuckDB yet.
    Use with caution. https://github.com/pola-rs/polars/issues/13560
    """
    df = df.with_columns(
        col(time_column).cast(pl.Duration(time_unit=time_unit)).alias(new_column_name)
    )
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
