"""Heartrate data is directly taken from the Shimmer device."""

import polars as pl
from polars import col

from src.features.exploratory.explore_filtering import non_causal_butterworth_filter
from src.features.exploratory.explore_resampling import non_causal_decimate
from src.features.resampling import interpolate_and_fill_nulls
from src.features.transforming import map_participants

SAMPLE_RATE = 100
MAX_HEARTRATE = 120
# 20 bpm above the maximum normal heart_rate to account for pain and stress


def explore_hr(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_heart_rate_nulls(df)
    df = non_causal_low_pass_filter_heart_rate(df)
    df = non_causal_decimate(df, factor=10)
    return df


def remove_heart_rate_nulls(
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(col("heart_rate").cast(pl.Float64) > MAX_HEARTRATE)
        .then(None)
        .when(col("heart_rate").cast(pl.Float64) == -1)
        .then(None)
        .otherwise(col("heart_rate").cast(pl.Float64))
        .alias("heart_rate")
    )
    # note that the interpolate function already has the map_participants decorator
    # so we don't need to add it at the top of this function
    return interpolate_and_fill_nulls(df, ["heart_rate"])


@map_participants
def non_causal_low_pass_filter_heart_rate(
    df: pl.DataFrame,
    sample_rate: int = SAMPLE_RATE,
    lowcut: float = 0,
    highcut: float = 0.8,
    order: int = 2,
    heart_rate_column: list[str] = ["heart_rate"],
) -> pl.DataFrame:
    """Low-pass filter the heart_rate data using a butterworth filter. Non-causal."""
    return df.with_columns(
        col(heart_rate_column)
        .map_batches(
            # map_batches to apply the filter to each column
            lambda x: non_causal_butterworth_filter(
                x,
                sample_rate,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
        .name.keep()
    )
