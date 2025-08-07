"""Heartrate data is directly taken from the Shimmer device."""

import polars as pl
from polars import col

from src.features.filtering import (
    butterworth_filter_non_causal,
    ema_smooth,
)
from src.features.resampling import decimate, interpolate_and_fill_nulls
from src.features.transforming import map_trials

SAMPLE_RATE = 100
MAX_HEARTRATE = 120
# 20 bpm above the maximum normal heart_rate to account for pain and stress


def preprocess_hr(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_heart_rate_nulls(df)
    # order matters here, as we need the original heart_rate column
    df = low_pass_filter_heart_rate_non_causal(df)
    df = ema_smooth_heart_rate(df, alpha=0.025)
    return df


def feature_hr(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate(df, factor=10)
    return df


def remove_heart_rate_nulls(
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.with_columns(
        col("ppg_heart_rate_shimmer").cast(pl.Float64),
        pl.when(col("ppg_heart_rate_shimmer") > MAX_HEARTRATE)
        .then(None)
        .when(col("ppg_heart_rate_shimmer") == -1)
        .then(None)
        .otherwise(col("ppg_heart_rate_shimmer"))
        .alias("heart_rate"),
    ).drop("ppg_heart_rate_shimmer")
    # note that the interpolate function already has the map_trials decorator
    # so we don't need to add it at the top of this function
    return interpolate_and_fill_nulls(df, ["heart_rate"])


@map_trials
def ema_smooth_heart_rate(
    df: pl.DataFrame,
    heart_rate_column: str = "heart_rate",
    alpha: float = 0.025,
) -> pl.DataFrame:
    """Causal median filter on heart_rate column."""
    return df.with_columns(
        col(heart_rate_column)
        .map_batches(lambda x: ema_smooth(x, alpha))
        .alias(heart_rate_column)
    )


@map_trials
def low_pass_filter_heart_rate_non_causal(
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
            lambda x: butterworth_filter_non_causal(
                x,
                sample_rate,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
        .name.suffix("_exploratory")
    )
