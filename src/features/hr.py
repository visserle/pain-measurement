"""Heartrate data is directly taken from the Shimmer device."""

import polars as pl
from polars import col

from src.features.resampling import decimate
from src.features.transforming import map_participants

SAMPLE_RATE = 100
MAX_HEARTRATE = 120
# 20 bpm above the maximum normal heart_rate to account for pain and stress


def preprocess_hr(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_heart_rate_nulls_and_fill_forward(df)
    return df


def feature_hr(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate(df, factor=10)
    return df


@map_participants
def remove_heart_rate_nulls_and_fill_forward(
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
    return df.with_columns(col("heart_rate").forward_fill().alias("heart_rate"))
