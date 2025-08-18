"""Using raw EDA data"""

import polars as pl

from src.features.resampling import decimate

SAMPLE_RATE = 100


def preprocess_eda(df: pl.DataFrame) -> pl.DataFrame:
    return df


def feature_eda(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate(df, factor=10)
    return df
