import polars as pl

from src.features.stimulus import feature_stimulus


def explore_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    # equal to feature engineering for stimulus data
    df = feature_stimulus(df)
    return df
