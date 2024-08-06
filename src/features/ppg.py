import polars as pl

from src.features.transformations import map_trials, remove_dulpicate_timestamps


def clean_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_dulpicate_timestamps(df)

    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    return df
