import polars as pl

from src.features.transformations import map_trials, remove_duplicate_timestamps


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_duplicate_timestamps(df)

    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    return df
