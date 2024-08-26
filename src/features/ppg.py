import polars as pl

from src.features.resampling import downsample
from src.features.transforming import map_trials


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df
