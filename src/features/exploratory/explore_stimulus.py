import polars as pl

from src.features.stimulus import feature_stimulus, preprocess_stimulus

# equal to feature engineering for stimulus data


def explore_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    """Explore stimulus data by preprocessing and extracting features."""
    df = preprocess_stimulus(df)
    df = feature_stimulus(df)
    return df
