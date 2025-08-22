import polars as pl

from src.features.scaling import scale_min_max, scale_percent_to_decimal


def feature_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_rating(df)
    df = scale_temperature(df)
    # no need to downsample, as the stimulus data is already at 10 Hz
    return df


def scale_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'rating' column to the range [0, 1] by dividing by 100."""
    return scale_percent_to_decimal(df, exclude_additional_columns=["temperature"])


def scale_temperature(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'temperature' column using min-max scaling (for each trial).
    Since temperature is the ground truth already, there is no problem with non-causal
    scaling."""
    return scale_min_max(df, exclude_additional_columns=["rating"])
