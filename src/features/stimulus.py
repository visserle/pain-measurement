import polars as pl

from src.features.aggregation import calculate_corr
from src.features.scaling import scale_min_max, scale_percent_to_decimal


def process_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the 'Stimulus' data by scaling the 'Rating' and 'Temperature' columns to
    [0, 1].
    """
    df = scale_rating(df)
    df = scale_temperature(df)
    return df


def scale_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'Rating' column to the range [0, 1] by dividing by 100."""
    return scale_percent_to_decimal(df, exclude_additional_columns=["Temperature"])


def scale_temperature(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'Temperature' column using min-max scaling (for each trial)."""
    return scale_min_max(df, exclude_additional_columns=["Rating"])


def corr_temperature_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the correlation between 'Temperature' and 'Rating' for each trial."""
    return calculate_corr(df, "Temperature", "Rating")
