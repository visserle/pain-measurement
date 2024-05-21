import polars as pl

from src.features.aggregation import calculate_corr


def corr_temperature_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the correlation between 'Temperature' and 'Rating' for each trial."""
    return calculate_corr(df, "Temperature", "Rating")
