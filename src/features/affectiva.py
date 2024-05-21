# Note: domain range is 0-1, scale accordingly
import polars as pl

from src.features.scaling import scale_percent_to_decimal


def process_affectiva(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize each affectiva feature to the range [0, 1] by dividing by 100."""
    return scale_percent_to_decimal(df)
