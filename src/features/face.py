import polars as pl

from src.features.scaling import scale_percent_to_decimal


def Preprocess_face(df: pl.DataFrame) -> pl.DataFrame:
    # Normalize each affectiva feature to the range [0, 1] by dividing by 100.
    df = scale_percent_to_decimal(df)
    return df


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    # Drop irrelevant columns
    return df
