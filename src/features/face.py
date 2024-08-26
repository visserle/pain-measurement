import polars as pl

from src.features.scaling import scale_percent_to_decimal


def preprocess_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    return df


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    # also drop irrelevant columns
    return df


def scale_face(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the facial expression columns to the range [0, 1] by dividing by 100.
    """
    return scale_percent_to_decimal(
        df,
        exclude_additional_columns=[
            "blink",
            "blinkrate",
            "interocular_distance",
        ],
    )
