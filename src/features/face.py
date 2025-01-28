import polars as pl

from src.features.scaling import scale_percent_to_decimal

FEATURE_COLUMNS = [
    "brow_furrow",
    "cheek_raise",
    "mouth_open",
    "upper_lip_raise",
    "nose_wrinkle",
]


def preprocess_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    return df


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(FEATURE_COLUMNS)
    # no need to decimate as the sample rate is already at 10 Hz
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
