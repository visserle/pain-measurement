import polars as pl
from polars import col

from src.features.filtering import ema_smooth
from src.features.scaling import scale_percent_to_decimal
from src.features.transforming import map_participants

INFO_COLUMNS = [
    "participant_id",
    "trial_id",
    "trial_number",
    "timestamp",
]

FEATURE_COLUMNS = [
    "brow_furrow",
    "cheek_raise",
    "mouth_open",
    "upper_lip_raise",
    "nose_wrinkle",
]


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    df = df.select(INFO_COLUMNS + FEATURE_COLUMNS)
    df = ema_smooth_face(df, alpha=0.05, expression_column=FEATURE_COLUMNS)
    # no need to decimate as the sample rate is already at 10 Hz (roughly)
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


@map_participants
def ema_smooth_face(
    df: pl.DataFrame,
    alpha: float,
    expression_column: list = FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Causal median filter."""
    return df.with_columns(
        col(expression_column).map_batches(
            lambda x: ema_smooth(x, alpha),
            return_dtype=pl.Float64,  # can be used instead of .alias
        )
    )
