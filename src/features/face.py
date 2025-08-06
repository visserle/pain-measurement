from math import exp

import polars as pl
from polars import col

from src.features.filtering import adaptive_ema_smooth
from src.features.scaling import scale_percent_to_decimal
from src.features.transforming import map_trials

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


def preprocess_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    return df


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(INFO_COLUMNS + FEATURE_COLUMNS)
    df = ema_smooth_face(df, expression_column=FEATURE_COLUMNS)
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


@map_trials
def ema_smooth_face(
    df: pl.DataFrame,
    fast_alpha: float = 0.05,
    slow_alpha: float = 0.09,
    threshold: float = 0.04,
    expression_column: list = FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Causal median filter on heart_rate column."""
    return df.with_columns(
        col(expression_column).map_batches(
            lambda x: adaptive_ema_smooth(
                x,
                fast_alpha=fast_alpha,
                slow_alpha=slow_alpha,
                threshold=threshold,
            ),
            return_dtype=pl.Float64,  # can be used instead of .alias
        )
    )
