import logging

import polars as pl
from polars import col
from scipy import signal

from src.features.explore_filtering import non_causal_butterworth_filter
from src.features.face import FEATURE_COLUMNS, INFO_COLUMNS, scale_face
from src.features.transforming import map_participants

SAMPLE_RATE = 10


def explore_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    df = df.select(INFO_COLUMNS + FEATURE_COLUMNS)
    df = non_causal_low_pass_filter_face(df, lowcut=0, highcut=0.2, order=2)
    return df


@map_participants
def non_causal_low_pass_filter_face(
    df: pl.DataFrame,
    lowcut: float = 0,
    highcut: float = 0.8,
    order: int = 2,
    expression_columns: list[str] = FEATURE_COLUMNS,
) -> pl.DataFrame:
    """Low-pass filter the heart_rate data using a butterworth filter. Non-causal."""
    return df.with_columns(
        col(expression_columns)
        .map_batches(
            # map_batches to apply the filter to each column
            lambda x: non_causal_butterworth_filter(
                x,
                sample_rate=SAMPLE_RATE,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
        .name.keep()
    )
