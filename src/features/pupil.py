import logging

import polars as pl
from polars import col

from src.features.filtering import median_filter
from src.features.resampling import decimate
from src.features.transforming import map_participants

SAMPLE_RATE = 60

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def preprocess_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = add_blink_threshold(df)
    df = remove_blinks_and_fill_forward(df)
    return df


def feature_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = median_filter_pupil(df, size_in_seconds=1)
    df = average_pupils(df, result_column="pupil")
    df = decimate(df, factor=6)
    return df


@map_participants
def remove_blinks_and_fill_forward(
    df: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
) -> pl.DataFrame:
    """Remove blinks and forward fill the pupil columns."""
    for pupil in pupil_columns:
        df = df.with_columns(
            pl.when(col(pupil) == -1).then(None).otherwise(col(pupil)).alias(pupil)
        )
    return df.with_columns(
        [
            col(pupil)
            .forward_fill()
            .backward_fill()  # backward fill to handle initial None values
            # otherwise, some dl architectures will not work
            # backward fill is non-causal but this is only a minor detail
            .alias(pupil)
            for pupil in pupil_columns
        ]
    )


def add_blink_threshold(
    df: pl.DataFrame,
    min_threshold: float = 2.0,
    max_threshold: float = 8.0,
    pupil_columns: list[str] = ["pupil_r_raw", "pupil_l_raw"],
) -> pl.DataFrame:
    """
    Apply a threshold to the pupil size to remove values below and above the
    physiological limits (lower and upper limits of 2 and 8 mm (Mathôt, 2018; Pan et
    al., 2022)).
    """
    return df.with_columns(
        [
            pl.when(col(pupil) < min_threshold)
            .then(None)
            .when(col(pupil) > max_threshold)
            .then(max_threshold)
            .otherwise(col(pupil))
            # with this first function we remove the "_raw" suffix, all other functions
            # apply to the result of the previous function (on "pupil_r" or "pupil_l")
            .alias(pupil.removesuffix("_raw"))
            for pupil in pupil_columns
        ]
    )


@map_participants
def median_filter_pupil(
    df: pl.DataFrame,
    size_in_seconds: int,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
) -> pl.DataFrame:
    """Causal median filter on pupil columns."""
    return df.with_columns(
        col(pupil_columns).map_batches(
            lambda x: median_filter(
                x,
                window_size=size_in_seconds * SAMPLE_RATE + 1,  # odd for median
            )
        )
    )


def average_pupils(
    df: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
    result_column: str = "pupil",
) -> pl.DataFrame:
    return df.with_columns(
        ((col(pupil_columns[0]) + col(pupil_columns[1])) / 2).alias(result_column)
    )
