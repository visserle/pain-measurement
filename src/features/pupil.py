import logging
import operator
from functools import reduce

import polars as pl
import scipy.signal as signal
from polars import col

from src.features.filtering import butterworth_filter
from src.features.resampling import decimate, interpolate_and_fill_nulls
from src.features.transforming import map_trials

SAMPLE_RATE = 60

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def preprocess_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = add_blink_threshold(df)
    df = extend_periods_around_blinks(df)  # blink gaps are filled with nulls
    df = interpolate_and_fill_nulls(df)
    return df


def feature_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = median_filter_pupil(df, size_in_seconds=1)
    df = average_pupils(df, result_column="pupil_mean")
    df = low_pass_filter_pupil_tonic(
        df.with_columns(pupil_mean_tonic=col("pupil_mean")),
        highcut=0.2,
    )
    df = decimate(df, factor=6)
    return df


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


@map_trials
def extend_periods_around_blinks(
    data: pl.DataFrame,
    period: int = 120,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
) -> pl.DataFrame:
    min_timestamp = data["timestamp"].min()
    max_timestamp = data["timestamp"].max()

    # Initialize the DataFrame to store the extended data
    data_extended = data

    for pupil in pupil_columns:
        blinks = _get_blink_segments(data).filter(col("pupil") == pupil.split("_")[1])

        # Expand the blink segments
        blinks_extended = blinks.with_columns(
            [
                col("start_timestamp")
                .sub(period)
                .clip(lower_bound=min_timestamp)
                .alias("expanded_start"),
                col("end_timestamp")
                .add(period)
                .clip(upper_bound=max_timestamp)
                .alias("expanded_end"),
            ]
        )

        # Create mask for the filter
        combined_filter = reduce(
            operator.or_,
            [
                col("timestamp").is_between(start, end)
                for start, end in zip(
                    blinks_extended["expanded_start"], blinks_extended["expanded_end"]
                )
            ],
            pl.lit(False),
        )

        # Apply the filter
        data_extended = data_extended.with_columns(
            pl.when(combined_filter).then(None).otherwise(col(pupil)).alias(pupil)
        )

    return data_extended


@map_trials
def _get_blink_segments(
    df: pl.DataFrame,
    pupil_columns: list[str, str] = ["pupil_r", "pupil_l"],
) -> pl.DataFrame:
    """
    Return start and end timestamps of blink segments in the pl.DataFrame.

    Note that this function does not depend on indices but on time stamps as
    indices are not preserved by the @map_trials decorator.
    """
    blink_segments_data = []
    for pupil in pupil_columns:
        participant_id = int(df["participant_id"][0])
        trial_number = int(df["trial_number"][0])
        trial_id = int(df["trial_id"].unique().item())

        # Missing data (blink or look-away) is marked with NaN
        neg_ones = df.select(col(pupil).is_null()).to_series()

        # Skip if there are no blinks
        if neg_ones.sum() == 0:
            # logger.warning(f"No blinks found in {pupil} for trial_id {trial_id}")
            continue

        # Shift the series to find the start and end of the blink segments
        start_conditions = neg_ones & ~neg_ones.shift(1)
        end_conditions = neg_ones & ~neg_ones.shift(-1)

        # Get the indices where the conditions are True
        start_indices = start_conditions.arg_true().to_list()
        end_indices = end_conditions.arg_true().to_list()

        # Check for edge cases where the first or last value is NaN
        if df[pupil][0] is None:
            start_indices.insert(0, 0)
        if df[pupil][-1] is None:
            end_indices.append(df.height - 1)

        # Get timestamps for the blink segments
        start_timestamps = df["timestamp"][start_indices].to_list()
        end_timestamps = df["timestamp"][end_indices].to_list()

        # Add to the blink segments list
        pupil_side = "r" if "_r" in pupil else "l"
        blink_segments_data.extend(
            zip(
                [pupil_side] * len(start_indices),
                start_timestamps,
                end_timestamps,
                [trial_id] * len(start_indices),
                [participant_id] * len(start_indices),
                [trial_number] * len(start_indices),
            )
        )

    # Create a DataFrame from the blink segments list
    blink_segments_df = pl.DataFrame(
        blink_segments_data,
        schema=[
            "pupil",
            "start_timestamp",
            "end_timestamp",
            "trial_id",
            "participant_id",
            "trial_number",
        ],
        strict=False,
        orient="row",
    ).sort("trial_id", "start_timestamp")

    # Add a duration column if there are any segments,
    # else create an empty DataFrame with the expected schema
    return (
        blink_segments_df.with_columns(
            (
                blink_segments_df["end_timestamp"]
                - blink_segments_df["start_timestamp"]
            ).alias("duration")
        ).sort("start_timestamp")
        if not blink_segments_df.is_empty()
        else pl.DataFrame(
            [],
            schema=[
                "pupil",
                "start_timestamp",
                "end_timestamp",
                "trial_id",
                "participant_id",
                "trial_number",
                "duration",
            ],
        )
    )


@map_trials
def median_filter_pupil(
    df: pl.DataFrame,
    size_in_seconds: int,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
) -> pl.DataFrame:
    return df.with_columns(
        col(pupil_columns).map_batches(
            lambda x: signal.medfilt(
                x,
                kernel_size=size_in_seconds * SAMPLE_RATE + 1,  # must be odd
            )
        )
    )


def average_pupils(
    df: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
    result_column: str = "pupil_mean",
) -> pl.DataFrame:
    return df.with_columns(
        ((col(pupil_columns[0]) + col(pupil_columns[1])) / 2).alias(result_column)
    )


@map_trials
def low_pass_filter_pupil_tonic(
    df: pl.DataFrame,
    sample_rate: float = SAMPLE_RATE,
    highcut: float | None = None,
    order: int = 2,
    pupil_column: list[str] = ["pupil_mean_tonic"],
) -> pl.DataFrame:
    """Low-pass filter to return the tonic component of the pupillometry."""
    return df.with_columns(
        col(pupil_column).map_batches(
            lambda x: butterworth_filter(
                x,
                SAMPLE_RATE,
                highcut=highcut,
                order=order,
            )
        )
    )
