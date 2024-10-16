# TODO:
# - improve docstrings
# insightfull report: https://github.com/kinleyid/PuPL/blob/master/manual.pdf

import logging

import polars as pl
from polars import col

from src.features.filtering import filter_butterworth
from src.features.resampling import downsample
from src.features.transforming import map_trials

SAMPLE_RATE = 60

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def preprocess_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = filter_pupil(df)  # this is just a low pass filter, quick & dirty TODO: improve
    return df


def feature_pupil(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df


@map_trials
def filter_pupil(
    df: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
    sample_rate: float = SAMPLE_RATE,
    lowcut: float = 0,
    highcut: float = 0.2,
    order: int = 2,
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(pupil_columns)
        .map_batches(  # use map_batches to apply the filter to each column
            lambda x: filter_butterworth(
                x,
                SAMPLE_RATE,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
        .name.suffix("_filtered")
    )


def add_blink_threshold(
    df: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r", "pupil_l"],
    min_threshold: float = 1.5,
    max_threshold: float = 9.0,
) -> pl.DataFrame:
    """
    1.5 and > 9.0 according to Kret et al., 2014
    # https://github.com/ElioS-S/pupil-size/blob/944523bff0ca583039039a3008ac1171ab46400a/code/helperFunctions/rawDataFilter.m#L66

    physiological lower and upper limits of 2 and 8 mm,  Mathôt & Vilotijević (2023)"
    """
    return df.with_columns(
        [
            pl.when(pl.col(pupil) < min_threshold)
            .then(-1)
            .when(pl.col(pupil) > max_threshold)
            .then(9.0)
            .otherwise(pl.col(pupil))
            .alias(pupil + "_thresholded")
            for pupil in pupil_columns
        ]
    )


@map_trials
def _get_blink_segments(
    df: pl.DataFrame,
    pupil_columns: list[str, str] = ["pupil_r_thresholded", "pupil_l_thresholded"],
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

        # Missing data (blink or look-away) is marked with -1
        neg_ones = df[pupil] == -1

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

        # Check for edge cases where the first or last value is -1
        if df[pupil][0] == -1:
            start_indices.insert(0, 0)
        if df[pupil][-1] == -1:
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
def extend_periods_around_blinks(
    data: pl.DataFrame,
    pupil_columns: list[str] = ["pupil_r_thresholded", "pupil_l_thresholded"],
    period: int = 120,
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
                pl.col("start_timestamp")
                .sub(period)
                .clip(lower_bound=min_timestamp)
                .alias("expanded_start"),
                pl.col("end_timestamp")
                .add(period)
                .clip(upper_bound=max_timestamp)
                .alias("expanded_end"),
            ]
        ).with_columns(
            (col("expanded_end") - col("expanded_start")).alias("expanded_duration")
        )

        # Create the filter by combining the is_between conditions for each range
        combined_filter = pl.lit(False)
        for start, end in zip(
            blinks_extended["expanded_start"], blinks_extended["expanded_end"]
        ):
            condition = pl.col("timestamp").is_between(start, end)
            combined_filter |= condition

        # Apply the filter to the DataFrame
        data_extended = data_extended.with_columns(
            pl.when(combined_filter)
            .then(None)
            .otherwise(pl.col(pupil))
            .alias(pupil.replace("_thresholded", "_extended")),
        )

    return data_extended


@map_trials
def interpolate_pupillometry(
    df: pl.DataFrame,
    pupil_columns: str | list[str] = ["pupil_r_extended", "pupil_l_extended"],
) -> pl.DataFrame:
    # Linearly interpolate and fill edge cases when the first or last value is null
    # NOTE: cubic spline? TODO
    for pupil in pupil_columns:
        df = df.with_columns(
            pl.col(pupil)
            .interpolate()
            .forward_fill()  # Fill remaining edge cases
            .backward_fill()
            .alias(pupil)
        )
    logger.warning(
        "Interpolation method does not consider non-equidistant time stamps. "
        "This is a bug that needs to be fixed by using interpolate_by."
    )
    return df
