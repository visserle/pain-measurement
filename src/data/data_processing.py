"""
Dataframe functions for each step in the pipeline.
Results will be inserted into the database.
"""

import logging

import polars as pl
from polars import col

from src.data.data_config import DataConfig
from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.features.eda import feature_eda, preprocess_eda
from src.features.eeg import feature_eeg, preprocess_eeg
from src.features.face import feature_face, preprocess_face
from src.features.labels import process_labels
from src.features.ppg import feature_ppg, preprocess_ppg
from src.features.pupil import feature_pupil, preprocess_pupil
from src.features.stimulus import feature_stimulus, preprocess_stimulus
from src.features.transforming import interpolate_and_fill_nulls, merge_dfs

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def create_participants_df():
    pass


def create_questionnaire_df():
    pass


def create_seeds_df():
    config = DataConfig.load_stimulus_config()
    seeds = config["seeds"]
    return pl.DataFrame(
        {
            "seed": seeds,
            "major_decreasing_intervals": [
                StimulusGenerator(config=config, seed=seed).labels[
                    "major_decreasing_intervals"
                ]
                for seed in seeds
            ],
        },
        schema={
            "seed": pl.UInt16,
            "major_decreasing_intervals": pl.List(pl.List(pl.UInt32)),
        },
    )


def create_trials_df(
    participant_id: str,
    iMotions_Marker: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a table with trial information (metadata) for each participant from iMotions
    marker data.
    """
    trials_df = (
        # 'markerdescription' from imotions contains the onset and offset of each
        # stimulus
        # drop all rows where the markerdescription is null to get the start and end
        # of each stimulus
        iMotions_Marker.filter(col("markerdescription").is_not_null())
        .select(
            [
                col("markerdescription").alias("stimulus_seed").cast(pl.UInt16),
                col("timestamp").alias("timestamp_start"),
            ]
        )
        # group by stimulus_seed to ensure one row per stimulus
        .group_by("stimulus_seed")
        # create columns for start and end of each stimulus
        .agg(
            [
                col("timestamp_start").min().alias("timestamp_start"),
                col("timestamp_start").max().alias("timestamp_end"),
            ]
        )
        .sort("timestamp_start")
    )
    # add column for duration of each stimulus
    trials_df = trials_df.with_columns(
        trials_df.select(
            (col("timestamp_end") - col("timestamp_start")).alias("duration")
        )
    )
    # add column for trial number
    trials_df = trials_df.with_columns(
        pl.arange(1, trials_df.height + 1).alias("trial_number").cast(pl.UInt8)
    )
    # add column for participant id
    trials_df = trials_df.with_columns(
        pl.lit(participant_id).alias("participant_id").cast(pl.UInt8)
    )
    # add column for skin area
    """
    Skin areas were distributed as follows:
    |---|---|
    | 1 | 4 |
    | 5 | 2 |
    | 3 | 6 |
    |---|---|
    Each skin area was stimulated twice (with 1 trial of 3 min each).
    For particpants with an even id the stimulation order is:
    6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> end.
    For participants with an odd id the stimulation order is:
    1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> end.
    """
    trials_df = trials_df.with_columns(
        pl.when(col("participant_id") % 2 == 1)
        .then(((col("trial_number") - 1) % 6) + 1)
        .otherwise(6 - ((col("trial_number") - 1) % 6))
        .alias("skin_area")
        .cast(pl.UInt8)
    )

    # change order of columns
    trials_df = trials_df.select(
        col(a := "trial_number"),
        col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    logger.debug("Created Trials DataFrame for participant %s.", participant_id)
    return trials_df


def create_raw_data_df(
    participant_id: int,
    imotions_data_df: pl.DataFrame,
    trials_df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """
    Create raw data tables for each data type based on the trial information.
    We only keep rows that are part of a trial from the iMotions data.
    """
    # Create a DataFrame with the trial information only
    trial_info_df = trials_df.select(
        col("timestamp_start").alias("trial_start"),
        col("timestamp_end").alias("trial_end"),
        col("trial_number"),
        col("participant_id"),
    )

    # Perform an asof join to assign trial numbers to each stimulus
    df = imotions_data_df.join_asof(
        trial_info_df,
        left_on="timestamp",
        right_on="trial_start",
        strategy="backward",
    )

    # Correct backwards join by checking for the end of each trial
    df = (
        df.with_columns(
            pl.when(
                col("timestamp").is_between(
                    col("trial_start"),
                    col("trial_end"),
                )
            )
            .then(col("trial_number"))
            .otherwise(None)
            .alias("trial_number")
        )
        .filter(col("trial_number").is_not_null())  # drop non-trial rows
        .drop(["trial_start", "trial_end"])
    )

    # Cast row number column
    df = df.with_columns(col("rownumber").cast(pl.UInt32))
    # assert that rownumber is ascending
    assert df["rownumber"].is_sorted()

    # Change order of columns
    df = df.select(
        col(a := "trial_number"),
        col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    return df


def create_preprocess_data_df(
    name: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    if "Stimulus" in name:
        return preprocess_stimulus(df)
    elif "EDA" in name:
        return preprocess_eda(df)
    elif "EEG" in name:
        return preprocess_eeg(df)
    elif "PPG" in name:
        return preprocess_ppg(df)
    elif "Pupil" in name:
        return preprocess_pupil(df)
    elif "Face" in name:
        return preprocess_face(df)


def create_feature_data_df(
    name: str,
    df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    if "Stimulus" in name:
        return feature_stimulus(df)
    elif "EDA" in name:
        return feature_eda(df)
    elif "EEG" in name:
        return feature_eeg(df)
    elif "PPG" in name:
        return feature_ppg(df)
    elif "Pupil" in name:
        return feature_pupil(df)
    elif "Face" in name:
        return feature_face(df)


def merge_feature_data_dfs(
    dfs: list[pl.DataFrame],
) -> pl.DataFrame:
    """
    Merge multiple feature DataFrames into a single DataFrame.
    """
    df = merge_dfs(dfs)
    df = interpolate_and_fill_nulls(df)
    # TODO: add final downsample
    return df
