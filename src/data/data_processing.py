"""
Create dataframes that will be inserted as tables into the database (see main function
of database_manager.py).
"""

import logging

import polars as pl
from polars import col

from src.data.data_config import DataConfig
from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.features.eda import feature_eda
from src.features.eeg import feature_eeg
from src.features.exploratory.explore_eda import explore_eda
from src.features.exploratory.explore_eeg import explore_eeg
from src.features.exploratory.explore_face import explore_face
from src.features.exploratory.explore_hr import explore_hr
from src.features.exploratory.explore_pupil import explore_pupil
from src.features.exploratory.explore_stimulus import explore_stimulus
from src.features.face import feature_face
from src.features.hr import feature_hr
from src.features.labels import add_labels
from src.features.pupil import feature_pupil
from src.features.resampling import (
    add_normalized_timestamp,
    interpolate_and_fill_nulls_in_trials,
    resample_at_10_hz_equidistant,
)
from src.features.stimulus import feature_stimulus
from src.features.transforming import merge_dfs

INVALID_PARTICIPANTS = (
    DataConfig.load_invalid_participants_config().get_column("participant_id").to_list()
)

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def create_participants_df():
    return (
        pl.read_csv(DataConfig.PARTICIPANT_DATA_FILE)
        .rename({"id": "participant_id"})
        .drop("timestamp")  # drop timestamp column for privacy reasons
    ).filter(~col("participant_id").is_in(INVALID_PARTICIPANTS))


def create_calibration_results_df():
    return (
        pl.read_csv(DataConfig.CALIBRATION_RESULTS_FILE)
        .rename({"id": "participant_id"})
        .drop("preexposure_painful", "vas0_temps", "vas70_temps")  # can be found in log
        .filter(~col("participant_id").is_in(INVALID_PARTICIPANTS))
    )


def create_measurement_results_df():
    return (
        pl.read_csv(DataConfig.MAESUREMENT_RESULTS_FILE)
        .rename({"id": "participant_id"})
        .filter(~col("participant_id").is_in(INVALID_PARTICIPANTS))
    )


def create_questionnaire_df(questionnaire: str):
    questionnaire_df = pl.read_csv(
        DataConfig.QUESTIONNAIRES_DATA_PATH / (questionnaire + "_results.csv")
    )
    # rename the id column to participant_id for consistency
    questionnaire_df = questionnaire_df.rename({"id": "participant_id"})
    # add a column for the questionnaire name
    questionnaire_df.insert_column(0, pl.lit(questionnaire).alias("questionnaire"))

    # remove particpants from PANAS that did not complete all trials because of thermode
    # issues (important for pre post comparison)
    if questionnaire == "PANAS":
        participants_with_thermode_issues = (
            (
                DataConfig.load_invalid_trials_config()
                .with_columns(
                    (col("modality").str.count_matches("thermode"))
                    .alias("issue_thermode")
                    .cast(pl.Boolean)
                )
                .filter(col("issue_thermode"))
            )
            .get_column("participant_id")
            .unique()
            .to_list()
        )
        questionnaire_df = questionnaire_df.filter(
            ~col("participant_id").is_in(participants_with_thermode_issues)
        )
    return questionnaire_df.filter(~col("participant_id").is_in(INVALID_PARTICIPANTS))


def create_seeds_df():
    config = DataConfig.load_stimulus_config()
    seeds = config["seeds"]

    # Generate labels for all seeds
    all_labels = [StimulusGenerator(config=config, seed=seed).labels for seed in seeds]

    # Get label keys from first stimulus generator
    label_keys = all_labels[0].keys()

    # Build data dictionary dynamically
    data = {"seed": seeds}
    data.update({key: [labels[key] for labels in all_labels] for key in label_keys})

    # Build schema dynamically
    schema = {
        "seed": pl.UInt16,
        **{key: pl.List(pl.List(pl.UInt32)) for key in label_keys},
    }

    return pl.DataFrame(data, schema=schema)


def remove_trials_with_thermode_or_rating_issues(
    invalid_trials_df: pl.DataFrame,
    df: pl.DataFrame,
):
    # remove trials with thermode or rating issues
    trials_with_thermode_or_rating_issues = invalid_trials_df.with_columns(
        col("participant_id").cast(pl.UInt8),
        col("trial_number").cast(pl.UInt8),
        (
            (col("modality").str.count_matches("thermode"))
            + (col("modality").str.count_matches("rating"))
        )
        .alias("issue_thermode_or_rating")
        .cast(pl.Boolean),
    ).filter(col("issue_thermode_or_rating"))
    return df.filter(
        ~pl.struct(["participant_id", "trial_number"]).is_in(
            trials_with_thermode_or_rating_issues.select(
                ["participant_id", "trial_number"]
            )
            .unique()
            .to_struct()
        )
    )


def create_trials_info_df(
    participant_id: int,
    iMotions_Marker: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a table with trial information (metadata) for each participant from iMotions
    marker data.
    """
    trials_info_df = (
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
    trials_info_df = trials_info_df.with_columns(
        trials_info_df.select(
            (col("timestamp_end") - col("timestamp_start")).alias("duration")
        )
    )
    # add column for trial number
    trials_info_df = trials_info_df.with_columns(
        pl.arange(1, trials_info_df.height + 1).alias("trial_number").cast(pl.UInt8)
    )
    # add column for participant id
    trials_info_df = trials_info_df.with_columns(
        pl.lit(participant_id).alias("participant_id").cast(pl.UInt8)
    )
    # add column for skin patch
    """
    Skin patches were distributed as follows:
    |---|---|
    | 1 | 4 |
    | 5 | 2 |
    | 3 | 6 |
    |---|---|
    Each skin patch was stimulated twice (with 1 trial of 3 min each).
    For particpants with an even id the stimulation order is:
    6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> end.
    For participants with an odd id the stimulation order is:
    1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> end.
    """
    trials_info_df = trials_info_df.with_columns(
        pl.when(col("participant_id") % 2 == 1)
        .then(((col("trial_number") - 1) % 6) + 1)
        .otherwise(6 - ((col("trial_number") - 1) % 6))
        .alias("skin_patch")
        .cast(pl.UInt8)
    )

    # change order of columns
    trials_info_df = trials_info_df.select(
        col(a := "trial_number"),
        col(b := "participant_id"),
        pl.all().exclude(a, b),
    )

    logger.debug("Created Trials_Info DataFrame for participant %s.", participant_id)
    return trials_info_df


def create_raw_data_df(
    participant_id: int,
    imotions_data_df: pl.DataFrame,
    trials_info_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create raw data tables for each data type based on the trial information.
    We only keep rows that are part of a trial from the iMotions data.
    """
    # Create a DataFrame with the trial information only
    trial_info_df = trials_info_df.select(
        col("timestamp_start").alias("trial_start"),
        col("timestamp_end").alias("trial_end"),
        col("trial_number"),
        col("participant_id"),
    ).filter(participant_id=participant_id)

    # Perform an asof join to assign trial numbers to each stimulus
    df = imotions_data_df.join_asof(
        trial_info_df,
        left_on="timestamp",
        right_on="trial_start",
        strategy="backward",
    )

    # Apply participant_id to all rows
    df = df.with_columns(pl.lit(participant_id).alias("participant_id").cast(pl.UInt8))

    # Correct backwards join by checking for the end of each trial
    df = df.with_columns(
        pl.when(
            col("timestamp").is_between(
                col("trial_start"),
                col("trial_end"),
            )
        )
        .then(col("trial_number"))
        .otherwise(None)
        .alias("trial_number")
    ).drop("trial_start", "trial_end")

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


def create_feature_data_df(
    name: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    name = name.lower()
    df = df.drop(["rownumber", "samplenumber"], strict=False)
    if "stimulus" in name:
        return feature_stimulus(df)
    elif "eda" in name:
        return feature_eda(df)
    elif "eeg" in name:
        return feature_eeg(df)
    elif "hr" in name:
        return feature_hr(df)
    elif "pupil" in name:
        return feature_pupil(df)
    elif "face" in name:
        return feature_face(df)
    else:
        raise ValueError(f"Unknown feature type: {name}")


def create_explore_data_df(
    name: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Create exploratory data for a given feature type."""
    name = name.lower()
    if "stimulus" in name:
        return explore_stimulus(df)
    elif "eda" in name:
        return explore_eda(df)
    elif "eeg" in name:
        return explore_eeg(df)
    elif "hr" in name:
        return explore_hr(df)
    elif "pupil" in name:
        return explore_pupil(df)
    elif "face" in name:
        return explore_face(df)
    else:
        raise ValueError(f"Unknown feature type: {name}")


def merge_and_label_data_dfs(
    data_dfs: list[pl.DataFrame],
    trials_info_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Merge multiple feature DataFrames into a single DataFrame. Only for data at 10 Hz.
    """
    df = merge_dfs(data_dfs)
    df = interpolate_and_fill_nulls_in_trials(df)
    df = add_normalized_timestamp(df)
    df = resample_at_10_hz_equidistant(df)
    df = add_labels(df, trials_info_df)  #  important: always add labels at the very end
    return df.drop("rownumber", "samplenumber", strict=False)
