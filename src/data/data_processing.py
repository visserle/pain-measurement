import logging

import polars as pl

from src.features.eda import clean_eda, feature_eda
from src.features.eeg import clean_eeg, feature_eeg
from src.features.face import clean_face, feature_face
from src.features.ppg import clean_ppg, feature_ppg
from src.features.pupil import clean_pupil, feature_pupil
from src.features.stimulus import clean_stimulus, feature_stimulus

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def create_clean_data_df(
    name: str,
    df: pl.DataFrame,
) -> pl.DataFrame:
    if name == "Stimulus":
        pass
    elif name == "EDA":
        df = clean_eda(df)
    elif name == "EEG":
        pass
    elif name == "PPG":
        pass
    elif name == "Pupil":
        pass
    elif name == "Face":
        pass

    return df


def create_feature_data_df(
    name: str,
    df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    if name == "Stimulus":
        pass
    elif name == "EDA":
        df = feature_eda(df)
    elif name == "EEG":
        pass
    elif name == "PPG":
        pass
    elif name == "Pupil":
        pass
    elif name == "Face":
        pass

    return df


# All functions below pertain to the creation of the raw data and metadata tables from
# iMotions data


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
        pl.col("timestamp_start").alias("trial_start"),
        pl.col("timestamp_end").alias("trial_end"),
        pl.col("trial_number"),
        pl.col("participant_id"),
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
                pl.col("timestamp").is_between(
                    pl.col("trial_start"),
                    pl.col("trial_end"),
                )
            )
            .then(pl.col("trial_number"))
            .otherwise(None)
            .alias("trial_number")
        )
        .filter(pl.col("trial_number").is_not_null())  # drop non-trial rows
        .drop(["trial_start", "trial_end"])
    )

    # Cast row number column
    df = df.with_columns(pl.col("rownumber").cast(pl.UInt32))
    # assert that rownumber is ascending
    assert df["rownumber"].is_sorted()

    # Drop rows with null values
    df = df.drop_nulls()  # this affects the rownumber in affectiva output (face)

    # Change order of columns
    df = df.select(
        pl.col(a := "trial_number"),
        pl.col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    return df


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
        iMotions_Marker.filter(pl.col("markerdescription").is_not_null())
        .select(
            [
                pl.col("markerdescription").alias("stimulus_seed").cast(pl.UInt16),
                pl.col("timestamp").alias("timestamp_start"),
            ]
        )
        # group by stimulus_seed to ensure one row per stimulus
        .group_by("stimulus_seed")
        # create columns for start and end of each stimulus
        .agg(
            [
                pl.col("timestamp_start").min().alias("timestamp_start"),
                pl.col("timestamp_start").max().alias("timestamp_end"),
            ]
        )
        .sort("timestamp_start")
    )
    # add column for duration of each stimulus
    trials_df = trials_df.with_columns(
        trials_df.select(
            (pl.col("timestamp_end") - pl.col("timestamp_start")).alias("duration")
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
    trials_df = _add_skin_area(trials_df)

    # change order of columns
    trials_df = trials_df.select(
        pl.col(a := "trial_number"),
        pl.col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    logger.debug("Created Trials DataFrame for participant %s.", participant_id)
    return trials_df


def _add_skin_area(
    trials_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Reconstructs the applied skin areas on the left forearm from the participant id and
    trial number.

    The skin areas are distributed as follows:

    |---|---|
    | 1 | 4 |
    | 5 | 2 |
    | 3 | 6 |
    |---|---|

    Each skin area is stimulated twice, where one trial is 3 minutes long.
    For particpants with an even id the stimulation order is:
    6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> end.
    For participants with an odd id the stimulation order is:
    1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> end.
    """
    return trials_df.with_columns(
        pl.when(pl.col("participant_id") % 2 == 1)
        .then(((pl.col("trial_number") - 1) % 6) + 1)
        .otherwise(6 - ((pl.col("trial_number") - 1) % 6))
        .alias("skin_area")
        .cast(pl.UInt8)
    )
