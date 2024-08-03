"""
Functions to load iMotions data from the file system and prepare them for database
insertion.

As the raw dataset is highly compressed by Apache Arrow (> 5 GB), all modalities are
loaded into memory for processing at once.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.data_config import DataConfig

IMOTIONS_DATA_CONFIG = DataConfig.load_imotions_config()
IMOTIONS_DATA_PATH = DataConfig.IMOTIONS_DATA_PATH

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def load_imotions_data_dfs(participant_id: int) -> dict[str, dict[str, str]]:
    """Load iMotions data for a participant from the file system into memory."""
    imotions_data_dfs = {}
    for key, modality_config in IMOTIONS_DATA_CONFIG.items():
        file_name = (
            IMOTIONS_DATA_PATH
            / str(participant_id)
            / f"{modality_config['file_name']}.csv"
        )
        if not file_name.is_file():
            logger.error(f"File {file_name} does not exist.")
            continue
        df = _load_csv(
            file_name,
            load_columns=modality_config["load_columns"],
            rename_columns=modality_config.get("rename_columns"),
        )
        imotions_data_dfs[key] = df
    return imotions_data_dfs


def _load_csv(
    file_name: Path,
    load_columns: list[str],
    rename_columns: dict[str, str],
):
    start_index = _get_start_index(file_name)
    df = pl.read_csv(
        file_name,
        skip_rows=start_index,
        columns=load_columns,
        infer_schema_length=20000,  # FIXME TODO
    )

    if rename_columns:
        df = df.rename(rename_columns)
    return _preprocess_df(df)


def _get_start_index(file_name: str) -> int:
    """Output files from iMotions have a header that needs to be skipped."""
    with open(file_name, "r") as file:
        lines = file.readlines(2**16)  # only read a few lines
    file_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
    return file_start_index


def _preprocess_df(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase and remove spaces from column names."""
    df = df.select([pl.col(col).alias(col.lower()) for col in df.columns])
    df = df.select([pl.col(col).alias(col.replace(" ", "_")) for col in df.columns])
    return df


def create_trials_df(
    participant_id: str,
    iMotions_Marker: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a table with trial information (metadata for each measurement).
    """
    trials_df = (
        # 'markerdescription' from imotions was used exclusively for the stimulus seed
        # drop all rows where the markerdescription is null
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
    id_is_odd = int(participant_id) % 2
    skin_areas = list(range(1, 7)) if id_is_odd else list(range(6, 0, -1))
    trials_df = trials_df.with_columns(
        pl.Series(
            name="skin_area",
            values=skin_areas * (trials_df.height // len(skin_areas) + 1),  # repeat
        )
        .head(trials_df.height)
        .alias("skin_area")
        .cast(pl.UInt8)
    )
    # change order of columns
    trials_df = trials_df.select(
        pl.col(a := "trial_number"),
        pl.col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    return trials_df


def create_raw_data_dfs(
    imotions_data_dfs: dict[str, pl.DataFrame],
    trials_df: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """
    Create raw data tables for each data type based on the trial information.
    (Only keep rows that are part of a trial.)
    """
    # Create a DataFrame with the trial information only
    trial_info_df = trials_df.select(
        pl.col("timestamp_start").alias("trial_start"),
        pl.col("timestamp_end").alias("trial_end"),
        pl.col("trial_number"),
        pl.col("participant_id"),
    )
    raw_data_dfs = {}
    for key, config in imotions_data_dfs.items():
        if key == "iMotions_Marker":  # skip trial data (metadata)
            continue

        # Perform an asof join to assign trial numbers to each stimulus
        df = imotions_data_dfs[key]
        df = df.join_asof(
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
        # Change order of columns
        df = df.select(
            pl.col(a := "trial_number"),
            pl.col(b := "participant_id"),
            pl.all().exclude(a, b),
        )

        key = key.replace("iMotions_", "Raw_")
        raw_data_dfs[key] = df

    logger.info("Created raw data dataframes.")
    return raw_data_dfs


def main():
    # for debugging only
    imotions_data_dfs = load_imotions_data_dfs(1)
    trials_df = create_trials_df(1, imotions_data_dfs["iMotions_Marker"])
    raw_data_dfs = create_raw_data_dfs(imotions_data_dfs, trials_df)
    print(raw_data_dfs)


if __name__ == "__main__":
    main()
