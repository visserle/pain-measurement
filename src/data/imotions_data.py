"""
Load iMotions data from the file system into memory.
"""

import logging
from pathlib import Path

import polars as pl

from src.data.data_config import DataConfig

IMOTIONS_DATA_CONFIG = DataConfig.load_imotions_config()
IMOTIONS_DATA_PATH = DataConfig.IMOTIONS_DATA_PATH


logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def load_imotions_data_df(
    participant_id: int,
    modality: str,
) -> pl.DataFrame:
    """Load iMotions data for a participant from the file system into memory."""
    modality_config = IMOTIONS_DATA_CONFIG[modality]
    file_name = (
        IMOTIONS_DATA_PATH / str(participant_id) / f"{modality_config['file_name']}.csv"
    )
    if not file_name.is_file():
        logger.error(f"File {file_name} does not exist.")
        return pl.DataFrame()
    return _load_csv(
        file_name,
        load_columns=modality_config["load_columns"],
        rename_columns=modality_config.get("rename_columns"),
    )


def _load_csv(
    file_name: Path,
    load_columns: list[str],
    rename_columns: dict[str, str],
):
    start_index = _get_start_index(str(file_name))
    df = pl.read_csv(
        file_name,
        skip_rows=start_index,
        columns=load_columns,
        infer_schema_length=20000,  # FIXME TODO
    )

    if rename_columns:
        df = df.rename(rename_columns)
    return _sanitize_df(df)


def _get_start_index(
    file_name: str,
) -> int:
    """Output files from iMotions have a header that needs to be skipped."""
    with open(file_name, "r") as file:
        lines = file.readlines(2**16)  # only read a few lines
    file_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
    return file_start_index


def _sanitize_df(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Sanitize the DataFrame by removing duplicate timestamps and renaming columns."""
    df = df.select([pl.col(col).alias(col.lower()) for col in df.columns])
    df = df.select([pl.col(col).alias(col.replace(" ", "_")) for col in df.columns])
    df = _remove_duplicate_timestamps(df)  # this affects shimmer output (eda, ppg)
    df = df.drop_nulls()  # this affects the rownumber in affectiva output (face)
    return df


def _remove_duplicate_timestamps(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Remove duplicate timestamps from the DataFrame.

    For instance, the Shimmer3 GSR+ unit collects 128 samples per second but with
    only 100 unique timestamps. This affects EDA and PPG data.
    """
    return df.unique("timestamp").sort("timestamp")
