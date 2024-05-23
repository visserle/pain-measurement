from functools import reduce

import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.make_dataset import load_dataset


def merge_datasets(
    dfs: list[pl.DataFrame],
    merge_on: list[str] = ["Timestamp", "Trial"],
    sort_by: list[str] = ["Timestamp"],
) -> pl.DataFrame:
    """
    Merge multiple DataFrames of one participant on the 'Timestamp' and 'Trial' columns.
    """
    if len(dfs) < 2:
        return dfs[0]

    df = reduce(
        lambda left, right: left.join(
            right,
            on=merge_on,
            how="outer_coalesce",
        ).sort(sort_by),
        dfs,
    )
    return df


def load_modality_data(
    participants: list[ParticipantConfig],
    modality: DataConfigBase,
) -> pl.DataFrame:
    """
    Load data from multiple participants for a specific modality. Used for exploratory
    data analysis.
    """

    dfs = []
    for participant in participants:
        df = load_dataset(participant, modality).df
        dfs.append(df)

    return pl.concat(dfs)
