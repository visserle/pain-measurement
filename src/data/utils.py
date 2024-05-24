from functools import reduce

import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.make_dataset import load_dataset


def merge_datasets(
    dfs: list[pl.DataFrame],
    merge_on: list[str] = ["Timestamp", "Trial", "Participant"],
    sort_by: list[str] = ["Timestamp"],
) -> pl.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame.

    The default merge_on and sort_by columns are for merging different modalities of
    one participant.

    The function can also be used to merge multiple participants' modalities with
    a different merge_on and sort_by column.

    Examples:

    Merge two datasets of different modalities of one participant:
    >>> dfs = load_participant_datasets(PARTICIPANT_LIST[0], INTERIM_LIST)
    >>> eda_plus_rating = merge_datasets([dfs.eda, dfs.stimulus])


    Merge multiple participants' modalities:
    ````python
    # The load function loads one modality for multiple participants
    stimuli = load_modality_data(PARTICIPANT_LIST, INTERIM_DICT["stimulus"])
    eda = load_modality_data(PARTICIPANT_LIST, INTERIM_DICT["eda"])
    multiple_eda_plus_rating = merge_datasets(
        [stimuli, eda],
        merge_on=["Timestamp", "Participant", "Trial"],
        sort_by=["Participant", "Trial", "Timestamp"],
    )
    # Normalzing, plotting, etc.
    features = ["Temperature", "Rating", "EDA_Tonic"]
    multiple_eda_plus_rating = interpolate(multiple_eda_plus_rating)
    multiple_eda_plus_rating = scale_min_max(
        multiple_eda_plus_rating, exclude_additional_columns=["Temperature", "Rating"]
    )
    multiple_eda_plus_rating.hvplot(
        x="Timestamp",
        y=features,
        groupby=["Participant", "Trial"],
        kind="line",
        width=800,
        height=400,
        ylim=(0, 1),
    )
    ````
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
