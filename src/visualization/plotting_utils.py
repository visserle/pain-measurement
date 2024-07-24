import logging
import os
from dataclasses import dataclass
from functools import reduce, wraps
from pathlib import Path
from typing import Dict, List

import holoviews as hv
import hvplot.polars
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.make_dataset import load_dataset


def add_nan_to_trials_for_hvplot(
    df: pl.DataFrame,
    time_column: str,
):
    """
    Add NaN separators to a DataFrame for each trial to prepare the data for plotting.

    See: https://hvplot.holoviz.org/user_guide/Large_Timeseries.html#multiple-lines-per-category-example
    """

    info_columns = [
        "trial_id",
        "participant_id",
        "stimulus_seed",
        "trial_specific_interval_id",
        "continuous_interval_id",
    ]

    # Create a new DataFrame with NaN rows
    group_by_columns = [column for column in df.columns if column in info_columns]

    nan_df = df.group_by(group_by_columns).agg(
        [
            pl.lit(float("nan")).alias(col)
            for col in df.columns
            if col not in info_columns
        ]
    )

    # Add a temporary column for sorting
    df = df.with_columns(pl.lit(0).alias("temp_sort"))
    nan_df = nan_df.with_columns(pl.lit(1).alias("temp_sort"))

    # Concatenate the original DataFrame with the NaN DataFrame
    result = pl.concat(
        [df, nan_df],
        how="diagonal",
        rechunk=True,
    ).sort(["continuous_interval_id", "temp_sort", time_column])

    # Remove the temporary sorting column and return the result
    return result.drop("temp_sort")


def concat_participants_on_modality(
    participant_list: List[ParticipantConfig],
    data_config: DataConfigBase,
    modality: str,
) -> pl.DataFrame:
    """
    Concatenate all participants on a given modality.
    Used for plotting data.

    Args:
        participant_list (List[ParticipantConfig]): List of participants to concatenate.
        modality (str): Modality to concatenate.

    Returns:
        pl.DataFrame: Concatenated dataframe.
    """
    dfs = []
    for participant in participant_list:
        dfs.append(
            load_dataset(participant, data_config[modality]).df.with_columns(
                pl.lit(participant.id).alias("Participant")
            )
        )
    return pl.concat(dfs)
