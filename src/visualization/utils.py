import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from functools import wraps, reduce
import logging

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import neurokit2 as nk
import plotly.express as px
import hvplot.polars
import panel as pn
import holoviews as hv

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.process_data import load_dataset


def concat_participants_on_modality(participant_list: List[ParticipantConfig], data_config: DataConfigBase, modality: str) -> pl.DataFrame:
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
            load_dataset(participant, data_config[modality])
            .dataset
            .with_columns(pl.lit(participant.id)
            .alias("Participant"))
            )
    return pl.concat(dfs)