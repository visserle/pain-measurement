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
