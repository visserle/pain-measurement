import os
from pathlib import Path
import logging

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px
import hvplot.polars
import panel as pn
import holoviews as hv

from src.log_config import configure_logging

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def plot_trial_matplotlib(df: pl.DataFrame, trial: int, features: list[str] = None):
    """
    To exclude a feature, simply use df.drop(feature)
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial','Participant').columns    
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for feature in features:
        ax.plot(
            df.filter(pl.col('Trial') == trial).select(pl.col('Timestamp'))/1000,
            df.filter(pl.col('Trial') == trial).select(feature),
            label=feature)
    ax.set_xlabel('Time (s)')
    ax.legend()
    plt.show()
    
    
def plot_trial_plotly(df: pl.DataFrame, trial: int, features: list[str] = None):
    """
    To exclude a feature, simply use df.drop(feature)
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial','Participant').columns    
    
    fig = px.line(
        df.filter(pl.col('Trial') == trial),
        x='Timestamp',
        y=features,
        title=f'Trial {trial}')
    fig.update_layout(
        autosize=True,
        height=500,
        width=1000,
        margin=dict(l=20, r=20, t=40, b=20))
    fig.show()


def plot_data_panel(df: pl.DataFrame, features: list[str] = None, groups: str = None):
    """
    Plots the data using hvPlot and Panel.
    To exclude a feature, simply use df.drop(feature)
    By default the plot will be grouped by Trial and (if available) Participant.
    
    To concat the data of several partcipants for one modality and add a participant column
    you can use concat_participants_on_modality from the plotting utils module.
    
    
    TODO: grouped legend? https://community.plotly.com/t/plots-with-grouped-legend/71864

    The following code would make life easier, but needs jupyter_bokeh to supports jupyter 4.0 first
    import hvplot
    hvplot.extension('plotly')
    eda.plot(x='Timestamp', y=['EDA_RAW', 'nk_EDA_Tonic', 'nk_EDA_Phasic'], groupby='Trial')
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial','Participant').columns
    
    if groups is None:
        groups = ['Trial','Participant'] if 'Participant' in df.columns else 'Trial'
    elif groups == "Trial":
        groups = "Trial"
        if 'Participant' in df.columns:
            logger.error("Plotting trials of several participants leads to duplicate timestamps.")
    elif groups == "Participant":
        groups = "Participant"

    # Explicitly tell hvPlot to use Plotly
    hv.extension('plotly')

    plot = df.hvplot(
        x='Timestamp',
        y=features,
        groupby=groups,
        backend='plotly',
        width=1400,
        height=600 
        )

    # Convert the plot to a Panel object
    panel = pn.panel(plot)

    # Serve the Panel object
    panel.show(port=np.random.randint(10000, 60000))
