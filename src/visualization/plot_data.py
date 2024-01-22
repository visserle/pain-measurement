import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px
import hvplot.polars
import panel as pn
import holoviews as hv


def plot_trial_matplotlib(df: pl.DataFrame, trial: int, features: List[str] = None):
    """
    To exclude a feature, simply use df.drop(feature)
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial').columns    
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for feature in features:
        ax.plot(
            df.filter(pl.col('Trial') == trial).select(pl.col('Timestamp'))/1000,
            df.filter(pl.col('Trial') == trial).select(feature),
            label=feature)
    ax.set_xlabel('Time (s)')
    ax.legend()
    plt.show()
    
    
def plot_trial_plotly(df: pl.DataFrame, trial: int, features: List[str] = None):
    """
    To exclude a feature, simply use df.drop(feature)
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial').columns    
    
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


def plot_data_panel(df: pl.DataFrame, features: List[str] = None):
    """
    To exclude a feature, simply use df.drop(feature)
    
    TODO: grouped legend? https://community.plotly.com/t/plots-with-grouped-legend/71864

    The following code would make life easier, but needs jupyter_bokeh to supports jupyter 4.0 first
    import hvplot
    hvplot.extension('plotly')
    eda.plot(x='Timestamp', y=['EDA_RAW', 'nk_EDA_Tonic', 'nk_EDA_Phasic'], groupby='Trial')
    """
    if features is None:
        features = df.drop('Timestamp','Time','Trial').columns

    # Explicitly tell hvPlot to use Plotly
    hv.extension('plotly')

    # maybe try in bokeh
    plot = df.hvplot(
        x='Timestamp',
        y=features,
        groupby='Trial',
        backend='plotly',
        width=1400,
        height=600 
        )

    # Convert the plot to a Panel object
    panel = pn.panel(plot)

    # Serve the Panel object
    panel.show(port=np.random.randint(10000, 60000))
