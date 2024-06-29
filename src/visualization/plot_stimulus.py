# TODO: add main function for figure generation for paper, etc.

import numpy as np
import plotly.graph_objects as go

from src.experiments.measurement.stimulus_generator import StimulusGenerator


def plot_stimulus_with_shapes(stimulus: StimulusGenerator):
    """
    Plot the stimulus data with shapes for the major decreasing intervals.
    """
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate

    # Plot functions and labels
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        height=300,
        width=900,
        margin=dict(l=20, r=20, t=40, b=20),
        title_text=f"Seed: {stimulus.seed}",
    )
    fig.update_xaxes(title_text="Time (s)", tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(
        title_text=r"Temperature (°C)",
    )

    fig.add_scatter(
        x=time,
        y=stimulus.y,
    )

    # Add shapes for the major increasing intervals
    for interval in stimulus.major_decreasing_intervals:
        start_time, end_time = (
            interval[0] / stimulus.sample_rate,
            interval[1] / stimulus.sample_rate,
        )
        fig.add_shape(
            type="rect",
            x0=start_time,
            y0=min(stimulus.y),
            x1=end_time,
            y1=max(stimulus.y),
            line=dict(width=0),
            fillcolor="salmon",
            opacity=0.125,
        )

    fig.show()
