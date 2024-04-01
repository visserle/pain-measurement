# TODO: add main functoion for figure generation for paper

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from src.experiments.measurement.stimulus_generator import StimulusGenerator


def plot_stimulus_extra(
    stimulus: StimulusGenerator,
    s_RoC: float,
    display_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, go.Figure]:
    """
    For plotly graphing of f(x), f'(x), and labels. Also displays the number and length of cooling segments.

    Parameters
    ----------
    stimulus : StimulusGenerator
        The stimulus object.
    s_RoC : float
        The rate of change threshold (°C/s) for alternative labels.
        For more information about thresholds, also see: http://www.scholarpedia.org/article/Thermal_touch#Thermal_thresholds
    display_stats : bool, optional
        If True, the number and length of cooling segments are displayed (default is True).

    Returns
    -------
    labels : array_like
        A binary array where 0 indicates cooling and 1 indicates heating.
    labels_alt : array_like
        A ternary array where 0 indicates cooling, 1 indicates heating, and 2 indicates a rate of change less than s_RoC.
    fig : plotly.graph_objects.Figure
        The plotly figure.
    """
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate
    # 0 for cooling, 1 for heating
    labels = (stimulus.y_dot >= 0).astype(int)
    # alternative: 0 for cooling, 1 for heating, 2 for RoC < s_RoC
    labels_alt = np.where(np.abs(stimulus.y_dot) > s_RoC, labels, 2)

    # Plot functions and labels
    fig = go.Figure()
    fig.update_layout(
        autosize=True, height=300, width=900, margin=dict(l=20, r=20, t=40, b=20)
    )
    fig.update_xaxes(title_text="Time (s)", tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(
        title_text=r"Temperature (°C) \ RoC (°C/s)",
        # range=[
        #     n:=min(stimulus.y) if np.sign(min(stimulus.y)) +1 else min(min(stimulus.y), -1),
        #     max(stimulus.y) if np.sign(n) + 1 else abs(n),
        # ]
    )

    func = [stimulus.y, stimulus.y_dot, labels, labels_alt]
    func_names = "f(x)", "f'(x)", "Label", "Label (alt)"
    colors = "royalblue", "skyblue", "springgreen", "violet"

    for idx, i in enumerate(func):
        visible = (
            "legendonly" if idx != 0 else True
        )  # only show the first function by default
        fig.add_scatter(
            x=time,
            y=i,
            name=func_names[idx],
            line=dict(color=colors[idx]),
            visible=visible,
        )
    fig.show()

    # Calculate the number and length of cooling segments from the alternative labels.
    # segment_change indicates where the label changes,
    # segment_number is the cumulative sum of segment_change
    df = pd.DataFrame({"label": labels_alt})
    df["segment_change"] = df["label"].ne(df["label"].shift())
    df["segment_number"] = df["segment_change"].cumsum()

    # group by segment_number and calculate the size of each group
    segment_sizes = df.groupby("segment_number").size()

    # filter the segments that correspond to the label 0
    label_0_segments = df.loc[df["label"] == 0, "segment_number"]
    label_0_sizes = segment_sizes.loc[label_0_segments.unique()]

    # calculate the number and length of segments in seconds
    if display_stats:
        print(
            f"Cooling segments [s] based on 'Label_alt' with a rate of change threshold of {s_RoC} (°C/s):\n"
        )
        print((label_0_sizes / stimulus.sample_rate).describe().apply("{:,.2f}".format))

    return labels, labels_alt, fig


def plot_all_seeds(config: dict):
    # Plot all seeds
    stimuli = []
    rows, cols = 4, 3
    fig, axes = plt.subplots(rows, cols, figsize=(9, 9), dpi=100)

    for ax, seed in zip(axes.flat, config["seeds"]):
        stimulus = StimulusGenerator(config, seed=seed, debug=0)
        stimuli.append(stimulus)

        ax.plot(stimulus.y)
        ax.set_title(f"Seed: {seed}")
        ax.axis(False)
    plt.show()
