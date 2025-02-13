import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models import BoxAnnotation, ColumnDataSource, FixedTicker, HoverTool
from bokeh.plotting import figure, show

from src.experiments.measurement.stimulus_generator import StimulusGenerator


def plot_stimulus(
    stimulus: StimulusGenerator,
    highlight_decreasing: bool = True,
):
    """
    Plot the stimulus data with shapes for the major decreasing intervals using Bokeh.
    Includes a hover tool for displaying point information.
    """
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate

    # Create a ColumnDataSource for the data
    source = ColumnDataSource(data=dict(time=time, temperature=stimulus.y))

    # Create a new plot
    plot = figure(
        title=f"Seed: {stimulus.seed}",
        x_axis_label="Time (s)",
        y_axis_label="Temperature (°C)",
        width=900,
        height=300,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Plot the main line
    plot.line("time", "temperature", source=source, line_color="navy", line_width=2)

    if highlight_decreasing:
        # Add shapes for the major decreasing intervals
        for interval in stimulus.major_decreasing_intervals_idx:
            start_time, end_time = (
                interval[0] / stimulus.sample_rate,
                interval[1] / stimulus.sample_rate,
            )
            plot.add_layout(
                BoxAnnotation(
                    left=start_time,
                    right=end_time,
                    fill_color="salmon",
                    fill_alpha=0.125,
                )
            )

    # Customize the plot
    plot.xaxis.axis_label_text_font_style = "bold"
    plot.yaxis.axis_label_text_font_style = "bold"
    plot.xaxis.ticker = FixedTicker(ticks=list(range(0, int(max(time)) + 2, 10)))

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Time", "@time{0.1f} s"),
            ("Temperature", "@temperature{0.2f} °C"),
        ],
        mode="vline",
    )
    plot.add_tools(hover)

    # Show the plot
    show(plot)
    return plot


def plot_stimulus_labels(
    stimulus: StimulusGenerator,
):
    """
    Plot labeled intervals using matplotlib patches.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 3))

    # Define colors for each type of interval
    colors = dict(
        zip(
            stimulus.labels.keys(),
            [
                "lightgreen",
                "green",
                "red",
                "yellow",
                "blue",
                "purple",
                "orange",
                "pink",
                "brown",
                "gray",
            ],
        )
    )

    # Plot patches for each interval type
    num_types = len(stimulus.labels)
    # Reverse the order of labels for more intuitive plotting
    labels = stimulus.labels
    labels = {k: v for k, v in reversed(labels.items())}
    for i, (interval_type, intervals) in enumerate(labels.items()):
        for start, end in intervals:
            ax.add_patch(
                patches.Rectangle(
                    (start, i - 0.4),
                    end - start,
                    0.8,
                    facecolor=colors[interval_type],
                    edgecolor="none",
                    alpha=0.7,
                )
            )

    # Set axis limits and labels
    ax.set_xlim(0, stimulus.duration * 1000)
    ax.set_ylim(-0.5, num_types - 0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks(range(num_types))
    ax.set_yticklabels(labels.keys())

    # Set title
    plt.title("Interval Analysis")

    # Show plot
    plt.tight_layout()
    plt.show()

    return fig
