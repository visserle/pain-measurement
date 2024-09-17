import numpy as np
from bokeh.layouts import column
from bokeh.models import BoxAnnotation, ColumnDataSource, FixedTicker, HoverTool, Title
from bokeh.plotting import figure, show

from src.experiments.measurement.stimulus_generator import StimulusGenerator


def plot_stimulus_with_shapes(stimulus: StimulusGenerator):
    """
    Plot the stimulus data with shapes for the major decreasing intervals using Bokeh.
    Includes a hover tool for displaying point information.
    """
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate

    # Create a ColumnDataSource for the data
    source = ColumnDataSource(data=dict(time=time, temperature=stimulus.y))

    # Create a new plot
    p = figure(
        title=f"Seed: {stimulus.seed}",
        x_axis_label="Time (s)",
        y_axis_label="Temperature (°C)",
        width=900,
        height=300,
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )

    # Plot the main line
    p.line("time", "temperature", source=source, line_color="navy", line_width=2)

    # Add shapes for the major decreasing intervals
    for interval in stimulus.major_decreasing_intervals_idx:
        start_time, end_time = (
            interval[0] / stimulus.sample_rate,
            interval[1] / stimulus.sample_rate,
        )
        p.add_layout(
            BoxAnnotation(
                left=start_time,
                right=end_time,
                fill_color="salmon",
                fill_alpha=0.125,
            )
        )

    # Customize the plot
    p.xaxis.axis_label_text_font_style = "bold"
    p.yaxis.axis_label_text_font_style = "bold"
    p.xaxis.ticker = FixedTicker(ticks=list(range(0, int(max(time)) + 1, 10)))

    # Add hover tool
    hover = HoverTool(
        tooltips=[
            ("Time", "@time{0.1f} s"),
            ("Temperature", "@temperature{0.2f} °C"),
        ],
        mode="vline",
    )
    p.add_tools(hover)

    # Show the plot
    show(p)
