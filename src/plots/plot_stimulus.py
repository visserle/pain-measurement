import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from bokeh.models import BoxAnnotation, ColumnDataSource, FixedTicker, HoverTool
from bokeh.plotting import figure, show

from src.experiments.measurement.stimulus_generator import StimulusGenerator

plt.style.use("./src/plots/style.mplstyle")


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
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "decreasing_intervals": "#E57373",
        "major_decreasing_intervals": "#FFAB91",
        "increasing_intervals": "#4CAF50",
        "strictly_increasing_intervals": "#66BB6A",
        "strictly_increasing_intervals_without_plateaus": "#A5D6A7",
        "plateau_intervals": "#1976D2",
        "prolonged_minima_intervals": "#64B5F6",
    }

    # Plot patches for each interval type
    num_types = len(stimulus.labels)
    # Reverse the order of labels for more intuitive plotting
    labels = stimulus.labels

    # Add strictly increasing intervals before plateaus for completeness
    labels["strictly_increasing_intervals"] = [
        stimulus._convert_interval(interval)
        for interval in stimulus.strictly_increasing_intervals_complete_idx
    ]

    labels = {k: v for k, v in reversed(labels.items())}

    for i, (interval_type, intervals) in enumerate(labels.items()):
        for start, end in intervals:
            ax.add_patch(
                patches.Rectangle(
                    (start, i - 0.4),
                    end - start,
                    0.8,
                    facecolor=colors.get(interval_type, "gray"),
                    edgecolor="none",
                    alpha=1,
                )
            )

    # Set axis limits and labels
    ax.set_xlim(0, stimulus.duration * 1000)
    ax.set_ylim(-0.5, num_types - 0.5)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_yticks(range(num_types))
    ax.set_yticklabels(
        [label.replace("_", " ").capitalize() for label in labels.keys()], fontsize=12
    )

    # Format x-axis
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{int(x / 1000)}" if x >= 1000 else f"{int(x)}")
    )

    # Use tight_layout with padding for better spacing
    plt.tight_layout(pad=2.0)

    plt.show()

    return fig


def plot_stimulus_matplotlib(
    stimulus,
    filename: str = "Stimulus_Plot.png",
    highlight_decreasing: bool = True,
    show_arrows: bool = True,
):
    """
    Plot the stimulus data using Matplotlib.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)

    # Get time data
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate

    # Plot the main line
    ax.plot(time, stimulus.y, color="navy", linewidth=2)

    # Add patches for the major decreasing intervals
    if highlight_decreasing:
        for interval in stimulus.major_decreasing_intervals_idx:
            start_time, end_time = (
                interval[0] / stimulus.sample_rate,
                interval[1] / stimulus.sample_rate,
            )
            ax.add_patch(
                patches.Rectangle(
                    (start_time, ax.get_ylim()[0]),
                    end_time - start_time,
                    ax.get_ylim()[1] - ax.get_ylim()[0],
                    facecolor="salmon",
                    alpha=0.125,
                    zorder=0,  # Place behind the line
                )
            )

    # Add red arrows at the start of major decreasing intervals
    if show_arrows:
        for interval in stimulus.major_decreasing_intervals_idx:
            start_idx = interval[0]
            start_time = start_idx / stimulus.sample_rate
            peak_temp = stimulus.y[start_idx]

            # Add arrow pointing diagonally down-right at the peak
            ax.annotate(
                "",
                xy=(
                    start_time + 3,
                    peak_temp - 0.05,
                ),  # Arrow tip (down-right from start)
                xytext=(
                    start_time - 3,
                    peak_temp + 0.05,
                ),  # Arrow start (up-left from tip)
                arrowprops=dict(
                    arrowstyle="->",
                    color="red",
                    lw=6,  # Thick line
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=20,  # Makes arrowhead bigger and less pixelated
                ),
                zorder=10,  # Place on top
            )

    # Set title and labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(
        "Temperature (°C)", labelpad=-60
    )  # Reduced padding to bring label closer

    # Improved y-axis handling
    y_min, y_max = ax.get_ylim()

    # Add some padding to y-limits for better visualization
    y_range = y_max - y_min
    y_min_padded = y_min - 0.05 * y_range
    y_max_padded = y_max + 0.05 * y_range
    ax.set_ylim(y_min_padded, y_max_padded)

    # Set meaningful y-axis ticks and labels
    ax.set_yticks([y_min, y_max])
    ax.set_yticklabels(
        ["Pain threshold", "VAS 70"],
        # fontsize=10,
    )

    # Add horizontal reference lines at the thresholds
    ax.axhline(y=y_min, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=0)
    ax.axhline(y=y_max, color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=0)

    # Customize the plot
    ax.set_xlim(0, 180)

    # Set x-axis ticks every 10 seconds
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    # Remove grid
    ax.grid(False)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Save the plot if a filename is provided
    if filename:
        plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()

    return fig, ax


def plot_stimulus_with_labels(
    stimulus: StimulusGenerator,
    filename: str = None,
):
    """
    Plot the stimulus temperature curve together with labeled intervals using two y-axes.
    """
    # Use a more professional style
    plt.style.use("seaborn-v0_8-paper")

    fig, ax1 = plt.subplots(figsize=(10, 4))  # Changed from 5 to 4

    # Create secondary y-axis for labels first
    ax2 = ax1.twinx()

    # More professional color palette
    colors = {
        "decreasing_intervals": "#d32f2f",
        "major_decreasing_intervals": "#ff6f60",
        "increasing_intervals": "#388e3c",
        "strictly_increasing_intervals": "#66bb6a",
        "strictly_increasing_intervals_without_plateaus": "#a5d6a7",
        "plateau_intervals": "#1976d2",
        "prolonged_minima_intervals": "#90caf9",
    }

    # Prepare labels with cleaner names
    label_names = {
        "decreasing_intervals": "Decreasing",
        "major_decreasing_intervals": "Major decreasing",
        "increasing_intervals": "Increasing",
        "strictly_increasing_intervals": "Strictly increasing",
        "strictly_increasing_intervals_without_plateaus": "Strictly increasing\n(no plateaus)",
        "plateau_intervals": "Plateau",
        "prolonged_minima_intervals": "Prolonged minima",
    }

    labels = stimulus.labels.copy()
    labels["strictly_increasing_intervals"] = [
        stimulus._convert_interval(interval)
        for interval in stimulus.strictly_increasing_intervals_complete_idx
    ]
    labels = {k: v for k, v in reversed(labels.items())}

    num_types = len(labels)

    # Plot patches for each interval type
    for i, (interval_type, intervals) in enumerate(labels.items()):
        for start, end in intervals:
            ax2.add_patch(
                patches.Rectangle(
                    (start / 1000, i - 0.3),
                    (end - start) / 1000,
                    0.6,
                    facecolor=colors.get(interval_type, "gray"),
                    edgecolor="none",
                    alpha=0.4,
                    zorder=1,
                )
            )

    # Configure secondary y-axis
    ax2.set_ylabel("Interval Type", fontsize=11, fontweight="normal")
    ax2.set_ylim(-0.5, num_types - 0.5)
    ax2.set_yticks(range(num_types))
    ax2.set_yticklabels([label_names.get(k, k) for k in labels.keys()], fontsize=9)
    ax2.tick_params(axis="y", which="both", length=0)

    # Plot temperature curve on primary y-axis
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate
    color_temp = "#000080"
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel(
        "Temperature (°C)", color=color_temp, fontsize=11, fontweight="normal"
    )
    ax1.plot(time, stimulus.y, color=color_temp, linewidth=1.8, zorder=10)
    ax1.tick_params(axis="y", labelcolor=color_temp)
    ax1.set_xlim(0, stimulus.duration)

    # Remove 0.25 and 0.75 ticks by setting specific ticks
    from matplotlib.ticker import MultipleLocator

    ax1.yaxis.set_major_locator(MultipleLocator(0.5))

    # Add subtle grid for better readability
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax1.set_axisbelow(True)

    # Lighten the spines for a cleaner look
    for spine in ax1.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(0.8)

    # Tighter layout
    plt.tight_layout()

    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=600, bbox_inches="tight", facecolor="white")

    plt.show()

    return fig, ax1, ax2
