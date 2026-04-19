import os
from pathlib import Path

import altair as alt
import holoviews as hv
import hvplot.polars  # noqa  # noqa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import tomllib
from dotenv import load_dotenv
from polars import col

from src.data.database_manager import DatabaseManager
from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.features.scaling import scale_min_max
from src.features.utils import to_describe
from src.log_config import configure_logging
from src.plots.utils import calculate_z_score

load_dotenv()
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))

configure_logging(
    ignore_libs=("Comm", "bokeh", "tornado", "matplotlib"),
)

pl.Config.set_tbl_rows(12)  # for 12 seeds
hv.output(widget_location="bottom", size=150)


def plot_stimulus_with_physiological_signals(
    ci_values, signals, signal_labels=None, signal_colors=None
):
    """
    Plot mean signals with confidence interval bands using Altair.

    Parameters
    ----------
    ci_values : pd.DataFrame
        DataFrame with columns: normalized_timestamp, mean_{sig}, ci_lower_{sig}, ci_upper_{sig}
    signals : list[str]
        List of signal keys to plot (must be keys in signal_labels).
    signal_labels : dict, optional
        Mapping from signal key to display label. Defaults to SIGNAL_LABELS.
    signal_colors : list[str], optional
        List of hex colors, one per signal. Defaults to SIGNAL_COLORS.
    """
    SIGNAL_LABELS = signal_labels or {
        "temperature": "Temperature",
        "pain_rating": "Pain rating",
        "pupil_diameter": "Pupil diameter",
        "eda_tonic": "Tonic EDA",
        "eda_phasic": "Phasic EDA",
        "heart_rate": "Heart rate",
        "mouth_open": "Mouth open",
    }

    SIGNAL_COLORS = signal_colors or [
        "#17396b",  # Temperature - dark navy
        "#d45f3c",  # Pain Rating - orange-red
        "#e8a020",  # Pupil Diameter - golden yellow
        "#5a7a2e",  # Tonic EDA - olive green
        "#3a9e5f",  # Phasic EDA - medium green
        "#3ab8b8",  # Heart Rate - teal
        "#7b4fa0",  # Mouth Open - purple
    ]

    layers = []
    for sig, color in zip(signals, SIGNAL_COLORS):
        label = SIGNAL_LABELS[sig]
        base = alt.Chart(ci_values).transform_calculate(label=f"'{label}'")

        line = base.mark_line(color=color).encode(
            x=alt.X("normalized_timestamp:Q", title="Time (ms)"),
            y=alt.Y(f"mean_{sig}:Q", title="Normalized Value"),
            tooltip=[
                alt.Tooltip("normalized_timestamp:Q", title="Time (ms)"),
                alt.Tooltip(f"mean_{sig}:Q", title=label, format=".2f"),
            ],
        )

        band = base.mark_area(opacity=0.2, color=color).encode(
            x=alt.X("normalized_timestamp:Q"),
            y=alt.Y(f"ci_lower_{sig}:Q"),
            y2=alt.Y2(f"ci_upper_{sig}:Q"),
        )

        layers.append(line + band)

    legend_data = (
        alt.Chart(
            alt.Data(
                values=[
                    {"signal": SIGNAL_LABELS[sig], "color": c}
                    for sig, c in zip(signals, SIGNAL_COLORS)
                ]
            )
        )
        .mark_line()
        .encode(
            color=alt.Color(
                "signal:N",
                scale=alt.Scale(
                    domain=[SIGNAL_LABELS[sig] for sig in signals],
                    range=SIGNAL_COLORS[: len(signals)],
                ),
                legend=alt.Legend(
                    title="",
                    orient="right",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    symbolSize=400,
                    labelFontSize=12,
                    rowPadding=4,
                ),
            )
        )
    )

    return (
        alt.layer(*layers, legend_data)
        .properties(width=800, height=400)
        .configure_axis(
            grid=False,
            gridColor="#dddddd",
            gridOpacity=0.8,
            titleFontWeight="normal",
            labelFontWeight="normal",
        )
        .configure_view(stroke=None, fill="white")
    )


def plot_stimulus_with_labels(
    stimulus,
    filename=None,
    width=1150,
    height=430,
):
    colors = {
        "decreasing_intervals": "#d32f2f",
        "major_decreasing_intervals": "#ff6f60",
        "increasing_intervals": "#388e3c",
        "strictly_increasing_intervals": "#66bb6a",
        "strictly_increasing_intervals_without_plateaus": "#a5d6a7",
        "plateau_intervals": "#1976d2",
        "prolonged_minima_intervals": "#90caf9",
    }

    label_names = {
        "decreasing_intervals": "Decreasing",
        "major_decreasing_intervals": "Major decreasing",
        "increasing_intervals": "Increasing",
        "strictly_increasing_intervals": "Strictly increasing",
        "strictly_increasing_intervals_without_plateaus": "Strictly increasing\n(no plateaus)",
        "plateau_intervals": "Plateau",
        "prolonged_minima_intervals": "Prolonged minima",
    }

    # Keep an explicit order so lane positions and axis labels are stable.
    order = [
        "decreasing_intervals",
        "major_decreasing_intervals",
        "increasing_intervals",
        "strictly_increasing_intervals",
        "strictly_increasing_intervals_without_plateaus",
        "plateau_intervals",
        "prolonged_minima_intervals",
    ]

    labels = dict(stimulus.labels)
    labels["strictly_increasing_intervals"] = [
        stimulus._convert_interval(interval)
        for interval in stimulus.strictly_increasing_intervals_complete_idx
    ]
    labels = {k: labels.get(k, []) for k in order}

    temp = np.asarray(stimulus.y, dtype=float)
    sample_rate = float(stimulus.sample_rate)
    time_s = np.arange(temp.size, dtype=float) / sample_rate
    duration = temp.size / sample_rate

    interval_rows = []
    for interval_type in order:
        interval_label = label_names[interval_type]
        for start_ms, end_ms in labels[interval_type]:
            # Snap boundaries to sample indices so bars align with the plotted signal grid.
            start_idx = int(np.round((start_ms / 1000.0) * sample_rate))
            end_idx = int(np.round((end_ms / 1000.0) * sample_rate))

            start_idx = int(np.clip(start_idx, 0, temp.size))
            end_idx = int(np.clip(end_idx, 0, temp.size))
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, temp.size)
            if end_idx <= start_idx:
                continue

            interval_rows.append(
                {
                    "interval_type": interval_type,
                    "interval_label": interval_label,
                    "start_s": start_idx / sample_rate,
                    "end_s": end_idx / sample_rate,
                }
            )

    interval_df = pl.DataFrame(
        {
            "interval_type": [r["interval_type"] for r in interval_rows],
            "interval_label": [r["interval_label"] for r in interval_rows],
            "start_s": [r["start_s"] for r in interval_rows],
            "end_s": [r["end_s"] for r in interval_rows],
        },
        schema={
            "interval_type": pl.String,
            "interval_label": pl.String,
            "start_s": pl.Float64,
            "end_s": pl.Float64,
        },
    )
    temp_df = pl.DataFrame({"time_s": time_s, "temperature": temp})

    interval_order = [label_names[k] for k in order]
    interval_axis_df = pl.DataFrame({"interval_label": interval_order})

    tick_step = 10 if duration <= 120 else 20
    x_ticks = np.arange(0, int(np.floor(duration)) + 1, tick_step).tolist()
    if not x_ticks:
        x_ticks = [0]
    if x_ticks[-1] != int(np.floor(duration)):
        x_ticks.append(int(np.floor(duration)))

    t_min = float(temp.min())
    t_max = float(temp.max())
    t_span = max(t_max - t_min, 0.25)

    x_scale = alt.Scale(domain=[0, duration])
    x_axis = alt.Axis(values=x_ticks, title="Time (s)")
    y_temp_scale = alt.Scale(domain=[t_min - 0.05 * t_span, t_max + 0.05 * t_span])

    interval_y = alt.Y(
        "interval_label:N",
        sort=interval_order,
        scale=alt.Scale(domain=interval_order),
        axis=alt.Axis(
            orient="right",
            title="Interval Type",
            titleAngle=90,
            titlePadding=16,
            ticks=True,
            tickSize=6,
            labelAngle=30,
            labelAlign="left",
            labelBaseline="middle",
        ),
    )

    bands = (
        alt.Chart(interval_df)
        .mark_bar(opacity=0.35, size=26)
        .encode(
            x=alt.X("start_s:Q", scale=x_scale, axis=x_axis),
            x2="end_s:Q",
            y=interval_y,
            color=alt.Color(
                "interval_type:N",
                scale=alt.Scale(
                    domain=order,
                    range=[colors[k] for k in order],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("interval_label:N", title="Type"),
                alt.Tooltip("start_s:Q", title="Start (s)", format=".2f"),
                alt.Tooltip("end_s:Q", title="End (s)", format=".2f"),
            ],
        )
    )

    # Invisible points keep all interval labels on the right axis even if one type has no spans.
    interval_axis = (
        alt.Chart(interval_axis_df)
        .mark_point(opacity=0)
        .encode(
            x=alt.value(0),
            y=interval_y,
        )
    )

    line = (
        alt.Chart(temp_df)
        .mark_line(color="#000080", strokeWidth=4)
        .encode(
            x=alt.X("time_s:Q", scale=x_scale, axis=x_axis),
            y=alt.Y(
                "temperature:Q",
                title="Temperature (deg C)",
                scale=y_temp_scale,
                axis=alt.Axis(tickMinStep=0.5),
            ),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("temperature:Q", title="Temperature (deg C)", format=".3f"),
            ],
        )
    )

    chart = (
        alt.layer(bands, interval_axis, line)
        .resolve_scale(y="independent")
        .properties(width=width, height=height)
        .configure_axis(
            grid=False,
            gridColor="#d9d9d9",
            gridDash=[4, 4],
            gridOpacity=0.55,
            labelFontSize=12,
            titleFontSize=14,
            titleFontWeight="normal",
            labelFontWeight="normal",
        )
        .configure_view(stroke="#b7b7b7")
    )

    if filename:
        chart.save(filename)

    return chart


def plot_stimulus_seed_grid(
    stimuli: pl.DataFrame,
    columns: int = 3,
    width: int = 220,
    height: int = 130,
) -> alt.Chart:
    """Plot each stimulus seed as a faceted Altair line chart using Polars data."""
    required_columns = {"seed", "time", "y"}
    missing_columns = required_columns.difference(stimuli.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    chart_data = (
        stimuli.select("seed", "time", "y")
        .sort(["seed", "time"])
        .with_columns(
            pl.concat_str(
                [pl.lit("Random seed: "), pl.col("seed").cast(pl.String)]
            ).alias("seed_label")
        )
    )

    base = (
        alt.Chart(chart_data)
        .mark_line(color="#1f77b4", strokeWidth=2)
        .encode(
            x=alt.X(
                "time:Q",
                axis=None,
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "y:Q",
                axis=None,
                scale=alt.Scale(zero=False, nice=False),
            ),
            tooltip=[
                alt.Tooltip("seed:Q", title="Seed"),
                alt.Tooltip("time:Q", title="Time", format=".0f"),
                alt.Tooltip("y:Q", title="Temperature", format=".3f"),
            ],
        )
        .properties(width=width, height=height)
    )

    return (
        base.facet(
            facet=alt.Facet("seed_label:N", title=None),
            columns=columns,
        )
        .resolve_scale(x="shared", y="independent")
        .configure_view(stroke=None)
        .configure_facet(spacing=10)
        .configure_header(
            labelFontSize=14,
            labelFontWeight="normal",
            labelOrient="top",
            labelPadding=6,
            title=None,
        )
    )
