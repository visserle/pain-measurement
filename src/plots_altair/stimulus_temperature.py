"""Altair temperature-stimulus panel."""

import altair as alt
import numpy as np
import polars as pl

from .style import NAVY, _with_optional_title


def plot_stimulus_temperature(
    stimulus,
    *,
    width: int = 720,
    height: int = 250,
    title: str | None = "Temperature over time",
) -> alt.LayerChart:
    """Plot a generated temperature stimulus and its major decreases."""
    temperature = np.asarray(stimulus.y, dtype=float)
    time_s = np.arange(temperature.size, dtype=float) / float(stimulus.sample_rate)
    duration_s = temperature.size / float(stimulus.sample_rate)
    tick_values = np.arange(0, duration_s + 1e-9, 10, dtype=float).tolist()
    stimulus_data = pl.DataFrame({"time_s": time_s, "temperature": temperature})

    intervals = [
        {
            "start_s": start / float(stimulus.sample_rate),
            "end_s": end / float(stimulus.sample_rate),
        }
        for start, end in stimulus.major_decreasing_intervals_idx
    ]
    interval_data = alt.Data(values=intervals)

    threshold = float(temperature.min())
    vas_70 = float(temperature.max())
    padding = 0.05 * (vas_70 - threshold)
    y_domain = [threshold - padding, vas_70 + padding]
    y_axis = alt.Axis(
        values=[threshold, vas_70],
        labelExpr=(
            f"datum.value === {threshold!r} ? 'Pain threshold' : "
            f"datum.value === {vas_70!r} ? 'VAS 70' : ''"
        ),
        title="Temperature (°C)",
        titleX=-10,
        labelLimit=110,
    )
    x = alt.X(
        "time_s:Q",
        title="Time (s)",
        scale=alt.Scale(domain=[0, duration_s], nice=False),
        axis=alt.Axis(
            values=tick_values,
            labelOverlap=False,
            labelFlush=False,
        ),
    )

    highlights = (
        alt.Chart(interval_data)
        .mark_rect(color="salmon", opacity=0.12)
        .encode(
            x=alt.X(
                "start_s:Q",
                scale=alt.Scale(domain=[0, duration_s]),
                axis=alt.Axis(
                    values=tick_values,
                    labelOverlap=False,
                    labelFlush=False,
                ),
            ),
            x2="end_s:Q",
        )
    )
    threshold_rules = (
        alt.Chart(
            alt.Data(values=[{"temperature": threshold}, {"temperature": vas_70}])
        )
        .mark_rule(color="#7f7f7f", strokeDash=[6, 4], strokeWidth=1)
        .encode(y=alt.Y("temperature:Q", scale=alt.Scale(domain=y_domain)))
    )
    line = (
        alt.Chart(stimulus_data)
        .mark_line(color=NAVY, strokeWidth=2.2)
        .encode(
            x=x,
            y=alt.Y(
                "temperature:Q",
                title="Temperature (°C)",
                scale=alt.Scale(domain=y_domain, nice=False),
                axis=y_axis,
            ),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("temperature:Q", title="Temperature (°C)", format=".2f"),
            ],
        )
    )

    chart = (
        alt.layer(highlights, threshold_rules, line)
        .properties(width=width, height=height)
        .resolve_scale(x="shared", y="shared")
    )
    return _with_optional_title(chart, title)
