"""Altair timeline of automatically generated stimulus interval types."""

from collections.abc import Mapping

import altair as alt

from .style import _with_optional_title

INTERVAL_ORDER = [
    "decreasing_intervals",
    "major_decreasing_intervals",
    "increasing_intervals",
    "strictly_increasing_intervals",
    "strictly_increasing_intervals_without_plateaus",
    "plateau_intervals",
    "prolonged_minima_intervals",
]

INTERVAL_LABELS = {
    "decreasing_intervals": "Decreasing intervals",
    "major_decreasing_intervals": "Major decreasing intervals",
    "increasing_intervals": "Increasing intervals",
    "strictly_increasing_intervals": "Strictly increasing intervals",
    "strictly_increasing_intervals_without_plateaus": (
        "Strictly increasing intervals without plateaus"
    ),
    "plateau_intervals": "Plateau intervals",
    "prolonged_minima_intervals": "Prolonged minima intervals",
}

INTERVAL_COLORS = {
    "decreasing_intervals": "#e57373",
    "major_decreasing_intervals": "#ffab91",
    "increasing_intervals": "#4caf50",
    "strictly_increasing_intervals": "#66bb6a",
    "strictly_increasing_intervals_without_plateaus": "#a5d6a7",
    "plateau_intervals": "#1976d2",
    "prolonged_minima_intervals": "#64b5f6",
}


def plot_stimulus_intervals(
    stimulus,
    *,
    interval_order: list[str] | None = None,
    interval_labels: Mapping[str, str] | None = None,
    interval_colors: Mapping[str, str] | None = None,
    width: int = 900,
    height: int = 390,
    bar_size: int = 38,
    title: str | None = None,
) -> alt.Chart:
    """Plot each generated interval type on a shared time axis."""
    order = INTERVAL_ORDER if interval_order is None else interval_order
    labels = INTERVAL_LABELS if interval_labels is None else interval_labels
    colors = INTERVAL_COLORS if interval_colors is None else interval_colors

    intervals = dict(stimulus.labels)
    intervals["strictly_increasing_intervals"] = [
        stimulus._convert_interval(interval)
        for interval in stimulus.strictly_increasing_intervals_complete_idx
    ]

    rows = [
        {
            "interval_type": interval_type,
            "interval_label": labels.get(
                interval_type,
                interval_type.replace("_", " ").capitalize(),
            ),
            "start_s": float(start_ms) / 1000,
            "end_s": float(end_ms) / 1000,
        }
        for interval_type in order
        for start_ms, end_ms in intervals.get(interval_type, [])
    ]
    duration_s = len(stimulus.y) / float(stimulus.sample_rate)
    label_order = [
        labels.get(interval_type, interval_type.replace("_", " ").capitalize())
        for interval_type in order
    ]

    chart = (
        alt.Chart(alt.Data(values=rows))
        .mark_bar(size=bar_size)
        .encode(
            x=alt.X(
                "start_s:Q",
                title="Time (s)",
                scale=alt.Scale(domain=[0, duration_s], nice=False),
            ),
            x2="end_s:Q",
            y=alt.Y(
                "interval_label:N",
                title=None,
                sort=label_order,
                axis=alt.Axis(labelLimit=360, zindex=1),
            ),
            color=alt.Color(
                "interval_type:N",
                scale=alt.Scale(
                    domain=order,
                    range=[colors[interval_type] for interval_type in order],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("interval_label:N", title="Interval type"),
                alt.Tooltip("start_s:Q", title="Start (s)", format=".1f"),
                alt.Tooltip("end_s:Q", title="End (s)", format=".1f"),
            ],
        )
        .properties(width=width, height=height)
    )
    return _with_optional_title(chart, title)
