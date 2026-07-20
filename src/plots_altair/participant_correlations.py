"""Altair participant-level correlation estimates with standard deviations."""

from collections.abc import Mapping, Sequence

import altair as alt
import polars as pl

from .style import SIGNAL_COLORS, SIGNAL_LABELS, _with_optional_title


def plot_participant_correlations(
    stats: pl.DataFrame,
    targets: Sequence[str],
    *,
    reference: str = "temperature",
    signal_labels: Mapping[str, str] | None = None,
    signal_colors: Sequence[str] | None = None,
    width: int = 1200,
    height: int = 440,
    y_domain: tuple[float, float] = (-0.4, 1.0),
    point_size: int = 58,
    point_opacity: float = 0.72,
    error_bar_width: float = 1.3,
    cap_size: int = 8,
    zero_line_opacity: float = 0.35,
    participant_group_padding: float = 0.45,
    vertical_grid_opacity: float = 0.35,
    legend_columns: int | None = None,
    title: str | None = None,
) -> alt.LayerChart:
    """Plot participant means and ±SD for correlations with a reference signal."""
    if not targets:
        raise ValueError("targets must not be empty")
    colors = (
        [SIGNAL_COLORS[target] for target in targets]
        if signal_colors is None
        else list(signal_colors)
    )
    if len(colors) < len(targets):
        raise ValueError("signal_colors must contain at least one color per target")
    if not 0 <= participant_group_padding < 1:
        raise ValueError("participant_group_padding must be between 0 and 1")
    if not 0 <= vertical_grid_opacity <= 1:
        raise ValueError("vertical_grid_opacity must be between 0 and 1")

    required_columns = {"participant_id"}
    for target in targets:
        required_columns.update({f"{target}_mean", f"{target}_std"})
    missing = required_columns.difference(stats.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    labels = SIGNAL_LABELS if signal_labels is None else signal_labels
    label_order = [
        labels.get(target, target.replace("_", " ").title()) for target in targets
    ]
    participant_ids = stats.get_column("participant_id").sort().to_list()
    participant_order = [str(participant_id) for participant_id in participant_ids]
    rows = []
    endpoint_rows = []
    for row in stats.sort("participant_id").iter_rows(named=True):
        for target in targets:
            mean = row[f"{target}_mean"]
            standard_deviation = row[f"{target}_std"]
            if mean is None or standard_deviation is None:
                continue
            label = labels.get(target, target.replace("_", " ").title())
            lower = float(mean) - float(standard_deviation)
            upper = float(mean) + float(standard_deviation)
            values = {
                "participant": str(row["participant_id"]),
                "target": target,
                "target_label": label,
                "mean": float(mean),
                "standard_deviation": float(standard_deviation),
                "lower": lower,
                "upper": upper,
            }
            rows.append(values)
            endpoint_rows.extend(
                [
                    {**values, "endpoint": lower},
                    {**values, "endpoint": upper},
                ]
            )

    data = alt.Data(values=rows)
    endpoint_data = alt.Data(values=endpoint_rows)
    x = alt.X(
        "participant:N",
        sort=participant_order,
        title="Participant ID",
        scale=alt.Scale(
            paddingInner=participant_group_padding,
            paddingOuter=0.15,
        ),
        axis=alt.Axis(
            labelAngle=0,
            labelOverlap=False,
            grid=True,
            gridColor="#d9d9d9",
            gridOpacity=vertical_grid_opacity,
            gridWidth=0.8,
        ),
    )
    x_offset = alt.XOffset(
        "target_label:N",
        sort=label_order,
        scale=alt.Scale(domain=label_order),
    )
    y_scale = alt.Scale(domain=list(y_domain), nice=False)
    color = alt.Color(
        "target_label:N",
        sort=label_order,
        scale=alt.Scale(
            domain=label_order,
            range=colors[: len(targets)],
        ),
        legend=alt.Legend(
            title=None,
            orient="bottom",
            direction="horizontal",
            columns=(len(targets) if legend_columns is None else legend_columns),
            symbolType="circle",
            symbolSize=80,
        ),
    )
    reference_label = labels.get(
        reference,
        reference.replace("_", " ").title(),
    )

    zero_line = (
        alt.Chart(alt.Data(values=[{"zero": 0}]))
        .mark_rule(color="black", opacity=zero_line_opacity, strokeWidth=1)
        .encode(y=alt.Y("zero:Q", scale=y_scale))
    )
    error_bars = (
        alt.Chart(data)
        .mark_rule(strokeWidth=error_bar_width, opacity=point_opacity)
        .encode(
            x=x,
            xOffset=x_offset,
            y=alt.Y(
                "lower:Q",
                title=f"Mean correlation with {reference_label} (±SD)",
                scale=y_scale,
                axis=alt.Axis(grid=True, gridColor="#dedede", gridOpacity=0.65),
            ),
            y2="upper:Q",
            color=color,
        )
    )
    caps = (
        alt.Chart(endpoint_data)
        .mark_tick(
            orient="horizontal",
            size=cap_size,
            thickness=error_bar_width,
            opacity=point_opacity,
        )
        .encode(
            x=x,
            xOffset=x_offset,
            y=alt.Y("endpoint:Q", scale=y_scale),
            color=color,
        )
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=point_size, opacity=point_opacity)
        .encode(
            x=x,
            xOffset=x_offset,
            y=alt.Y("mean:Q", scale=y_scale),
            color=color,
            tooltip=[
                alt.Tooltip("participant:N", title="Participant ID"),
                alt.Tooltip("target_label:N", title="Signal"),
                alt.Tooltip("mean:Q", title="Mean correlation", format=".3f"),
                alt.Tooltip(
                    "standard_deviation:Q",
                    title="Standard deviation",
                    format=".3f",
                ),
            ],
        )
    )

    chart = alt.layer(zero_line, error_bars, caps, points).properties(
        width=width,
        height=height,
    )
    return _with_optional_title(chart, title)
