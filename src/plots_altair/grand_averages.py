"""Altair small multiples of grand-averaged signals and confidence intervals."""

from collections.abc import Mapping, Sequence

import altair as alt
import polars as pl

from .style import SIGNAL_COLORS, SIGNAL_LABELS, _with_optional_title


def _grand_averages_long_data(
    averages: pl.DataFrame,
    signals: Sequence[str],
    signal_labels: Mapping[str, str],
    display_step_ms: int | None,
) -> pl.DataFrame:
    required_columns = {"stimulus_seed", "normalized_timestamp"}
    for signal in signals:
        required_columns.update(
            {
                f"mean_{signal}",
                f"ci_lower_{signal}",
                f"ci_upper_{signal}",
            }
        )
    missing = required_columns.difference(averages.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    data = averages.with_columns(
        pl.col("normalized_timestamp").round(0).cast(pl.Int64).alias("_timestamp_ms")
    )
    if display_step_ms is not None:
        if display_step_ms <= 0:
            raise ValueError("display_step_ms must be positive or None")
        data = data.filter(pl.col("_timestamp_ms") % display_step_ms == 0)

    frames = []
    for signal in signals:
        frames.append(
            data.select(
                "stimulus_seed",
                (pl.col("_timestamp_ms") / 1000).alias("time_s"),
                pl.lit(signal).alias("signal"),
                pl.lit(
                    signal_labels.get(
                        signal,
                        signal.replace("_", " ").title(),
                    )
                ).alias("signal_label"),
                pl.col(f"mean_{signal}").alias("mean"),
                pl.col(f"ci_lower_{signal}").alias("ci_lower"),
                pl.col(f"ci_upper_{signal}").alias("ci_upper"),
            )
        )
    return pl.concat(frames).sort(["stimulus_seed", "time_s", "signal"])


def plot_grand_averaged_signals(
    averages: pl.DataFrame,
    signals: Sequence[str],
    *,
    stimulus_seed: int,
    signal_labels: Mapping[str, str] | None = None,
    signal_colors: Mapping[str, str] | None = None,
    width: int = 720,
    height: int = 250,
    display_step_ms: int | None = 500,
    y_domain: tuple[float, float] = (-0.05, 1.05),
    line_width: float = 1.8,
    line_opacity: float = 0.95,
    show_ci: bool = True,
    ci_opacity: float = 0.15,
    legend_title: str | None = "Signal",
    legend_orient: str = "right",
    legend_columns: int = 1,
    title: str | None = None,
) -> alt.LayerChart:
    """Plot grand-averaged signals and confidence intervals for one stimulus."""
    if not signals:
        raise ValueError("signals must not be empty")
    if not 0 <= ci_opacity <= 1:
        raise ValueError("ci_opacity must be between 0 and 1")

    labels = SIGNAL_LABELS if signal_labels is None else signal_labels
    colors = SIGNAL_COLORS if signal_colors is None else signal_colors
    missing_colors = set(signals).difference(colors)
    if missing_colors:
        raise ValueError(f"Missing signal colors: {', '.join(sorted(missing_colors))}")

    data = _grand_averages_long_data(
        averages,
        signals,
        labels,
        display_step_ms,
    ).filter(pl.col("stimulus_seed") == stimulus_seed)
    if data.is_empty():
        raise ValueError(f"No data found for stimulus_seed={stimulus_seed}")

    label_order = [
        labels.get(signal, signal.replace("_", " ").title()) for signal in signals
    ]
    color = alt.Color(
        "signal_label:N",
        sort=label_order,
        scale=alt.Scale(
            domain=label_order,
            range=[colors[signal] for signal in signals],
        ),
        legend=alt.Legend(
            title=legend_title,
            orient=legend_orient,
            direction=(
                "vertical" if legend_orient in {"left", "right"} else "horizontal"
            ),
            columns=legend_columns,
            labelLimit=150,
            rowPadding=3,
            symbolType="stroke",
            symbolStrokeWidth=2.5,
            symbolSize=300,
            symbolOpacity=1,
        ),
    )
    max_time_s = float(data.get_column("time_s").max())
    x = alt.X(
        "time_s:Q",
        title="Time (s)",
        scale=alt.Scale(domain=[0, max_time_s], nice=False),
        axis=alt.Axis(values=[0, 30, 60, 90, 120, 150, 180]),
    )
    y = alt.Y(
        "mean:Q",
        title="Normalized value",
        scale=alt.Scale(domain=list(y_domain), nice=False),
        axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".2f"),
    )

    layers = []
    if show_ci:
        layers.append(
            alt.Chart(data)
            .mark_area(opacity=ci_opacity, clip=True)
            .encode(
                x=x,
                y=alt.Y(
                    "ci_lower:Q",
                    scale=alt.Scale(domain=list(y_domain), nice=False),
                ),
                y2="ci_upper:Q",
                color=color,
                detail="signal:N",
            )
        )
    layers.append(
        alt.Chart(data)
        .mark_line(
            strokeWidth=line_width,
            opacity=line_opacity,
            clip=True,
        )
        .encode(
            x=x,
            y=y,
            color=color,
            detail="signal:N",
            tooltip=[
                alt.Tooltip("signal_label:N", title="Signal"),
                alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                alt.Tooltip("ci_lower:Q", title="CI lower", format=".3f"),
                alt.Tooltip("ci_upper:Q", title="CI upper", format=".3f"),
            ],
        )
    )

    chart = alt.layer(*layers).properties(width=width, height=height)
    return _with_optional_title(chart, title)


def plot_grand_averages_grid(
    averages: pl.DataFrame,
    signals: Sequence[str],
    *,
    signal_labels: Mapping[str, str] | None = None,
    signal_colors: Mapping[str, str] | None = None,
    columns: int = 3,
    width: int = 330,
    height: int = 145,
    display_step_ms: int | None = 1000,
    y_domain: tuple[float, float] = (-0.05, 1.05),
    line_width: float = 1.5,
    line_opacity: float = 0.95,
    show_ci: bool = True,
    ci_opacity: float = 0.13,
    column_spacing: int = 18,
    row_spacing: int = 14,
    legend_columns: int | None = None,
    panel_border_color: str = "#606060",
    panel_border_width: float = 0.8,
    title: str | None = None,
) -> alt.VConcatChart:
    """Plot grand-averaged signals for each stimulus seed in a shared grid."""
    if not signals:
        raise ValueError("signals must not be empty")
    if columns <= 0:
        raise ValueError("columns must be positive")
    if not 0 <= ci_opacity <= 1:
        raise ValueError("ci_opacity must be between 0 and 1")
    if panel_border_width < 0:
        raise ValueError("panel_border_width must be non-negative")

    labels = SIGNAL_LABELS if signal_labels is None else signal_labels
    colors = SIGNAL_COLORS if signal_colors is None else signal_colors
    missing_colors = set(signals).difference(colors)
    if missing_colors:
        raise ValueError(f"Missing signal colors: {', '.join(sorted(missing_colors))}")

    long_data = _grand_averages_long_data(
        averages,
        signals,
        labels,
        display_step_ms,
    )
    seeds = long_data.get_column("stimulus_seed").unique().sort().to_list()
    signal_label_order = [
        labels.get(signal, signal.replace("_", " ").title()) for signal in signals
    ]
    color = alt.Color(
        "signal_label:N",
        sort=signal_label_order,
        scale=alt.Scale(
            domain=signal_label_order,
            range=[colors[signal] for signal in signals],
        ),
        legend=alt.Legend(
            title=None,
            orient="bottom",
            direction="horizontal",
            columns=(len(signals) if legend_columns is None else legend_columns),
            symbolType="stroke",
            symbolStrokeWidth=2.5,
            symbolSize=260,
            symbolOpacity=1,
        ),
    )
    nrows = (len(seeds) + columns - 1) // columns
    max_time_s = float(long_data.get_column("time_s").max())
    panels = []

    for panel_index, seed in enumerate(seeds):
        panel_row = panel_index // columns
        panel_column = panel_index % columns
        bottom_row = panel_row == nrows - 1
        left_column = panel_column == 0
        x_axis = (
            alt.Axis(
                values=[0, 50, 100, 150],
                title=(
                    "Time (s)" if bottom_row and panel_column == columns // 2 else None
                ),
            )
            if bottom_row
            else None
        )
        y_axis = (
            alt.Axis(
                values=[0, 0.25, 0.5, 0.75, 1],
                format=".2f",
                title=("Normalized value" if panel_row == (nrows - 1) // 2 else None),
            )
            if left_column
            else None
        )
        seed_data = long_data.filter(pl.col("stimulus_seed") == seed)
        base = alt.Chart(seed_data)
        layers = []
        if show_ci:
            layers.append(
                base.mark_area(opacity=ci_opacity, clip=True).encode(
                    x=alt.X(
                        "time_s:Q",
                        scale=alt.Scale(domain=[0, max_time_s], nice=False),
                        axis=x_axis,
                    ),
                    y=alt.Y(
                        "ci_lower:Q",
                        scale=alt.Scale(domain=list(y_domain), nice=False),
                        axis=y_axis,
                    ),
                    y2="ci_upper:Q",
                    color=color,
                )
            )
        layers.append(
            base.mark_line(
                strokeWidth=line_width,
                opacity=line_opacity,
                clip=True,
            ).encode(
                x=alt.X(
                    "time_s:Q",
                    scale=alt.Scale(domain=[0, max_time_s], nice=False),
                    axis=x_axis,
                ),
                y=alt.Y(
                    "mean:Q",
                    scale=alt.Scale(domain=list(y_domain), nice=False),
                    axis=y_axis,
                ),
                color=color,
                detail="signal:N",
                tooltip=[
                    alt.Tooltip("stimulus_seed:O", title="Stimulus seed"),
                    alt.Tooltip("signal_label:N", title="Signal"),
                    alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                    alt.Tooltip("mean:Q", title="Mean", format=".3f"),
                    alt.Tooltip("ci_lower:Q", title="CI lower", format=".3f"),
                    alt.Tooltip("ci_upper:Q", title="CI upper", format=".3f"),
                ],
            )
        )
        if panel_border_width > 0:
            frame_data = alt.Data(
                values=[
                    {
                        "x_min": 0,
                        "x_max": max_time_s,
                        "y_min": y_domain[0],
                        "y_max": y_domain[1],
                    }
                ]
            )
            layers.append(
                alt.Chart(frame_data)
                .mark_rect(
                    fillOpacity=0,
                    stroke=panel_border_color,
                    strokeWidth=panel_border_width,
                )
                .encode(
                    x=alt.X(
                        "x_min:Q",
                        scale=alt.Scale(domain=[0, max_time_s], nice=False),
                        axis=x_axis,
                    ),
                    x2="x_max:Q",
                    y=alt.Y(
                        "y_min:Q",
                        scale=alt.Scale(domain=list(y_domain), nice=False),
                        axis=y_axis,
                    ),
                    y2="y_max:Q",
                )
            )
        panels.append(alt.layer(*layers).properties(width=width, height=height))

    rows = [
        alt.hconcat(
            *panels[start : start + columns],
            spacing=column_spacing,
        ).resolve_scale(x="shared", y="shared", color="shared")
        for start in range(0, len(panels), columns)
    ]
    chart = alt.vconcat(*rows, spacing=row_spacing).resolve_scale(
        x="shared",
        y="shared",
        color="shared",
    )
    return _with_optional_title(chart, title)
