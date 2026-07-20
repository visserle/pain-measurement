"""Altair small multiples for the experiment's generated temperature curves."""

import altair as alt
import polars as pl

from .style import FONT, FONT_SIZE, SIGNAL_COLORS, _with_optional_title


def plot_stimulus_seed_grid(
    stimuli: pl.DataFrame,
    *,
    columns: int = 3,
    width: int = 300,
    height: int = 105,
    line_color: str = SIGNAL_COLORS["temperature"],
    line_width: float = 2,
    panel_spacing: int = 12,
    header_font_size: int = FONT_SIZE,
    title: str | None = None,
) -> alt.FacetChart:
    """Plot one axis-free temperature curve for every random stimulus seed."""
    required_columns = {"seed", "time_s", "temperature"}
    missing = required_columns.difference(stimuli.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if columns <= 0:
        raise ValueError("columns must be positive")

    chart_data = (
        stimuli.select("seed", "time_s", "temperature")
        .sort(["seed", "time_s"])
        .with_columns(
            pl.concat_str(
                [pl.lit("Random seed: "), pl.col("seed").cast(pl.String)]
            ).alias("seed_label")
        )
    )
    seed_labels = (
        chart_data.select("seed", "seed_label")
        .unique()
        .sort("seed")
        .get_column("seed_label")
        .to_list()
    )

    chart = (
        alt.Chart(chart_data)
        .mark_line(color=line_color, strokeWidth=line_width)
        .encode(
            x=alt.X("time_s:Q", axis=None, scale=alt.Scale(zero=False)),
            y=alt.Y(
                "temperature:Q",
                axis=None,
                scale=alt.Scale(zero=False, nice=False),
            ),
            tooltip=[
                alt.Tooltip("seed:O", title="Random seed"),
                alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                alt.Tooltip(
                    "temperature:Q",
                    title="Temperature (°C)",
                    format=".2f",
                ),
            ],
        )
        .properties(width=width, height=height)
        .facet(
            facet=alt.Facet(
                "seed_label:N",
                sort=seed_labels,
                title=None,
                header=alt.Header(
                    labelFont=FONT,
                    labelFontSize=header_font_size,
                    labelFontWeight="normal",
                    labelPadding=6,
                ),
            ),
            columns=columns,
            spacing=panel_spacing,
        )
        .resolve_scale(x="shared", y="shared")
    )
    return _with_optional_title(chart, title)
