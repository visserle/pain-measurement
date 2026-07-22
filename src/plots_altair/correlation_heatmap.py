"""Altair correlation-matrix panel."""

from collections.abc import Sequence

import altair as alt
import polars as pl

from .style import FONT_SIZE, SIGNAL_LABELS, _with_optional_title


def plot_correlation_heatmap(
    averages: pl.DataFrame,
    *,
    features: Sequence[str] | None = None,
    skip_first_n_seconds: float = 20,
    width: int = 250,
    height: int = 250,
    legend_title: str = "Correlation coefficient",
    title: str | None = "Pearson correlations",
) -> alt.LayerChart:
    """Plot the lower triangle of the Pearson correlation matrix."""
    if features is None:
        features = [
            "temperature",
            "pain_rating",
            "pupil_diameter",
            "heart_rate",
            "eda_tonic",
            "eda_phasic",
        ]

    filtered = averages.filter(
        pl.col("normalized_timestamp") >= skip_first_n_seconds * 1000
    )
    feature_columns = [f"mean_{feature}" for feature in features]
    missing = set(feature_columns).difference(filtered.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    correlations = filtered.select(feature_columns).corr().to_numpy()
    labels = [
        SIGNAL_LABELS.get(feature, feature.replace("_", " ").title())
        for feature in features
    ]
    rows = [
        {
            "x": labels[column],
            "y": labels[row],
            "correlation": correlations[row, column],
        }
        for row in range(len(labels))
        for column in range(row + 1)
    ]
    data = alt.Data(values=rows)
    x = alt.X(
        "x:N",
        sort=labels,
        title=None,
        axis=alt.Axis(labelAngle=-45, labelAlign="right", labelLimit=130),
    )
    y = alt.Y("y:N", sort=labels, title=None, axis=alt.Axis(labelLimit=130))

    cells = (
        alt.Chart(data)
        .mark_rect(stroke="white", strokeWidth=0.3)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "correlation:Q",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=["#e8eef7", "#0033cc"],
                    clamp=True,
                ),
                legend=alt.Legend(
                    title=legend_title,
                    orient="right",
                    titleOrient="right",
                    titlePadding=8,
                    titleLimit=height,
                    gradientLength=height,
                    gradientThickness=10,
                    values=[0, 0.2, 0.4, 0.6, 0.8, 1],
                    format=".1f",
                ),
            ),
            tooltip=[
                alt.Tooltip("y:N", title="Signal 1"),
                alt.Tooltip("x:N", title="Signal 2"),
                alt.Tooltip("correlation:Q", title="r", format=".3f"),
            ],
        )
    )

    values = (
        alt.Chart(data)
        .mark_text(fontSize=FONT_SIZE, baseline="middle")
        .encode(
            x=x,
            y=y,
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition(
                "datum.correlation >= 0.35",
                alt.value("white"),
                alt.value("black"),
            ),
        )
    )

    chart = (cells + values).properties(width=width, height=height)
    return _with_optional_title(chart, title)
