"""Altair participant-accuracy panel."""

from collections.abc import Mapping

import altair as alt
import numpy as np
import polars as pl

from src.plots.utils import FEATURE_LABELS

from .style import CHANCE_RED, PARTICIPANT_COLORS, _with_optional_title


def plot_participant_accuracies(
    results: Mapping[str, pl.DataFrame],
    *,
    width: int = 1320,
    height: int = 210,
    x_label_angle: float = -45,
    title: str | None = "Classification accuracy across models and participants",
) -> alt.LayerChart:
    """Plot per-model accuracy box plots and consistently offset participants."""
    feature_order = list(results)
    feature_labels = [
        FEATURE_LABELS.get(feature, feature.replace("_", " ").title())
        for feature in feature_order
    ]
    rows = []
    for feature, feature_label in zip(feature_order, feature_labels):
        participant_data = results[feature].filter(pl.col("participant") != "overall")
        rows.extend(
            {
                "feature": feature_label,
                "participant": str(row["participant"]),
                "accuracy": float(row["accuracy"]),
            }
            for row in participant_data.iter_rows(named=True)
        )

    participants = sorted(
        {row["participant"] for row in rows},
        key=lambda participant: int(participant),
    )
    feature_indices = dict(zip(feature_labels, range(len(feature_labels))))
    max_jitter = 16 * len(feature_labels) / width
    participant_jitter = dict(
        zip(
            participants,
            np.linspace(-max_jitter, max_jitter, len(participants)),
        )
    )
    for row in rows:
        feature_index = feature_indices[row["feature"]]
        row["feature_index"] = feature_index
        row["feature_position"] = feature_index + participant_jitter[row["participant"]]

    data = alt.Data(values=rows)
    x_axis = alt.Axis(
        values=list(range(len(feature_labels))),
        labelExpr=f"{feature_labels!r}[datum.value]",
        labelAngle=x_label_angle,
        labelAlign="right",
        labelBaseline="middle",
        labelLimit=180,
        labelOverlap=False,
        labelPadding=5,
    )
    x_scale = alt.Scale(
        domain=[-0.5, len(feature_labels) - 0.5],
        nice=False,
    )
    box_x = alt.X(
        "feature_index:Q",
        title=None,
        axis=x_axis,
        scale=x_scale,
    )
    point_x = alt.X(
        "feature_position:Q",
        title=None,
        axis=x_axis,
        scale=x_scale,
    )
    y = alt.Y(
        "accuracy:Q",
        title="Test set accuracy",
        scale=alt.Scale(domain=[0, 1], nice=False),
        axis=alt.Axis(
            format=".0%", tickCount=6, grid=True, gridColor="#dedede", gridOpacity=0.7
        ),
    )

    boxes = (
        alt.Chart(data)
        .mark_boxplot(
            extent=1.5,
            size=42,
            outliers=False,
            box={"fill": "white", "stroke": "black", "strokeWidth": 1.1},
            median={"color": "black", "strokeWidth": 1.5},
            rule={"color": "black", "strokeWidth": 1},
            ticks={"color": "black", "strokeWidth": 1},
        )
        .encode(x=box_x, y=y)
    )
    points = (
        alt.Chart(data)
        .mark_circle(size=55, opacity=0.9)
        .encode(
            x=point_x,
            y=y,
            color=alt.Color(
                "participant:N",
                sort=participants,
                scale=alt.Scale(
                    domain=participants,
                    range=PARTICIPANT_COLORS[: len(participants)],
                ),
                legend=alt.Legend(
                    title="Participant ID",
                    orient="right",
                    direction="vertical",
                    offset=8,
                    padding=0,
                    titlePadding=4,
                    rowPadding=2,
                    labelLimit=180,
                    symbolType="circle",
                    symbolSize=60,
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Model"),
                alt.Tooltip("participant:N", title="Participant ID"),
                alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
            ],
        )
    )
    chance = (
        alt.Chart(alt.Data(values=[{"accuracy": 0.5}]))
        .mark_rule(color=CHANCE_RED, strokeDash=[6, 4], opacity=0.7, strokeWidth=1.2)
        .encode(y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1])))
    )

    chart = alt.layer(chance, boxes, points).properties(width=width, height=height)
    return _with_optional_title(chart, title)
