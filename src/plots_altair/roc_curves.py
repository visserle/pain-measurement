"""Altair ROC-curve panel."""

from collections.abc import Mapping

import altair as alt
import numpy as np
from sklearn.metrics import roc_curve

from src.plots.utils import FEATURE_LABELS

from .style import CHANCE_RED, MODEL_COLORS, _with_optional_title


def plot_roc_curves(
    results: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    width: int = 390,
    height: int = 270,
    title: str | None = "ROC curves (All models)",
) -> alt.LayerChart:
    """Plot ROC curves for multiple feature sets."""
    model_order = list(results)
    model_labels = [
        FEATURE_LABELS.get(model, model.replace("_", " ").title())
        for model in model_order
    ]
    rows = []
    for model, label in zip(model_order, model_labels):
        probabilities, y_true = results[model]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        rows.extend(
            {
                "model": label,
                "line_type": "Model",
                "false_positive_rate": float(fpr),
                "true_positive_rate": float(tpr),
            }
            for fpr, tpr in zip(false_positive_rate, true_positive_rate)
        )

    chance_label = "Random classifier"
    rows.extend(
        [
            {
                "model": chance_label,
                "line_type": chance_label,
                "false_positive_rate": 0.0,
                "true_positive_rate": 0.0,
            },
            {
                "model": chance_label,
                "line_type": chance_label,
                "false_positive_rate": 1.0,
                "true_positive_rate": 1.0,
            },
        ]
    )
    legend_order = [*model_labels, chance_label]

    x = alt.X(
        "false_positive_rate:Q",
        title="False positive rate",
        scale=alt.Scale(domain=[0, 1], nice=False),
        axis=alt.Axis(format=".1f", tickCount=6),
    )
    y = alt.Y(
        "true_positive_rate:Q",
        title="True positive rate",
        scale=alt.Scale(domain=[0, 1], nice=False),
        axis=alt.Axis(format=".1f", tickCount=6),
    )
    legend = alt.Legend(
        title="Feature set",
        orient="right",
        direction="vertical",
        offset=8,
        padding=0,
        titlePadding=4,
        rowPadding=2,
        labelLimit=180,
        symbolDash=alt.ExprRef(
            expr=f"datum.value === '{chance_label}' ? [5, 4] : [1, 0]"
        ),
        symbolStrokeWidth=2,
        symbolSize=400,
        # labelFontSize=12,  # default is 10, reduce as needed
    )
    curves = (
        alt.Chart(alt.Data(values=rows))
        .mark_line(strokeWidth=1.7)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "model:N",
                sort=legend_order,
                scale=alt.Scale(
                    domain=legend_order,
                    range=[*MODEL_COLORS[: len(model_labels)], CHANCE_RED],
                ),
                legend=legend,
            ),
            strokeDash=alt.StrokeDash(
                "line_type:N",
                scale=alt.Scale(
                    domain=["Model", chance_label],
                    range=[[1, 0], [5, 4]],
                ),
                legend=None,
            ),
            detail="model:N",
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(
                    "false_positive_rate:Q", title="False positive rate", format=".3f"
                ),
                alt.Tooltip(
                    "true_positive_rate:Q", title="True positive rate", format=".3f"
                ),
            ],
        )
    )

    chart = curves.properties(width=width, height=height)
    return _with_optional_title(chart, title)
