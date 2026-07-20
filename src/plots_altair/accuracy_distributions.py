"""Altair seed-stability histogram panels."""

from collections.abc import Mapping, Sequence

import altair as alt
import numpy as np

from .style import BLUE, FONT, FONT_SIZE, _with_optional_title


def plot_accuracy_distributions(
    distributions: Mapping[str, Sequence[float]],
    reference_accuracies: Mapping[str, float],
    *,
    bin_width: float = 0.01,
    width: int = 280,
    height: int = 210,
    title: str | None = "Accuracy distributions",
) -> alt.HConcatChart:
    """Plot model-stability histograms with shared bins and axes."""
    if distributions.keys() != reference_accuracies.keys():
        raise ValueError(
            "Distributions and reference accuracies must have identical keys"
        )

    arrays = {
        label: np.asarray(values, dtype=float)
        for label, values in distributions.items()
    }
    all_values = np.concatenate(list(arrays.values()))
    step = 0.05
    x_min = float(np.floor(all_values.min() / step) * step)
    x_max = float(np.ceil(all_values.max() / step) * step)
    bins = np.arange(x_min, x_max + bin_width * 0.5, bin_width)
    if bins[-1] < x_max:
        bins = np.append(bins, x_max)
    histograms = {
        label: np.histogram(values, bins=bins) for label, values in arrays.items()
    }
    largest_count = max(int(counts.max()) for counts, _ in histograms.values())
    y_step = 5
    y_max = max(
        y_step,
        y_step * np.ceil(largest_count * 1.25 / y_step),
    )
    y_scale = alt.Scale(domain=[0, float(y_max)], nice=False)
    rule_top = min(largest_count * 1.05, y_max - y_step)
    rule_y = round(height * (1 - rule_top / y_max))

    panels = []
    for panel_index, label in enumerate(arrays):
        counts, edges = histograms[label]
        histogram_data = alt.Data(
            values=[
                {
                    "bin_start": float(start),
                    "bin_end": float(end),
                    "count": int(count),
                }
                for start, end, count in zip(edges[:-1], edges[1:], counts)
            ]
        )
        x = alt.X(
            "bin_start:Q",
            title="Test set accuracy",
            scale=alt.Scale(domain=[x_min, x_max], nice=False),
            axis=alt.Axis(
                format=".2f", tickCount=max(2, round((x_max - x_min) / step) + 1)
            ),
        )
        bars = (
            alt.Chart(histogram_data)
            .mark_bar(
                color=BLUE,
                opacity=0.92,
                orient="vertical",
                stroke="white",
                strokeWidth=0.6,
            )
            .encode(
                x=x,
                x2="bin_end:Q",
                y=alt.Y(
                    "count:Q",
                    title="Number of train-test splits" if panel_index == 0 else None,
                    scale=y_scale,
                ),
                y2=alt.Y2(datum=0),
                tooltip=[
                    alt.Tooltip("bin_start:Q", title="From", format=".2f"),
                    alt.Tooltip("bin_end:Q", title="To", format=".2f"),
                    alt.Tooltip("count:Q", title="Count"),
                ],
            )
        )
        reference = (
            alt.Chart(
                alt.Data(
                    values=[
                        {
                            "reference": float(reference_accuracies[label]),
                        }
                    ]
                )
            )
            .mark_rule(color="black", strokeDash=[6, 4], strokeWidth=1.5)
            .encode(
                x=alt.X("reference:Q", scale=alt.Scale(domain=[x_min, x_max])),
                y=alt.value(rule_y),
                y2=alt.value(height),
            )
        )
        subplot_label = (
            alt.Chart(alt.Data(values=[{"label": label}]))
            .mark_text(
                align="center",
                baseline="top",
                dy=5,
                font=FONT,
                fontSize=FONT_SIZE,
                fontWeight="normal",
            )
            .encode(
                x=alt.value(width / 2),
                y=alt.value(5),
                text="label:N",
            )
        )
        panel = alt.layer(bars, reference, subplot_label).properties(
            width=width,
            height=height,
        )
        panels.append(panel)

    chart = alt.hconcat(*panels, spacing=20).resolve_scale(x="shared", y="shared")
    return _with_optional_title(chart, title)
