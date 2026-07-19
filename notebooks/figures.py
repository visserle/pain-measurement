"""Altair plots used by ``notebooks/figures.ipynb``.

The panel functions intentionally return charts without top-level configuration.
This keeps them reusable; call :func:`style_figure` for standalone display or
:func:`compose_figure_svg` to assemble the final multi-panel figure.
"""

import base64
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import altair as alt
import numpy as np
import polars as pl
import vl_convert as vlc
from sklearn.metrics import roc_curve

from src.plots.utils import FEATURE_LABELS

FONT = "Arial"
FONT_SIZE = 12
TITLE_FONT_SIZE = 15
BLUE = "#2171b5"
NAVY = "#000080"
CHANCE_RED = "#d62728"

MODEL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#7f7f7f",
    "#bcbd22",
    "#4c78a8",
]

PARTICIPANT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

SIGNAL_LABELS = {
    "temperature": "Temperature",
    "pain_rating": "Pain rating",
    "pupil_diameter": "Pupil diameter",
    "heart_rate": "Heart rate",
    "eda_tonic": "Tonic EDA",
    "eda_phasic": "Phasic EDA",
}


def style_figure(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the shared publication style to a complete chart."""
    return (
        chart.configure(font=FONT, background="white")
        .configure_axis(
            domainColor="black",
            domainWidth=1,
            grid=False,
            labelColor="black",
            labelFont=FONT,
            labelFontSize=FONT_SIZE,
            labelPadding=4,
            tickColor="black",
            tickSize=5,
            titleColor="black",
            titleFont=FONT,
            titleFontSize=FONT_SIZE,
            titleFontWeight="normal",
            titlePadding=6,
        )
        .configure_header(
            labelFont=FONT,
            labelFontSize=TITLE_FONT_SIZE,
            labelFontWeight="bold",
            titleFont=FONT,
            titleFontSize=TITLE_FONT_SIZE,
        )
        .configure_legend(
            labelFont=FONT,
            labelFontSize=FONT_SIZE - 1,
            titleFont=FONT,
            titleFontSize=FONT_SIZE - 1,
            titleFontWeight="normal",
        )
        .configure_title(
            anchor="middle",
            color="black",
            font=FONT,
            fontSize=TITLE_FONT_SIZE,
            fontWeight="bold",
            offset=8,
        )
        .configure_view(stroke=None)
    )


def _title(text: str) -> alt.TitleParams:
    return alt.TitleParams(text=text, anchor="middle")


def _with_optional_title(
    chart: alt.TopLevelMixin,
    title: str | None,
) -> alt.TopLevelMixin:
    if title is None:
        return chart
    return chart.properties(title=_title(title))


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
        labelLimit=110,
    )
    x = alt.X(
        "time_s:Q",
        title="Time (s)",
        scale=alt.Scale(domain=[0, float(time_s.max())], nice=False),
        axis=alt.Axis(tickCount=10),
    )

    highlights = (
        alt.Chart(interval_data)
        .mark_rect(color="salmon", opacity=0.12)
        .encode(
            x=alt.X("start_s:Q", scale=alt.Scale(domain=[0, float(time_s.max())])),
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


def plot_correlation_heatmap(
    averages: pl.DataFrame,
    *,
    features: Sequence[str] | None = None,
    skip_first_n_seconds: float = 20,
    width: int = 250,
    height: int = 250,
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
                    title="Pearson correlation coefficient",
                    orient="right",
                    titleOrient="left",
                    titlePadding=8,
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
                    range=[*MODEL_COLORS[: len(model_labels)], "black"],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="right",
                    direction="vertical",
                    offset=8,
                    fillColor="white",
                    strokeColor="#bdbdbd",
                    padding=4,
                    symbolStrokeWidth=2,
                    symbolSize=100,
                ),
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


def plot_participant_accuracies(
    results: Mapping[str, pl.DataFrame],
    *,
    width: int = 1320,
    height: int = 210,
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
        labelAngle=-45,
        labelAlign="right",
        labelBaseline="middle",
        labelFontSize=11,
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
                legend=alt.Legend(title="Participant ID", symbolType="circle"),
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


def _panel_label_layer(label: str, *, x: int = -42, y: int = -24) -> alt.Chart:
    return (
        alt.Chart(alt.Data(values=[{"label": label}]))
        .mark_text(
            align="left",
            baseline="bottom",
            clip=False,
            font=FONT,
            fontSize=22,
            fontWeight="bold",
        )
        .encode(text="label:N", x=alt.value(x), y=alt.value(y))
    )


def _without_title(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    titleless = chart.copy(deep=True)
    titleless.title = alt.Undefined
    return titleless


def _without_panel_titles(
    panel_a: alt.LayerChart,
    panel_b: alt.LayerChart,
    panel_c: alt.LayerChart,
    panel_d: alt.HConcatChart,
    panel_e: alt.LayerChart,
) -> tuple[
    alt.LayerChart,
    alt.LayerChart,
    alt.LayerChart,
    alt.HConcatChart,
    alt.LayerChart,
]:
    titleless_d = _without_title(panel_d)
    return (
        _without_title(panel_a),
        _without_title(panel_b),
        _without_title(panel_c),
        titleless_d,
        _without_title(panel_e),
    )


def compose_figure_altair(
    panel_a: alt.LayerChart,
    panel_b: alt.LayerChart,
    panel_c: alt.LayerChart,
    panel_d: alt.HConcatChart,
    panel_e: alt.LayerChart,
) -> alt.VConcatChart:
    """Arrange the five panels without redundant chart titles."""
    panel_a, panel_b, panel_c, panel_d, panel_e = _without_panel_titles(
        panel_a,
        panel_b,
        panel_c,
        panel_d,
        panel_e,
    )
    labeled_a = panel_a + _panel_label_layer("A")
    labeled_b = panel_b + _panel_label_layer("B")
    labeled_c = panel_c + _panel_label_layer("C")
    panel_d_label = _panel_label_layer("D")
    # Match the label overhang so both histograms stay vertically aligned.
    panel_d_spacer = panel_d_label.encode(opacity=alt.value(0))
    labeled_d = alt.hconcat(
        panel_d.hconcat[0] + panel_d_label,
        *(panel + panel_d_spacer for panel in panel_d.hconcat[1:]),
        spacing=panel_d.spacing,
    ).resolve_scale(x="shared", y="shared")
    labeled_e = panel_e + _panel_label_layer("E")

    top_row = alt.hconcat(labeled_a, labeled_b, spacing=25).resolve_scale(
        x="independent",
        y="independent",
        color="independent",
    )
    middle_row = alt.hconcat(labeled_c, labeled_d, spacing=25).resolve_scale(
        x="independent",
        y="independent",
        color="independent",
    )

    return (
        alt.vconcat(top_row, middle_row, labeled_e, spacing=20)
        .resolve_scale(x="independent", y="independent", color="independent")
        .properties(autosize=alt.AutoSizeParams(type="pad", contains="padding"))
    )


def _render_svg(chart: alt.TopLevelMixin) -> tuple[str, float, float]:
    svg = vlc.vegalite_to_svg(style_figure(chart).to_dict())
    root = ET.fromstring(svg)
    width = float(root.attrib["width"])
    height = float(root.attrib["height"])
    encoded = base64.b64encode(svg.encode()).decode()
    return encoded, width, height


def compose_figure_svg(
    panel_a: alt.LayerChart,
    panel_b: alt.LayerChart,
    panel_c: alt.LayerChart,
    panel_d: alt.HConcatChart,
    panel_e: alt.LayerChart,
    *,
    column_gap: int = 25,
    row_gap: int = 20,
    label_gutter: int = 36,
) -> str:
    """Arrange titleless standalone panel renders on one SVG canvas."""
    panel_a, panel_b, panel_c, panel_d, panel_e = _without_panel_titles(
        panel_a,
        panel_b,
        panel_c,
        panel_d,
        panel_e,
    )
    rendered = {
        label: _render_svg(chart)
        for label, chart in zip(
            "ABCDE",
            (panel_a, panel_b, panel_c, panel_d, panel_e),
        )
    }

    _, width_a, height_a = rendered["A"]
    _, width_b, height_b = rendered["B"]
    _, width_c, height_c = rendered["C"]
    _, width_d, height_d = rendered["D"]
    _, width_e, height_e = rendered["E"]

    x_b = label_gutter + width_a + column_gap
    x_d = label_gutter + width_c + column_gap
    top_height = max(height_a, height_b)
    middle_height = max(height_c, height_d)
    y_middle = top_height + row_gap
    y_bottom = y_middle + middle_height + row_gap
    canvas_width = max(
        x_b + label_gutter + width_b,
        x_d + label_gutter + width_d,
        label_gutter + width_e,
    )
    canvas_height = y_bottom + height_e

    placements = {
        "A": (0, 0),
        "B": (x_b, 0),
        "C": (0, y_middle),
        "D": (x_d, y_middle),
        "E": (0, y_bottom),
    }
    elements = []
    for label, (x, y) in placements.items():
        encoded, width, height = rendered[label]
        elements.append(
            f'<text x="{x + 2:g}" y="{y + 24:g}" '
            f'font-family="{FONT}" font-size="22" font-weight="bold">'
            f"{label}</text>"
        )
        elements.append(
            f'<image x="{x + label_gutter:g}" y="{y:g}" '
            f'width="{width:g}" height="{height:g}" '
            f'href="data:image/svg+xml;base64,{encoded}"/>'
        )

    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{canvas_width:g}" height="{canvas_height:g}" '
        f'viewBox="0 0 {canvas_width:g} {canvas_height:g}" '
        'style="max-width:100%;height:auto;background:white">'
        '<rect width="100%" height="100%" fill="white"/>' + "".join(elements) + "</svg>"
    )
