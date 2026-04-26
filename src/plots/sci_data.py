import base64
import os
from copy import deepcopy
from pathlib import Path

import altair as alt
import numpy as np
import polars as pl
from dotenv import load_dotenv
from lxml import etree
from PIL import Image

load_dotenv()
FIGURE_DIR = Path(os.getenv("SCI_DATA_FIGURE_DIR"))

SVG_NS = "http://www.w3.org/2000/svg"
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
# Font size constants
FONT_SIZE_SMALL = 12
FONT_SIZE_LARGE = 14
FONT_WEIGHT_NORMAL = "normal"
LABEL_PADDING = 6


def plot_stimulus_with_physiological_signals(
    ci_values,
    signals,
    signal_labels=None,
    signal_colors=None,
    width=1150,
    height=430,
):
    """
    Plot mean signals with confidence interval bands using Altair.

    Parameters
    ----------
    ci_values : pd.DataFrame
        DataFrame with columns: normalized_timestamp, mean_{sig}, ci_lower_{sig}, ci_upper_{sig}
    signals : list[str]
        List of signal keys to plot (must be keys in signal_labels).
    signal_labels : dict, optional
        Mapping from signal key to display label. Defaults to SIGNAL_LABELS.
    signal_colors : list[str], optional
        List of hex colors, one per signal. Defaults to SIGNAL_COLORS.
    """
    SIGNAL_LABELS = signal_labels or {
        "temperature": "Temperature",
        "pain_rating": "Pain rating",
        "pupil_diameter": "Pupil diameter",
        "eda_tonic": "Tonic EDA",
        "eda_phasic": "Phasic EDA",
        "heart_rate": "Heart rate",
        "mouth_open": "Mouth open",
    }

    SIGNAL_COLORS = signal_colors or [
        "#17396b",  # Temperature - dark navy
        "#d45f3c",  # Pain Rating - orange-red
        "#e8a020",  # Pupil Diameter - golden yellow
        "#5a7a2e",  # Tonic EDA - olive green
        "#3a9e5f",  # Phasic EDA - medium green
        "#3ab8b8",  # Heart Rate - teal
        "#7b4fa0",  # Mouth Open - purple
    ]

    ci_values = ci_values.with_columns(
        (pl.col("normalized_timestamp") / 1000).round(2).alias("time_s")
    )

    layers = []
    for sig, color in zip(signals, SIGNAL_COLORS):
        label = SIGNAL_LABELS[sig]
        base = alt.Chart(ci_values).transform_calculate(label=f"'{label}'")

        line = base.mark_line(color=color).encode(
            x=alt.X("time_s:Q", title="Time (s)"),
            y=alt.Y(f"mean_{sig}:Q", title="Normalized Value"),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)"),
                alt.Tooltip(f"mean_{sig}:Q", title=label, format=".2f"),
            ],
        )

        band = base.mark_area(opacity=0.2, color=color).encode(
            x=alt.X("time_s:Q"),
            y=alt.Y(f"ci_lower_{sig}:Q"),
            y2=alt.Y2(f"ci_upper_{sig}:Q"),
        )

        layers.append(line + band)

    legend_data = (
        alt.Chart(
            alt.Data(
                values=[
                    {"signal": SIGNAL_LABELS[sig], "color": c}
                    for sig, c in zip(signals, SIGNAL_COLORS)
                ]
            )
        )
        .mark_line()
        .encode(
            color=alt.Color(
                "signal:N",
                scale=alt.Scale(
                    domain=[SIGNAL_LABELS[sig] for sig in signals],
                    range=SIGNAL_COLORS[: len(signals)],
                ),
                legend=alt.Legend(
                    title="",
                    orient="right",
                    symbolType="stroke",
                    symbolStrokeWidth=3,
                    symbolSize=400,
                    labelFontSize=FONT_SIZE_LARGE,
                    rowPadding=4,
                ),
            )
        )
    )

    return (
        alt.layer(*layers, legend_data)
        .properties(width=width, height=height)
        .configure_axis(
            grid=False,
            gridColor="#dddddd",
            gridOpacity=0.8,
            labelFontSize=FONT_SIZE_LARGE,
            titleFontSize=FONT_SIZE_LARGE,
            titleFontWeight=FONT_WEIGHT_NORMAL,
            labelFontWeight=FONT_WEIGHT_NORMAL,
        )
        .configure_view(stroke=None, fill="white")
    )


def plot_stimulus_with_labels(
    stimulus,
    filename=None,
    width=1150,
    height=430,
):
    colors = {
        "decreasing_intervals": "#d32f2f",
        "major_decreasing_intervals": "#ff6f60",
        "increasing_intervals": "#388e3c",
        "strictly_increasing_intervals": "#66bb6a",
        "strictly_increasing_intervals_without_plateaus": "#a5d6a7",
        "plateau_intervals": "#1976d2",
        "prolonged_minima_intervals": "#90caf9",
    }

    label_names = {
        "decreasing_intervals": "Decreasing",
        "major_decreasing_intervals": "Major decreasing",
        "increasing_intervals": "Increasing",
        "strictly_increasing_intervals": "Strictly increasing",
        "strictly_increasing_intervals_without_plateaus": "Strictly increasing\n(no plateaus)",
        "plateau_intervals": "Plateau",
        "prolonged_minima_intervals": "Prolonged minima",
    }

    # Keep an explicit order so lane positions and axis labels are stable.
    order = [
        "decreasing_intervals",
        "major_decreasing_intervals",
        "increasing_intervals",
        "strictly_increasing_intervals",
        "strictly_increasing_intervals_without_plateaus",
        "plateau_intervals",
        "prolonged_minima_intervals",
    ]

    labels = dict(stimulus.labels)
    labels["strictly_increasing_intervals"] = [
        stimulus._convert_interval(interval)
        for interval in stimulus.strictly_increasing_intervals_complete_idx
    ]
    labels = {k: labels.get(k, []) for k in order}

    temp = np.asarray(stimulus.y, dtype=float)
    sample_rate = float(stimulus.sample_rate)
    time_s = np.arange(temp.size, dtype=float) / sample_rate
    duration = temp.size / sample_rate

    interval_rows = []
    for interval_type in order:
        interval_label = label_names[interval_type]
        for start_ms, end_ms in labels[interval_type]:
            # Snap boundaries to sample indices so bars align with the plotted signal grid.
            start_idx = int(np.round((start_ms / 1000.0) * sample_rate))
            end_idx = int(np.round((end_ms / 1000.0) * sample_rate))

            start_idx = int(np.clip(start_idx, 0, temp.size))
            end_idx = int(np.clip(end_idx, 0, temp.size))
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, temp.size)
            if end_idx <= start_idx:
                continue

            interval_rows.append(
                {
                    "interval_type": interval_type,
                    "interval_label": interval_label,
                    "start_s": start_idx / sample_rate,
                    "end_s": end_idx / sample_rate,
                }
            )

    interval_df = pl.DataFrame(
        {
            "interval_type": [r["interval_type"] for r in interval_rows],
            "interval_label": [r["interval_label"] for r in interval_rows],
            "start_s": [r["start_s"] for r in interval_rows],
            "end_s": [r["end_s"] for r in interval_rows],
        },
        schema={
            "interval_type": pl.String,
            "interval_label": pl.String,
            "start_s": pl.Float64,
            "end_s": pl.Float64,
        },
    )
    temp_df = pl.DataFrame({"time_s": time_s, "temperature": temp})

    interval_order = [label_names[k] for k in order]

    tick_step = 10
    x_ticks = np.arange(0, int(np.floor(duration)) + 1, tick_step).tolist()
    if not x_ticks:
        x_ticks = [0]
    if x_ticks[-1] != int(np.floor(duration)):
        x_ticks.append(int(np.floor(duration)))

    t_min = float(temp.min())
    t_max = float(temp.max())
    y_tick_step = 0.5
    y_min = np.floor(t_min / y_tick_step) * y_tick_step
    y_max = np.ceil(t_max / y_tick_step) * y_tick_step
    if y_max <= y_min:
        y_max = y_min + y_tick_step
    y_ticks = np.arange(y_min, y_max + 0.5 * y_tick_step, y_tick_step).tolist()

    x_scale = alt.Scale(domain=[0, duration])
    x_axis = alt.Axis(values=x_ticks, title="Time (s)")
    y_temp_scale = alt.Scale(domain=[y_min, y_max], nice=False)

    interval_y = alt.Y(
        "interval_label:N",
        sort=interval_order,
        scale=alt.Scale(domain=interval_order),
        axis=alt.Axis(
            orient="right",
            title="Interval Type",
            titleAngle=270,
            titlePadding=-10,
            ticks=True,
            tickSize=6,
            # tickBand="extent",
            labelLimit=1000,
            labelAngle=30,
            labelAlign="left",
            labelBaseline="middle",
        ),
    )

    bands = (
        alt.Chart(interval_df)
        .mark_bar(opacity=0.35, size=26)
        .encode(
            x=alt.X("start_s:Q", scale=x_scale, axis=x_axis),
            x2="end_s:Q",
            y=interval_y,
            color=alt.Color(
                "interval_type:N",
                scale=alt.Scale(
                    domain=order,
                    range=[colors[k] for k in order],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("interval_label:N", title="Type"),
                alt.Tooltip("start_s:Q", title="Start (s)", format=".2f"),
                alt.Tooltip("end_s:Q", title="End (s)", format=".2f"),
            ],
        )
    )

    line = (
        alt.Chart(temp_df)
        .mark_line(color="#000080", strokeWidth=3)
        .encode(
            x=alt.X("time_s:Q", scale=x_scale, axis=x_axis),
            y=alt.Y(
                "temperature:Q",
                title="Temperature (°C)",
                scale=y_temp_scale,
                axis=alt.Axis(tickMinStep=0.5, values=y_ticks),  # , titlePadding=5),
            ),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("temperature:Q", title="Temperature (deg C)", format=".3f"),
            ],
        )
    )
    chart = (
        alt.layer(bands, line)
        .resolve_scale(y="independent")
        .properties(width=width, height=height)
        .configure_axis(
            grid=False,
            gridColor="#d9d9d9",
            gridDash=[4, 4],
            gridOpacity=0.55,
            labelFontSize=FONT_SIZE_LARGE,
            titleFontSize=FONT_SIZE_LARGE,
            titleFontWeight=FONT_WEIGHT_NORMAL,
            labelFontWeight=FONT_WEIGHT_NORMAL,
        )
        .configure_view(stroke="#b7b7b7", strokeOpacity=0)
    )

    if filename:
        chart.save(filename)

    return chart


def plot_stimulus_seed_grid(
    stimuli: pl.DataFrame,
    columns: int = 3,
    width: int = 220,
    height: int = 130,
    font_size: int = FONT_SIZE_LARGE,
    title_x_offset: int = 8,
) -> alt.Chart:
    """Plot each stimulus seed as a faceted Altair line chart using Polars data."""
    required_columns = {"seed", "time", "y"}
    missing_columns = required_columns.difference(stimuli.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    chart_data = (
        stimuli.select("seed", "time", "y")
        .sort(["seed", "time"])
        .with_columns(
            pl.concat_str(
                [pl.lit("Random seed: "), pl.col("seed").cast(pl.String)]
            ).alias("seed_label")
        )
    )

    seed_labels = chart_data.select("seed_label").unique().sort("seed_label")

    charts = []
    for row in seed_labels.iter_rows(named=True):
        seed_label = row["seed_label"]
        seed_data = chart_data.filter(pl.col("seed_label") == seed_label)

        line = (
            alt.Chart(seed_data)
            .mark_line(color="#1f77b4", strokeWidth=2)
            .encode(
                x=alt.X("time:Q", axis=None, scale=alt.Scale(zero=False)),
                y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, nice=False)),
                tooltip=[
                    alt.Tooltip("seed:Q", title="Seed"),
                    alt.Tooltip("time:Q", title="Time", format=".0f"),
                    alt.Tooltip("y:Q", title="Temperature", format=".3f"),
                ],
            )
            .properties(width=width, height=height)
        )

        title = (
            alt.Chart({"values": [{"label": seed_label}]})
            .mark_text(
                align="center",
                baseline="bottom",
                fontSize=font_size,
                fontWeight=FONT_WEIGHT_NORMAL,
                clip=False,
            )
            .encode(
                text="label:N",
                x=alt.value(width / 2 - title_x_offset),
                y=alt.value(-LABEL_PADDING),
            )
        )

        charts.append(alt.layer(line, title))

    return (
        alt.concat(*charts, columns=columns)
        .resolve_scale(x="independent", y="independent")
        .configure_view(stroke=None)
        .configure_concat(spacing=10)
    )


def _viewbox_dims(path: Path) -> tuple[float, float]:
    if path.suffix.lower() in _RASTER_SUFFIXES:
        with Image.open(path) as img:
            return float(img.width), float(img.height)
    root = etree.parse(str(path)).getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, w, h = vb.split()
        return float(w), float(h)
    return float(root.get("width", 0)), float(root.get("height", 0))


def compose_panel_figure(
    output_path: Path | str,
    row1: Path,
    row2_left: Path,
    row2_right: Path,
    gap: int = 16,
    label_size: int = 18,
) -> None:
    """Compose three SVGs/images: one full-width top row, two side-by-side bottom row.

    Scales both bottom panels to the same height so they fill the canvas width.
    Adds lowercase alphabetical panel labels (a, b, c).
    Supports SVG and raster images (PNG, JPEG, etc.) for any panel.
    """
    w_a, h_a = _viewbox_dims(row1)
    w_b, h_b = _viewbox_dims(row2_left)
    w_c, h_c = _viewbox_dims(row2_right)

    canvas_w = int(w_a)

    # Scale B and C to equal height so (wB + gap + wC) == canvas_w
    ar_b, ar_c = w_b / h_b, w_c / h_c
    h2 = (canvas_w - gap) / (ar_b + ar_c)
    w_panel_b = round(h2 * ar_b)
    w_panel_c = canvas_w - gap - w_panel_b
    h_row2 = round(h2)

    canvas_h = round(h_a) + gap + h_row2

    nsmap = {None: SVG_NS}
    root_svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap=nsmap)
    root_svg.set("width", str(canvas_w))
    root_svg.set("height", str(canvas_h))
    root_svg.set("viewBox", f"0 0 {canvas_w} {canvas_h}")

    bg = etree.SubElement(root_svg, f"{{{SVG_NS}}}rect")
    bg.set("width", "100%")
    bg.set("height", "100%")
    bg.set("fill", "white")

    y2 = round(h_a) + gap
    panels = [
        # (label, path, x, y, w, h, vb_w, vb_h)
        ("a", row1, 0, 0, canvas_w, round(h_a), int(w_a), int(h_a)),
        ("b", row2_left, 0, y2, w_panel_b, h_row2, int(w_b), int(h_b)),
        ("c", row2_right, w_panel_b + gap, y2, w_panel_c, h_row2, int(w_c), int(h_c)),
    ]

    for label, path, x, y, w, h, vb_w, vb_h in panels:
        if path.suffix.lower() in _RASTER_SUFFIXES:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode()
            img_elem = etree.SubElement(root_svg, f"{{{SVG_NS}}}image")
            img_elem.set("x", str(x))
            img_elem.set("y", str(y))
            img_elem.set("width", str(w))
            img_elem.set("height", str(h))
            img_elem.set("href", f"data:{mime};base64,{encoded}")
            img_elem.set("preserveAspectRatio", "xMidYMid meet")
        else:
            src_root = etree.parse(str(path)).getroot()
            child_svg = etree.SubElement(root_svg, f"{{{SVG_NS}}}svg")
            child_svg.set("x", str(x))
            child_svg.set("y", str(y))
            child_svg.set("width", str(w))
            child_svg.set("height", str(h))
            child_svg.set("viewBox", f"0 0 {vb_w} {vb_h}")
            child_svg.set("preserveAspectRatio", "xMidYMid meet")
            for child in src_root:
                child_svg.append(deepcopy(child))

        text = etree.SubElement(root_svg, f"{{{SVG_NS}}}text")
        text.set("x", str(x))
        text.set("y", str(y + label_size - 3))
        text.set("font-family", "Arial, Helvetica, sans-serif")
        text.set("font-size", str(label_size))
        text.set("font-weight", "bold")
        text.text = label

    etree.ElementTree(root_svg).write(
        str(output_path),
        pretty_print=True,
        xml_declaration=True,
        encoding="utf-8",
    )
    print(f"Saved → {output_path}")
