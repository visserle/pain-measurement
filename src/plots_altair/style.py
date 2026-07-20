"""Shared styling and visual constants for the Altair figures."""

import altair as alt

FONT = "Arial"
FONT_SIZE = 15
TITLE_FONT_SIZE = 18
BLUE = "#2171b5"
NAVY = "#000080"
CHANCE_RED = "#d62728"
MODEL_INFERENCE_DURATION_MS = 180_000

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
