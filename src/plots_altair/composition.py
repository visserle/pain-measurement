"""Native Altair composition for the multi-panel figure."""

import altair as alt

from .style import FONT


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
    *,
    row_spacing: int = 6,
    middle_row_label_y: int = -10,
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
    labeled_c = panel_c + _panel_label_layer("C", y=middle_row_label_y)
    panel_d_label = _panel_label_layer("D", y=middle_row_label_y)
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
        alt.vconcat(top_row, middle_row, labeled_e, spacing=row_spacing)
        .resolve_scale(x="independent", y="independent", color="independent")
        .properties(autosize=alt.AutoSizeParams(type="pad", contains="padding"))
    )
