"""Altair participant-by-time model-inference visualization."""

from collections.abc import Mapping, Sequence

import altair as alt
import numpy as np
import polars as pl

from src.experiments.measurement.stimulus_generator import StimulusGenerator

from .style import FONT, FONT_SIZE, MODEL_INFERENCE_DURATION_MS, _with_optional_title


def _signed_prediction_confidence(
    decrease_probabilities: np.ndarray,
    classification_threshold: float,
) -> np.ndarray:
    confidence = np.empty_like(decrease_probabilities, dtype=float)
    below_threshold = decrease_probabilities < classification_threshold
    confidence[below_threshold] = (
        decrease_probabilities[below_threshold] - classification_threshold
    ) / classification_threshold
    confidence[~below_threshold] = (
        decrease_probabilities[~below_threshold] - classification_threshold
    ) / (1 - classification_threshold)
    return confidence


def _model_inference_color_encoding(
    *,
    only_decreases: bool,
    only_non_decreases: bool,
    gradient_length: int,
) -> alt.Color:
    if only_non_decreases:
        domain = [-1, 0]
        color_range = ["#ff591a", "white"]
        values = [-1, -0.75, -0.5, -0.25, 0]
        legend_title = "Prediction confidence for non-decreases"
    elif only_decreases:
        domain = [0, 1]
        color_range = ["white", "#0033cc"]
        values = [0, 0.25, 0.5, 0.75, 1]
        legend_title = "Prediction confidence for decreases"
    else:
        domain = [-1, 0, 1]
        color_range = ["#ff591a", "white", "#0033cc"]
        values = [-1, -0.5, 0, 0.5, 1]
        legend_title = "Prediction confidence"

    return alt.Color(
        "confidence:Q",
        scale=alt.Scale(
            domain=domain,
            range=color_range,
            clamp=True,
            interpolate="rgb",
        ),
        legend=alt.Legend(
            title=legend_title,
            orient="right",
            titleOrient="left",
            titlePadding=8,
            titleLimit=gradient_length,
            gradientLength=gradient_length,
            gradientThickness=12,
            values=values,
            format=".2f",
        ),
    )


def plot_model_inference(
    all_probabilities: Mapping[int, Mapping[str, Sequence]],
    *,
    sample_duration_ms: int = 7000,
    classification_threshold: float = 0.9,
    step_size_ms: int = 250,
    display_step_size_ms: int = 250,
    seeds_to_plot: Sequence[int] | None = None,
    only_decreases: bool = True,
    only_non_decreases: bool = False,
    ncols: int = 2,
    width: int = 500,
    height: int = 110,
    stimulus_scale: float = 0.5,
    stimulus_linewidth: float = 1.5,
    ratings_df: pl.DataFrame | None = None,
    show_actual_rating_ci: bool = False,
    rating_confidence_level: float = 0.95,
    rating_linewidth: float = 1.2,
    rating_scale: float = 0.25,
    rating_color: str = "#2ca25f",
    rating_ci_opacity: float = 0.12,
    rating_label: str = "Actual rating (mean ± CI)",
    show_legend: bool = True,
    column_spacing: int = 20,
    row_spacing: int = 10,
    panel_border_color: str = "#606060",
    panel_border_width: float = 0.8,
    title: str | None = None,
) -> alt.VConcatChart:
    """Plot participant-level prediction confidence over time for each stimulus."""
    if not 0 < classification_threshold < 1:
        raise ValueError("classification_threshold must be between 0 and 1")
    if step_size_ms <= 0 or MODEL_INFERENCE_DURATION_MS % step_size_ms:
        raise ValueError("step_size_ms must evenly divide 180000 ms")
    if sample_duration_ms % step_size_ms:
        raise ValueError("sample_duration_ms must be a multiple of step_size_ms")
    if (
        display_step_size_ms <= 0
        or MODEL_INFERENCE_DURATION_MS % display_step_size_ms
        or display_step_size_ms % step_size_ms
    ):
        raise ValueError(
            "display_step_size_ms must evenly divide 180000 ms and be a "
            "multiple of step_size_ms"
        )
    if ncols <= 0:
        raise ValueError("ncols must be positive")
    if not 0 < rating_confidence_level < 1:
        raise ValueError("rating_confidence_level must be between 0 and 1")
    if not 0 <= rating_ci_opacity <= 1:
        raise ValueError("rating_ci_opacity must be between 0 and 1")
    if panel_border_width < 0:
        raise ValueError("panel_border_width must be non-negative")
    if show_actual_rating_ci and ratings_df is None:
        raise ValueError("ratings_df is required when show_actual_rating_ci is True")

    available_seeds = sorted(all_probabilities)
    seeds = (
        available_seeds
        if seeds_to_plot is None
        else [seed for seed in seeds_to_plot if seed in all_probabilities]
    )
    if not seeds:
        raise ValueError("None of the requested stimulus seeds are available")

    participants = sorted(
        {str(participant) for seed in seeds for participant in all_probabilities[seed]},
        key=int,
    )
    if not participants:
        raise ValueError("No participant probabilities are available")

    participant_count = len(participants)
    participant_ticks = [index + 0.5 for index in range(participant_count)]
    participant_label_expr = f"{participants!r}[floor(datum.value)]"
    source_time_point_count = MODEL_INFERENCE_DURATION_MS // step_size_ms
    time_point_count = MODEL_INFERENCE_DURATION_MS // display_step_size_ms
    aggregation_factor = display_step_size_ms // step_size_ms
    time_step_s = display_step_size_ms / 1000
    padding_steps = sample_duration_ms // step_size_ms
    nrows = (len(seeds) + ncols - 1) // ncols
    color = _model_inference_color_encoding(
        only_decreases=only_decreases,
        only_non_decreases=only_non_decreases,
        gradient_length=nrows * height + (nrows - 1) * row_spacing,
    )
    rating_ci_by_seed = {}
    if show_actual_rating_ci:
        from src.plots.model_inference import compute_actual_rating_ci_by_seed

        rating_ci_by_seed = compute_actual_rating_ci_by_seed(
            ratings_df=ratings_df,
            all_probabilities=all_probabilities,
            seeds_to_plot=seeds,
            step_size=display_step_size_ms,
            confidence_level=rating_confidence_level,
        )

    panels = []
    for panel_index, seed in enumerate(seeds):
        probability_by_participant = all_probabilities[seed]
        confidence_by_participant = {}
        for participant, trials in probability_by_participant.items():
            if not trials:
                continue
            trial_probabilities = np.asarray(trials[-1], dtype=float)
            if trial_probabilities.ndim != 2 or trial_probabilities.shape[1] < 2:
                raise ValueError(
                    f"Probabilities for participant {participant} and seed {seed} "
                    "must have shape (time, classes)"
                )
            signed_confidence = _signed_prediction_confidence(
                trial_probabilities[:, 1],
                classification_threshold,
            )
            padded_confidence = np.zeros(source_time_point_count, dtype=float)
            prediction_count = min(
                len(signed_confidence),
                source_time_point_count - padding_steps,
            )
            padded_confidence[padding_steps : padding_steps + prediction_count] = (
                signed_confidence[:prediction_count]
            )
            confidence_by_participant[str(participant)] = padded_confidence.reshape(
                time_point_count,
                aggregation_factor,
            ).mean(axis=1)

        heatmap_rows = []
        for participant_index, participant in enumerate(participants):
            participant_confidence = confidence_by_participant.get(participant)
            for time_index in range(time_point_count):
                if participant_confidence is None:
                    confidence_value = None
                else:
                    confidence_value = float(participant_confidence[time_index])
                heatmap_rows.append(
                    {
                        "time_start_s": time_index * time_step_s,
                        "time_end_s": (time_index + 1) * time_step_s,
                        "participant_start": participant_index,
                        "participant_end": participant_index + 1,
                        "participant": participant,
                        "confidence": confidence_value,
                        "seed": str(seed),
                    }
                )

        panel_row = panel_index // ncols
        bottom_row = panel_row == nrows - 1
        left_column = panel_index % ncols == 0
        x_axis = alt.Axis(values=[0, 90, 180], title="Time (s)") if bottom_row else None
        y_axis = (
            alt.Axis(
                values=participant_ticks,
                labelExpr=participant_label_expr,
                labelOverlap=False,
                title=("Participant ID" if panel_row == (nrows - 1) // 2 else None),
                tickSize=0,
            )
            if left_column
            else None
        )
        x = alt.X(
            "time_start_s:Q",
            scale=alt.Scale(domain=[0, 180], nice=False),
            axis=x_axis,
        )
        y = alt.Y(
            "participant_start:Q",
            scale=alt.Scale(domain=[0, participant_count], nice=False),
            axis=y_axis,
        )
        heatmap_data = alt.Data(values=heatmap_rows)
        missing_background = (
            alt.Chart(heatmap_data)
            .mark_rect(color="#f0f0f0")
            .encode(
                x=x,
                x2="time_end_s:Q",
                y=y,
                y2="participant_end:Q",
            )
        )
        confidence_cells = (
            alt.Chart(heatmap_data)
            .transform_filter("isValid(datum.confidence)")
            .mark_rect(opacity=1)
            .encode(
                x=x,
                x2="time_end_s:Q",
                y=y,
                y2="participant_end:Q",
                color=color,
                tooltip=[
                    alt.Tooltip("seed:N", title="Stimulus seed"),
                    alt.Tooltip("participant:N", title="Participant ID"),
                    alt.Tooltip("time_start_s:Q", title="Time (s)", format=".2f"),
                    alt.Tooltip("confidence:Q", title="Confidence", format=".3f"),
                ],
            )
        )

        stimulus = StimulusGenerator(
            seed=seed,
            config={"sample_rate": max(1, 1000 // display_step_size_ms)},
        )
        stimulus_values = np.asarray(stimulus.y, dtype=float)
        stimulus_values = (
            2
            * (
                (stimulus_values - stimulus_values.min())
                / (stimulus_values.max() - stimulus_values.min())
            )
            - 1
        )
        if len(stimulus_values) != time_point_count:
            stimulus_values = np.interp(
                np.linspace(0, 1, time_point_count),
                np.linspace(0, 1, len(stimulus_values)),
                stimulus_values,
            )
        y_center = participant_count / 2
        y_amplitude = y_center * stimulus_scale
        stimulus_data = alt.Data(
            values=[
                {
                    "time_s": time_index * time_step_s,
                    "participant_position": y_center + value * y_amplitude,
                }
                for time_index, value in enumerate(stimulus_values)
            ]
        )
        stimulus_line = (
            alt.Chart(stimulus_data)
            .mark_line(
                color="black",
                opacity=0.8,
                strokeWidth=stimulus_linewidth,
                clip=True,
            )
            .encode(
                x=alt.X(
                    "time_s:Q",
                    scale=alt.Scale(domain=[0, 180], nice=False),
                    axis=x_axis,
                ),
                y=alt.Y(
                    "participant_position:Q",
                    scale=alt.Scale(domain=[0, participant_count], nice=False),
                    axis=y_axis,
                ),
            )
        )
        rating_layers = []
        rating_ci = rating_ci_by_seed.get(seed)
        if rating_ci:
            rating_mean = np.asarray(rating_ci["mean"], dtype=float)
            rating_lower = np.asarray(rating_ci["ci_lower"], dtype=float)
            rating_upper = np.asarray(rating_ci["ci_upper"], dtype=float)
            rating_time_s = np.asarray(rating_ci["time_points_s"], dtype=float)
            rating_y_amplitude = y_center * rating_scale
            rating_data = alt.Data(
                values=[
                    {
                        "time_s": float(time_s),
                        "rating_mean": float(mean),
                        "rating_lower": float(lower),
                        "rating_upper": float(upper),
                        "rating_mean_position": float(
                            np.clip(
                                (2 * mean - 1) * rating_y_amplitude + y_center,
                                0,
                                participant_count,
                            )
                        ),
                        "rating_lower_position": float(
                            np.clip(
                                (2 * lower - 1) * rating_y_amplitude + y_center,
                                0,
                                participant_count,
                            )
                        ),
                        "rating_upper_position": float(
                            np.clip(
                                (2 * upper - 1) * rating_y_amplitude + y_center,
                                0,
                                participant_count,
                            )
                        ),
                    }
                    for time_s, mean, lower, upper in zip(
                        rating_time_s,
                        rating_mean,
                        rating_lower,
                        rating_upper,
                    )
                    if np.isfinite(time_s)
                    and np.isfinite(mean)
                    and np.isfinite(lower)
                    and np.isfinite(upper)
                ]
            )
            rating_band = (
                alt.Chart(rating_data)
                .mark_area(
                    color=rating_color,
                    opacity=rating_ci_opacity,
                    clip=True,
                )
                .encode(
                    x=alt.X(
                        "time_s:Q",
                        scale=alt.Scale(domain=[0, 180], nice=False),
                        axis=x_axis,
                    ),
                    y=alt.Y(
                        "rating_lower_position:Q",
                        scale=alt.Scale(
                            domain=[0, participant_count],
                            nice=False,
                        ),
                        axis=y_axis,
                    ),
                    y2="rating_upper_position:Q",
                )
            )
            rating_line = (
                alt.Chart(rating_data)
                .mark_line(
                    color=rating_color,
                    opacity=0.95,
                    strokeWidth=rating_linewidth,
                    clip=True,
                )
                .encode(
                    x=alt.X(
                        "time_s:Q",
                        scale=alt.Scale(domain=[0, 180], nice=False),
                        axis=x_axis,
                    ),
                    y=alt.Y(
                        "rating_mean_position:Q",
                        scale=alt.Scale(
                            domain=[0, participant_count],
                            nice=False,
                        ),
                        axis=y_axis,
                    ),
                    tooltip=[
                        alt.Tooltip("time_s:Q", title="Time (s)", format=".1f"),
                        alt.Tooltip(
                            "rating_mean:Q",
                            title=rating_label,
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "rating_lower:Q",
                            title=f"{rating_confidence_level:.0%} CI lower",
                            format=".2f",
                        ),
                        alt.Tooltip(
                            "rating_upper:Q",
                            title=f"{rating_confidence_level:.0%} CI upper",
                            format=".2f",
                        ),
                    ],
                )
            )
            rating_layers = [rating_band, rating_line]
        panel_layers = [
            missing_background,
            confidence_cells,
            *rating_layers,
            stimulus_line,
        ]
        if panel_border_width > 0:
            frame_data = alt.Data(
                values=[
                    {
                        "x_min": 0,
                        "x_max": 180,
                        "y_min": 0,
                        "y_max": participant_count,
                    }
                ]
            )
            panel_layers.append(
                alt.Chart(frame_data)
                .mark_rect(
                    fillOpacity=0,
                    stroke=panel_border_color,
                    strokeWidth=panel_border_width,
                )
                .encode(
                    x=alt.X(
                        "x_min:Q",
                        scale=alt.Scale(domain=[0, 180], nice=False),
                        axis=x_axis,
                    ),
                    x2="x_max:Q",
                    y=alt.Y(
                        "y_min:Q",
                        scale=alt.Scale(
                            domain=[0, participant_count],
                            nice=False,
                        ),
                        axis=y_axis,
                    ),
                    y2="y_max:Q",
                )
            )
        panels.append(alt.layer(*panel_layers).properties(width=width, height=height))

    rows = [
        alt.hconcat(
            *panels[start : start + ncols],
            spacing=column_spacing,
        ).resolve_scale(x="shared", y="shared", color="shared")
        for start in range(0, len(panels), ncols)
    ]
    chart = alt.vconcat(*rows, spacing=row_spacing).resolve_scale(
        x="shared",
        y="shared",
        color="shared",
    )
    if show_legend:
        used_columns = min(ncols, len(panels))
        plot_width = used_columns * width + (used_columns - 1) * column_spacing
        legend_entries = [
            {
                "label": "Temperature",
                "color": "black",
                "show_band": False,
            }
        ]
        if show_actual_rating_ci:
            legend_entries.append(
                {
                    "label": rating_label,
                    "color": rating_color,
                    "show_band": True,
                }
            )
        legend_font_size = FONT_SIZE - 1
        sample_width = 26
        sample_label_gap = 8
        item_gap = 28
        approximate_character_width = legend_font_size * 0.52
        label_widths = [
            max(60, len(entry["label"]) * approximate_character_width)
            for entry in legend_entries
        ]
        legend_width = sum(
            sample_width + sample_label_gap + label_width
            for label_width in label_widths
        ) + item_gap * (len(legend_entries) - 1)
        legend_start = max(0, (plot_width - legend_width) / 2)
        legend_rows = []
        legend_cursor = legend_start
        for entry, label_width in zip(legend_entries, label_widths, strict=True):
            legend_rows.append(
                {
                    **entry,
                    "sample_start": legend_cursor,
                    "sample_end": legend_cursor + sample_width,
                    "text_x": legend_cursor + sample_width + sample_label_gap,
                }
            )
            legend_cursor += sample_width + sample_label_gap + label_width + item_gap
        legend_data = alt.Data(values=legend_rows)
        legend_x = alt.X(
            "sample_start:Q",
            scale=alt.Scale(domain=[0, plot_width], nice=False),
            axis=None,
        )
        legend_band = (
            alt.Chart(legend_data)
            .transform_filter("datum.show_band")
            .mark_rect(opacity=rating_ci_opacity)
            .encode(
                x=legend_x,
                x2="sample_end:Q",
                y=alt.value(5),
                y2=alt.value(17),
                color=alt.Color("color:N", scale=None, legend=None),
            )
        )
        legend_lines = (
            alt.Chart(legend_data)
            .mark_rule(strokeWidth=1.5)
            .encode(
                x=legend_x,
                x2="sample_end:Q",
                y=alt.value(11),
                color=alt.Color("color:N", scale=None, legend=None),
            )
        )
        legend_labels = (
            alt.Chart(legend_data)
            .mark_text(
                align="left",
                baseline="middle",
                font=FONT,
                fontSize=legend_font_size,
                color="black",
            )
            .encode(
                x=alt.X(
                    "text_x:Q",
                    scale=alt.Scale(domain=[0, plot_width], nice=False),
                    axis=None,
                ),
                y=alt.value(11),
                text="label:N",
            )
        )
        overlay_legend = alt.layer(
            legend_band,
            legend_lines,
            legend_labels,
        ).properties(width=plot_width, height=20)
        chart = alt.vconcat(chart, overlay_legend, spacing=2).resolve_scale(
            x="independent",
            y="independent",
            color="independent",
        )
    return _with_optional_title(chart, title)
