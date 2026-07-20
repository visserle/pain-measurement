"""Compatibility imports for the Altair plotting package.

New code should import these functions from :mod:`src.plots_altair`.
"""

from src.plots_altair import (
    BLUE,
    CHANCE_RED,
    FONT,
    FONT_SIZE,
    MODEL_COLORS,
    MODEL_INFERENCE_DURATION_MS,
    NAVY,
    PARTICIPANT_COLORS,
    SIGNAL_COLORS,
    SIGNAL_LABELS,
    TITLE_FONT_SIZE,
    compose_figure_altair,
    plot_accuracy_distributions,
    plot_correlation_heatmap,
    plot_grand_averaged_signals,
    plot_model_inference,
    plot_participant_accuracies,
    plot_roc_curves,
    plot_stimulus_temperature,
    style_figure,
)

__all__ = [
    "BLUE",
    "CHANCE_RED",
    "FONT",
    "FONT_SIZE",
    "MODEL_COLORS",
    "MODEL_INFERENCE_DURATION_MS",
    "NAVY",
    "PARTICIPANT_COLORS",
    "SIGNAL_COLORS",
    "SIGNAL_LABELS",
    "TITLE_FONT_SIZE",
    "compose_figure_altair",
    "plot_accuracy_distributions",
    "plot_correlation_heatmap",
    "plot_grand_averaged_signals",
    "plot_model_inference",
    "plot_participant_accuracies",
    "plot_roc_curves",
    "plot_stimulus_temperature",
    "style_figure",
]
