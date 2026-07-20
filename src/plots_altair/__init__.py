"""Reusable Altair figures for the pain-measurement project."""

from .accuracy_distributions import plot_accuracy_distributions
from .composition import compose_figure_altair
from .correlation_heatmap import plot_correlation_heatmap
from .grand_averages import plot_grand_averaged_signals, plot_grand_averages_grid
from .model_inference import plot_model_inference
from .participant_accuracies import plot_participant_accuracies
from .participant_correlations import plot_participant_correlations
from .roc_curves import plot_roc_curves
from .stimulus_intervals import plot_stimulus_intervals
from .stimulus_seed_grid import plot_stimulus_seed_grid
from .stimulus_temperature import plot_stimulus_temperature
from .style import (
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
    "plot_grand_averages_grid",
    "plot_model_inference",
    "plot_participant_accuracies",
    "plot_participant_correlations",
    "plot_roc_curves",
    "plot_stimulus_intervals",
    "plot_stimulus_seed_grid",
    "plot_stimulus_temperature",
    "style_figure",
]
