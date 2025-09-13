import logging

import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from polars import col

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def calculate_correlations_per_trial(
    df: pl.DataFrame,
    reference: str,
    targets: str | list[str],
    trial_column: str = "trial_id",
):
    """
    Calculate correlations between a reference column and one or more target columns at trial level

    Args:
        df: Polars DataFrame
        reference: Reference column name to correlate against
        targets: Target column name(s) to correlate - can be a string or list of strings
        trial_column: Column name for trial identifier (default: "trial_id")

    Returns:
        DataFrame with trial-level correlations

    Note:
        To aggregate the correlations to participant level, use the
        `aggregate_correlations_fisher_z` function.
    """
    # Convert single string to list for uniform processing
    if isinstance(targets, str):
        targets = [targets]

    # Build aggregation expressions
    agg_exprs = [
        pl.first(["trial_number", "participant_id", "stimulus_seed"]),
        # Single n_obs column for the reference (assuming all targets have same non-null pattern)
        pl.col(reference).is_not_null().sum().alias("n_obs"),
    ]

    # Add correlation for each target column
    for target_col in targets:
        # Create correlation column name
        corr_column = f"{reference}_{target_col}_corr"
        agg_exprs.append(pl.corr(reference, target_col).alias(corr_column))

    # Calculate correlations for each trial_id
    corr_per_trial = (
        df.group_by(trial_column, maintain_order=True)
        .agg(agg_exprs)
        .sort([trial_column])
    )

    return corr_per_trial


def calculate_participant_stats(
    corr_by_trial: pl.DataFrame,
    targets: list[str],
    reference: str = "temperature",
):
    """Calculate mean and std correlations per participant from trial-level correlation data"""

    # Generate aggregation expressions dynamically
    agg_expressions = []

    for target in targets:
        corr_col = f"{reference}_{target}_corr"
        mean_alias = f"{target}_mean"
        std_alias = f"{target}_std"

        agg_expressions.extend(
            [
                col(corr_col).mean().alias(mean_alias),
                col(corr_col).std().alias(std_alias),
            ]
        )

    participant_stats = corr_by_trial.group_by("participant_id").agg(agg_expressions)

    return participant_stats


def plot_correlations_per_participant(
    stats_df: pl.DataFrame,
    targets: list[str],
    labels_dict: dict[str, str],
    reference: str = "temperature",
    ylim: tuple[float, float] = (-0.4, 1.0),
):
    """Plot correlation data from the wide format statistics DataFrame"""
    fig = plt.figure(figsize=(12, 6))

    # Define plotting properties
    colors = ["red", "blue", "green", "orange", "purple"]
    labels = [
        labels_dict.get(target, target.replace("_", " ").title()) for target in targets
    ]
    # Generate column names dynamically based on targets
    mean_cols = [f"{target}_mean" for target in targets]
    std_cols = [f"{target}_std" for target in targets]

    # Create dodge offsets
    dodge_width = 0.1
    n_signals = len(targets)
    dodge_positions = np.linspace(-dodge_width / 2, dodge_width / 2, n_signals)

    # Get participant IDs
    participant_ids = stats_df.get_column("participant_id").to_numpy()

    # Plot each signal type
    for i, (mean_col, std_col, color, label, dodge) in enumerate(
        zip(mean_cols, std_cols, colors, labels, dodge_positions)
    ):
        mean_corrs = stats_df.get_column(mean_col).to_numpy()
        std_corrs = stats_df.get_column(std_col).to_numpy()
        x_positions = participant_ids + dodge

        plt.errorbar(
            x_positions,
            mean_corrs,
            yerr=std_corrs,
            fmt="o",
            color=color,
            label=label,
            capsize=3,
            alpha=0.6,
            markersize=6,
        )

    plt.xlabel("Participant ID")
    plt.ylabel(f"Mean Correlation with {reference.title()} (±SD)")
    plt.xlim(participant_ids.min() - 0.5, participant_ids.max() + 0.5)
    plt.ylim(ylim)
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center", ncol=5)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xticks(participant_ids)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return fig
