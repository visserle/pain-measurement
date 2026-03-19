"""
Global average over stimulus seeds for each trial.
"""

import logging
from statistics import NormalDist

import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from polars import col
from scipy import signal

from src.features.resampling import add_normalized_timestamp, add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_robust_standard, scale_standard

plt.style.use("./src/plots/style.mplstyle")

BIN_SIZE = 0.1  # in seconds
CONFIDENCE_LEVEL = 0.95
LABELS = {
    "temperature": "Temperature",
    "pain_rating": "Pain rating",
    "pupil_diameter": "Pupil diameter",
    "heart_rate": "Heart rate",
    "eda_tonic": "Tonic EDA",
    "eda_phasic": "Phasic EDA",
    "cheek_raise": "Cheek raise",
    "mouth_open": "Mouth open",
    "upper_lip_raise": "Upper lip raise",
    "nose_wrinkle": "Nose wrinkle",
    "brow_furrow": "Brow furrow",
}

# Explicit, high-contrast colors for consistent signal identity across plots.
SIGNAL_COLORS = {
    "temperature": "#0072B2",  # blue
    "pain_rating": "#D55E00",  # vermillion
    "pupil_diameter": "#CC79A7",  # magenta
    "heart_rate": "#56B4E9",  # sky blue
    "eda_tonic": "#009E73",  # green
    "eda_phasic": "#F0E442",  # yellow
    "cheek_raise": "#7A68A6",  # violet
    "mouth_open": "#8C564B",  # brown
    "upper_lip_raise": "#E377C2",  # pink
    "nose_wrinkle": "#17BECF",  # cyan
    "brow_furrow": "#1F77B4",  # deep blue
}


logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def _calculate_z_score(confidence_level: float) -> float:
    """
    Calculate z-score for the given confidence level (e.g., 0.95 -> 1.96).
    """
    return NormalDist().inv_cdf((1 + confidence_level) / 2)


def average_over_stimulus_seeds(
    df: pl.DataFrame,
    signals: list[str],
    scaling: str | None = "min_max",
    bin_size: int | float = BIN_SIZE,
    confidence_level: float = CONFIDENCE_LEVEL,
    participant_ids: list[int] | None = None,
) -> pl.DataFrame:
    """
    Aggregate over stimulus seeds by calculating mean, std, sem, and confidence
    intervals for each signal at each time point."""
    if bin_size <= 0:
        raise ValueError(f"bin_size must be > 0, got {bin_size}.")

    required_cols = ["participant_id", "stimulus_seed", *signals]
    missing_cols = [column for column in required_cols if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}.")

    match scaling:
        case "min_max":
            df = scale_min_max(
                df,
                exclude_additional_columns=[
                    "temperature",  # already normalized
                    "pain_rating",
                    "brow_furrow",
                    "cheek_raise",
                    "mouth_open",
                    "upper_lip_raise",
                    "nose_wrinkle",
                ],
            )
        case "standard":
            df = scale_standard(df)
        case "robust_standard":
            df = scale_robust_standard(df)
        case None:
            pass
        case _:
            raise ValueError(f"Unknown scaling method: {scaling}")

    if participant_ids is not None:
        df = df.filter(col("participant_id").is_in(participant_ids))

    # Use the precomputed normalized time axis when available.
    if "normalized_timestamp" not in df.columns:
        raise ValueError("df must contain 'normalized_timestamp'.")

    z_score = _calculate_z_score(confidence_level)

    # Group by stimulus seed and normalized timestamp, then calculate mean, std, sem, ci
    return (
        df.group_by(
            col("stimulus_seed"), col("normalized_timestamp"), maintain_order=True
        )
        .agg(
            *[col(c).mean().alias(f"mean_{c}") for c in signals],
            *[col(c).std().alias(f"std_{c}") for c in signals],
            pl.len().alias("n"),
        )
        .sort("normalized_timestamp")
        .with_columns(
            *[(col(f"std_{c}") / col("n").sqrt()).alias(f"sem_{c}") for c in signals],
        )
        .with_columns(
            *[
                (col(f"mean_{c}") - z_score * col(f"sem_{c}")).alias(f"ci_lower_{c}")
                for c in signals
            ],
            *[
                (col(f"mean_{c}") + z_score * col(f"sem_{c}")).alias(f"ci_upper_{c}")
                for c in signals
            ],
        )
    )


def plot_single_stimulus_seed(
    averages_with_ci_df: pl.DataFrame,
    stimulus_seed: int,
    signals: list[str],
    alpha: float = 1.0,
    show_ci: bool = True,
) -> plt.Figure:
    """
    Plot averages with confidence intervals for a single stimulus seed.

    Args:
        averages_with_ci_df: DataFrame containing averages and confidence intervals
        stimulus_seed: The stimulus seed to plot
        signals: List of signal names to plot
        alpha: Transparency of the lines (0-1)
        show_ci: Whether to show confidence intervals

    Returns:
        Matplotlib figure
    """
    # Filter data for the specified stimulus seed
    seed_data = averages_with_ci_df.filter(pl.col("stimulus_seed") == stimulus_seed)

    if seed_data.height == 0:
        raise ValueError(f"No data found for stimulus_seed={stimulus_seed}")

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 4))

    # Color palette matching the reference image
    color_map = {
        # "temperature": "#5B9BD5",  # blue
        "temperature": "#000080",  # blue
        "pain_rating": "#FD5030",  # orange
        "pupil_diameter": "#FFC000",  # yellow
        "eda_tonic": "#70AD47",  # green
        "eda_phasic": "#A6A6A6",  # gray
        "heart_rate": "#4BACC6",  # cyan
        "mouth_open": "#9E7BB5",  # purple
    }

    # Plot each signal
    for sig in signals:
        color = color_map.get(sig, plt.cm.tab10(len(signals)))
        signal_label = LABELS.get(sig, sig)

        # Plot the average line
        ax.plot(
            seed_data["normalized_timestamp"],
            seed_data[f"mean_{sig}"],
            label=signal_label,
            color=color,
            alpha=alpha,
            linewidth=1.5,
        )

        # Plot confidence interval if requested
        if show_ci:
            ax.fill_between(
                seed_data["normalized_timestamp"],
                seed_data[f"ci_lower_{sig}"],
                seed_data[f"ci_upper_{sig}"],
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    # Customize plot
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized value")
    ax.set_xlim(
        seed_data["normalized_timestamp"].min(), seed_data["normalized_timestamp"].max()
    )
    ax.grid(True, alpha=0.3)

    # Place legend below the plot
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        framealpha=0.9,
    )

    plt.tight_layout()

    return fig


def plot_averages_with_ci_plt(
    averages_with_ci_df: pl.DataFrame,
    signals: list[str] = None,
    alpha: float = 0.0,
) -> plt.Figure:
    """
    Plot confidence intervals for the given modality for all participants for each stimulus seed.
    """
    # Get unique stimulus seeds
    stimulus_seeds = sorted(averages_with_ci_df["stimulus_seed"].unique())
    n_seeds = len(stimulus_seeds)

    # Create subplot grid (4x3 for 12 plots)
    n_cols = 3
    n_rows = (n_seeds + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, seed in enumerate(stimulus_seeds):
        ax = axes[idx]

        # Filter data for current stimulus seed
        seed_data = averages_with_ci_df.filter(pl.col("stimulus_seed") == seed)

        # Plot each signal
        for sig_idx, sig in enumerate(signals):
            color = SIGNAL_COLORS.get(sig, plt.get_cmap("tab10")(sig_idx % 10))
            alpha = alpha if alpha > 0 else 1.0
            signal_label = LABELS.get(sig, sig)

            # Plot the average line
            ax.plot(
                seed_data["normalized_timestamp"],
                seed_data[f"mean_{sig}"],
                label=signal_label,
                color=color,
                alpha=alpha,
                linewidth=0.9,
            )

            # Plot confidence interval
            ax.fill_between(
                seed_data["normalized_timestamp"],
                seed_data[f"ci_lower_{sig}"],
                seed_data[f"ci_upper_{sig}"],
                color=color,
                alpha=0.15 * alpha,
                linewidth=0,
            )

        # Customize subplot
        ax.set_xlim(
            seed_data["normalized_timestamp"].min(),
            seed_data["normalized_timestamp"].max(),
        )

        # Configure ticks: only show on bottom row and leftmost column
        row = idx // n_cols
        col = idx % n_cols

        # Show x-axis ticks only on bottom row
        if row < n_rows - 1:
            ax.tick_params(bottom=False, labelbottom=False)

        # Show y-axis ticks only on leftmost column
        if col > 0:
            ax.tick_params(left=False, labelleft=False)

    # Remove empty subplots
    for idx in range(n_seeds, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout with more bottom padding for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, left=0.07)  # Add left padding

    # Add single x and y labels to the figure
    fig.text(0.5, 0.05, "Time (s)", ha="center", va="bottom")
    fig.text(
        0.02, 0.55, "Normalized value", ha="center", va="center", rotation="vertical"
    )

    # Add legend to the bottom of the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(signals),
    )

    return fig


def plot_averages_with_ci(
    averages_with_ci_df: pl.DataFrame,
    signals: list[str] = None,
    muted_alpha: float = 0.0,
) -> pl.DataFrame:
    """
    Plot confidence intervals for the given modality for all participants for each stimulus seed.
    """
    # Create plot
    plots = averages_with_ci_df.hvplot(
        x="normalized_timestamp",
        y=[f"mean_{sig}" for sig in signals],
        groupby="stimulus_seed",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
        grid=True,
        muted_alpha=muted_alpha,
    )
    for sig in signals:
        plots *= averages_with_ci_df.hvplot.area(
            x="normalized_timestamp",
            y=f"ci_lower_{sig}",
            y2=f"ci_upper_{sig}",
            groupby="stimulus_seed",
            alpha=0.15,
            line_width=0,
            grid=True,
            muted_alpha=muted_alpha,
            label=f"mean_{sig}",
        )

    return plots


def calculate_crosscorr_matrix(
    averages_df: pl.DataFrame,
    signals: list[str],
    reference_signal: str = "temperature",
    fs: int = 10,
    skip_first_n_seconds: float = 20,
):
    """Calculate cross-correlation lags between reference signal and all other signals."""
    averages_df = averages_df.filter(
        pl.col("normalized_timestamp") >= skip_first_n_seconds * 1000
    )

    results = []

    for sig in signals:
        if sig == reference_signal:
            continue

        col1 = f"mean_{reference_signal}"
        col2 = f"mean_{sig}"

        lag_arr = []
        stimulus_seeds = []

        for stimulus in (
            averages_df.get_column("stimulus_seed")
            .unique(maintain_order=True)
            .to_numpy()
        ):
            col1_arr = averages_df.filter(stimulus_seed=stimulus)[col1].to_numpy()
            col2_arr = averages_df.filter(stimulus_seed=stimulus)[col2].to_numpy()

            # Cross-correlation
            corr = signal.correlate(col1_arr, col2_arr, method="auto")
            lags = signal.correlation_lags(len(col1_arr), len(col2_arr))

            # Find the maximum correlation and the lag
            lag = lags[np.argmax(corr)] / fs  # lag in seconds
            lag_arr.append(lag)
            stimulus_seeds.append(stimulus)

        # Create summary statistics for this signal pair
        lag_arr = np.array(lag_arr)
        mean_lag = np.mean(lag_arr)
        std_lag = np.std(lag_arr)

        # Add to results
        results.append(
            {
                "reference_signal": reference_signal,
                "target_signal": sig,
                "mean_lag": mean_lag,
                "std_lag": std_lag,
                "individual_lags": lag_arr.tolist(),
                "stimulus_seeds": stimulus_seeds,
            }
        )
    return pl.DataFrame(results).sort("mean_lag", descending=True)


def plot_correlation_heatmap(
    averages: pl.DataFrame,
    features: list[str] | None = None,
    skip_first_n_seconds: float = 20,
):
    """Calculated for all stimulus seeds at once, i.e. ignoring stimulus seed."""
    averages = averages.filter(
        col("normalized_timestamp") >= skip_first_n_seconds * 1000
    )

    # Default features if none provided
    if features is None:
        features = [
            "temperature",
            "pain_rating",
            "pupil_diameter",
            "heart_rate",
            "eda_tonic",
            "eda_phasic",
        ]

    # Get correlation matrix and create labels using LABELS dict
    feature_cols = [f"mean_{f}" for f in features]
    corr_matrix = averages.select(feature_cols).corr()

    # Use LABELS dict for better feature names
    labels = [
        LABELS.get(
            col.replace("mean_", ""), col.replace("mean_", "").replace("_", " ").title()
        )
        for col in corr_matrix.columns
    ]

    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Custom colormap and upper triangle mask
    custom_cmap = LinearSegmentedColormap.from_list("blues", ["#e8eef7", "#0033cc"])
    mask = np.triu(np.ones_like(corr_matrix), k=1)

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        cbar_kws={"shrink": 0.8, "label": "Pearson correlation coefficient"},
        linewidths=0.3,
        ax=ax,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig
