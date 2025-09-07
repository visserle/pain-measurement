"""
Global average over stimulus seeds for each trial.
The data can be scaled before aggregation.
Time binning is applied to the data.
Confidence intervals are calculated for visualization.
Cross-correlation is calculated for the given modality.
"""

import logging

import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from polars import col
from scipy import signal

from src.features.resampling import add_normalized_timestamp, add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_robust_standard, scale_standard

plt.style.use("./src/plots/style.mplstyle")

BIN_SIZE = 0.1  # in seconds
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval
# (95% chance your population mean will fall between lower and upper limit)
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


logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def average_over_stimulus_seeds(
    df: pl.DataFrame,
    signals: list[str],
    scaling: str | None = "min_max",
    bin_size: int | float = BIN_SIZE,
) -> pl.DataFrame:
    """Aggregate over stimulus seeds for each trial using group_by_dynamic.
    Using scaling, the data can be scaled before aggregation.
    """
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

    # Zero-based timestamp in milliseconds
    df = add_normalized_timestamp(df)
    # Add microsecond timestamp column for better precision as group_by_dynamic uses int
    df = add_timestamp_μs_column(df, "normalized_timestamp")

    # Time binning
    return (
        (
            df.sort("normalized_timestamp_µs")
            # Note: without group_by_dynamic, this would be something like
            # ````
            # df.with_columns(
            #   [(col("normalized_timestamp") // 1000).cast(pl.Int32).alias("time_bin")]
            #   )
            #   .group_by(["stimulus_seed", "time_bin"])
            # ````
            .group_by_dynamic(
                "normalized_timestamp_µs",
                every=f"{int((1000 / (1 / bin_size)) * 1000)}i",
                group_by=["stimulus_seed"],
            )
            .agg(
                # Average and standard deviation for each signal
                [
                    col(signal).mean().alias(f"avg_{signal.lower()}")
                    for signal in signals
                ]
                + [
                    col(signal).std().alias(f"std_{signal.lower()}")
                    for signal in signals
                ]
                # Sample size for each bin
                + [pl.len().alias("sample_size")]
            )
        )
        .with_columns((col("normalized_timestamp_µs") / 1_000_000).alias("time_bin"))
        .sort("stimulus_seed", "time_bin")
        # remove measures at exactly 180s so that they don't get their own bin
        .filter(col("time_bin") < 180)
        .drop("normalized_timestamp_µs")
    )


def add_ci_to_averages(
    averages_df: pl.DataFrame,
    signals: str,
    min_sample_size: int = 30,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pl.DataFrame:
    """
    Create confidence intervals for visualization.
    """
    small_samples = averages_df.filter(col("sample_size") < min_sample_size)
    if small_samples.height > 0:
        logger.warn(
            f"Warning: {small_samples.height} bins have sample size < {min_sample_size}"
        )

    z_score = _calculate_z_score(confidence_level)

    return averages_df.with_columns(
        [
            (
                col(f"avg_{signal}")
                - z_score * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_lower_{signal}")
            for signal in signals
        ]
        + [
            (
                col(f"avg_{signal}")
                + z_score * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_upper_{signal}")
            for signal in signals
        ]
    ).sort("stimulus_seed", "time_bin")


def _calculate_z_score(confidence_level: float) -> float:
    """
    Calculate z-score for the given confidence level (e.g., 0.95 -> 1.96).
    """
    return stats.norm.ppf((1 + confidence_level) / 2).round(2)


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

    # Color palette for different signals
    colors = plt.cm.tab10(np.linspace(0, 1, len(signals)))

    for idx, seed in enumerate(stimulus_seeds):
        ax = axes[idx]

        # Filter data for current stimulus seed
        seed_data = averages_with_ci_df.filter(pl.col("stimulus_seed") == seed)

        # Plot each signal
        for sig_idx, sig in enumerate(signals):
            color = colors[sig_idx]
            alpha = alpha if alpha > 0 else 1.0
            signal_label = LABELS.get(sig, sig)

            # Plot the average line
            ax.plot(
                seed_data["time_bin"],
                seed_data[f"avg_{sig}"],
                label=signal_label,
                color=color,
                alpha=alpha,
                linewidth=0.9,
            )

            # Plot confidence interval
            ax.fill_between(
                seed_data["time_bin"],
                seed_data[f"ci_lower_{sig}"],
                seed_data[f"ci_upper_{sig}"],
                color=color,
                alpha=0.15 * alpha,
                linewidth=0,
            )

        # Customize subplot
        ax.set_xlim(seed_data["time_bin"].min(), seed_data["time_bin"].max())

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
        x="time_bin",
        y=[f"avg_{sig}" for sig in signals],
        groupby="stimulus_seed",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
        grid=True,
        muted_alpha=muted_alpha,
    )
    for sig in signals:
        plots *= averages_with_ci_df.hvplot.area(
            x="time_bin",
            y=f"ci_lower_{sig}",
            y2=f"ci_upper_{sig}",
            groupby="stimulus_seed",
            alpha=0.15,
            line_width=0,
            grid=True,
            muted_alpha=muted_alpha,
            label=f"avg_{sig}",
        )

    return plots


def calculate_crosscorr_matrix(
    averages_df: pl.DataFrame,
    signals: list[str],
    reference_signal: str = "temperature",
    fs: int = 10,
):
    """Calculate cross-correlation lags between reference signal and all other signals."""
    results = []

    for sig in signals:
        if sig == reference_signal:
            continue

        col1 = f"avg_{reference_signal}"
        col2 = f"avg_{sig}"

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
):
    """Calculated for all stimulus seeds at once, i.e. ignoring stimulus seed."""
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
    feature_cols = [f"avg_{f}" for f in features]
    corr_matrix = averages.select(feature_cols).corr()

    # Use LABELS dict for better feature names
    labels = [
        LABELS.get(
            col.replace("avg_", ""), col.replace("avg_", "").replace("_", " ").title()
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
