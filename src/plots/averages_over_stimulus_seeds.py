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

BIN_SIZE = 0.1  # in seconds
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval
# (95% chance your population mean will fall between lower and upper limit)


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
    muted_alpha: float = 0.0,
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
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
            alpha = muted_alpha if muted_alpha > 0 else 1.0

            # Plot the average line
            ax.plot(
                seed_data["time_bin"],
                seed_data[f"avg_{sig}"],
                label=sig,
                color=color,
                alpha=alpha,
                linewidth=2,
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
        ax.set_title(f"Stimulus {seed}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(seed_data["time_bin"].min(), seed_data["time_bin"].max())

        # Add labels to edge plots only
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Time (s)", fontsize=10)
        if idx % n_cols == 0:
            ax.set_ylabel("Normalized value", fontsize=10)

        # Add legend to first subplot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Remove empty subplots
    for idx in range(n_seeds, len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout
    plt.tight_layout()

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


def calculate_max_crosscorr_lag_over_averages(
    averages_df: pl.DataFrame,
    col1: str,
    col2: str,
    fs: int = 10,
    plot: bool = False,
):
    # Note that normalizing cross correlations for intpretation is done by dividing by
    # the maximum value of the cross-correlation so that the maximum value is 1.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html
    lag_arr = []
    stimulus_seed = []
    for stimulus in (
        averages_df.get_column("stimulus_seed").unique(maintain_order=True).to_numpy()
    ):
        col1_arr = averages_df.filter(stimulus_seed=stimulus)[col1].to_numpy()
        col2_arr = averages_df.filter(stimulus_seed=stimulus)[col2].to_numpy()

        # Cross-correlation
        corr = signal.correlate(
            col1_arr,
            col2_arr,
            method="auto",
        )
        # Lag indices for the cross-correlation
        lags = signal.correlation_lags(
            len(col1_arr),
            len(col2_arr),
        )
        if plot:
            plt.plot(lags, corr)

        # Find the maximum correlation and the lag
        lag = lags[np.argmax(corr)] / fs  # lag in seconds
        lag_arr.append(lag)
        stimulus_seed.append(stimulus)

    lag_arr = np.array(lag_arr)
    lag_df = pl.DataFrame({"stimulus_seed": stimulus_seed, "lag": lag_arr})
    logger.info(
        f"{col1} : {col2} | mean lag: {lag_df['lag'].mean():.2f}, std lag: {lag_df['lag'].std():.2f}"
    )
    return lag_df


def plot_correlation_heatmap(
    averages: pl.DataFrame,
    features: list[str] | None = None,
):
    # Default feature list if none provided
    if features is None:
        features = [
            "avg_temperature",
            "avg_pain_rating",
            "avg_pupil_diameter",
            "avg_eda_tonic",
            "avg_heart_rate",
            "avg_eda_phasic",
        ]
    else:
        features = [f"avg_{feature}" for feature in features]

    # Get correlation matrix
    corr_matrix = (
        averages.select(pl.col("^avg.*$"))
        .select(features)  # Use the features parameter
        .corr()
    )
    # Create more professional labels
    labels = [
        col.replace("avg_", "").replace("_", " ").title() for col in corr_matrix.columns
    ]

    # Create a custom colormap
    dark_blue = "#0033cc"
    light_blue = "#e8eef7"
    colors = [light_blue, dark_blue]
    custom_cmap = LinearSegmentedColormap.from_list("dark_blues", colors, N=256)

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix), k=1)

    # Set blue as the main color for the plot styling
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"

    # Create the figure with higher DPI for better quality
    fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="white")

    # Plot the heatmap with color styling
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        cmap=custom_cmap,
        vmin=0,  # Set minimum to 0 since all correlations are positive
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        annot_kws={"size": 14, "weight": "bold", "color": "black"},
        cbar_kws={"shrink": 0.8, "label": "Correlation Strength"},
        linewidths=0.5,
        linecolor="white",
    )

    # Access the colorbar and set its label font properties
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel("Correlation Strength", fontsize=14, fontweight="bold")

    # Set annotation colors based on background darkness for better readability
    for text in heatmap.texts:
        value = float(text.get_text())
        text.set_color("black" if value < 0.6 else "white")

    # Adjust font sizes and styles for conference poster visibility
    plt.xticks(rotation=45, ha="right", fontsize=14, fontweight="bold")
    plt.yticks(rotation=0, fontsize=14, fontweight="bold")

    # Add a border to the heatmap
    for _, spine in heatmap.spines.items():
        spine.set_visible(False)
        spine.set_linewidth(2)

    # Ensure everything fits within the figure bounds
    plt.tight_layout()

    return fig
