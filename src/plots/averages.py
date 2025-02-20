"""
Plot confidence intervals for the given modality for all participants for each stimulus seed.

Uses 95% confidence interval (1.96) for the mean of a normal distribution.
(95% chance your population mean will fall between lower and upper limit)
"""

import logging

import hvplot.polars  # noqa
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.stats as stats
from polars import col
from scipy import signal

from src.features.resampling import add_normalized_timestamp, add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_robust_standard, scale_standard

BIN_SIZE = 0.1  # in seconds
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval


logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def average_over_stimulus_seeds(
    df: pl.DataFrame,
    signals: list[str],
    scaling: str | None = "min_max",
    bin_size: int = BIN_SIZE,
) -> pl.DataFrame:
    """Aggregate over stimulus seeds for each trial using group_by_dynamic.
    Using scaling, the data can be scaled before aggregation.
    """
    match scaling:
        case "min_max":
            df = scale_min_max(
                df,
                exclude_additional_columns=[
                    "rating",  # already normalized
                    "temperature",  # already normalized
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
        y=[f"avg_{signal}" for signal in signals],
        groupby="stimulus_seed",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
        grid=True,
        muted_alpha=muted_alpha,
    )
    for signal in signals:
        plots *= averages_with_ci_df.hvplot.area(
            x="time_bin",
            y=f"ci_lower_{signal}",
            y2=f"ci_upper_{signal}",
            groupby="stimulus_seed",
            alpha=0.15,
            line_width=0,
            grid=True,
            muted_alpha=muted_alpha,
            label=f"avg_{signal}",
        )

    return plots


def calculate_max_crosscorr_lag_over_averages(
    averages_df: pl.DataFrame,
    col1: str,
    col2: str,
    fs: int = 10,
    plot: bool = False,
):
    lag_arr = []
    stimulus_seed = []
    for stimulus in (
        averages_df.get_column("stimulus_seed").unique(maintain_order=True).to_numpy()
    ):
        temperature = averages_df.filter(stimulus_seed=stimulus)[col1].to_numpy()
        rating = averages_df.filter(stimulus_seed=stimulus)[col2].to_numpy()

        # Cross-correlation
        corr = signal.correlate(
            temperature,
            rating,
            method="auto",
        )
        # Lag indices for the cross-correlation
        lags = signal.correlation_lags(
            len(temperature),
            len(rating),
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
