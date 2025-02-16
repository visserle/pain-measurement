"""
Plot confidence intervals for the given modality for all participants for each stimulus seed.

Uses 95% confidence interval (1.96) for the mean of a normal distribution.
(95% chance your population mean will fall between lower and upper limit)
"""

import logging

import hvplot.polars  # noqa
import polars as pl
import scipy.stats as stats
from polars import col

from src.features.resampling import add_normalized_timestamp, add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_robust_standard, scale_standard

BIN_SIZE = 0.1  # in seconds
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval


logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def plot_confidence_intervals(
    confidence_intervals_df: pl.DataFrame,
    signals: list[str] = None,
) -> pl.DataFrame:
    """
    Plot confidence intervals for the given modality for all participants for each stimulus seed.
    """
    # Create plot
    plots = confidence_intervals_df.hvplot(
        x="time_bin",
        y=[f"avg_{signal}" for signal in signals],
        groupby="stimulus_seed",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
        grid=True,
    )
    for signal in signals:
        plots *= confidence_intervals_df.hvplot.area(
            x="time_bin",
            y=f"ci_lower_{signal}",
            y2=f"ci_upper_{signal}",
            groupby="stimulus_seed",
            alpha=0.2,
            line_width=0,
            grid=True,
        )

    return plots


def create_confidence_intervals(
    df: pl.DataFrame,
    signals: list[str],
    bin_size: int = BIN_SIZE,
    scaling: str | None = "min_max",
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pl.DataFrame:
    """
    Create confidence intervals for visualization.
    Using scaling, the data can be scaled before calculating the confidence intervals.

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

    # Create confidence intervals
    df = aggregate_over_stimulus_seeds(df, signals, bin_size=bin_size)
    return calculate_confidence_intervals(
        df,
        signals,
        min_sample_size=30,
        confidence_level=confidence_level,
    )


def aggregate_over_stimulus_seeds(
    df: pl.DataFrame,
    signals: list[str],
    bin_size: int = BIN_SIZE,
) -> pl.DataFrame:
    """Aggregate over stimulus seeds for each trial using group_by_dynamic."""

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


def calculate_confidence_intervals(
    df: pl.DataFrame,
    signals: str,
    min_sample_size: int = 30,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> pl.DataFrame:
    small_samples = df.filter(col("sample_size") < min_sample_size)
    if small_samples.height > 0:
        logger.warn(
            f"Warning: {small_samples.height} bins have sample size < {min_sample_size}"
        )

    z_score = _calculate_z_score(confidence_level)

    return df.with_columns(
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
