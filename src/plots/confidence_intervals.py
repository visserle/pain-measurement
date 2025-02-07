import hvplot.polars
import polars as pl
from icecream import ic
from polars import col

from src.features.resampling import add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_robust_standard, scale_standard

BIN_SIZE = 1  # seconds
CONFIDENCE_LEVEL = 1.96  # 95% confidence interval


def plot_confidence_intervals(
    df: pl.DataFrame,
    signals: list[str] = None,
) -> pl.DataFrame:
    """
    Plot confidence intervals for the given modality for all participants for each stimulus seed.
    """
    # Create plot
    plots = df.hvplot(
        x="time_bin",
        y=[f"avg_{signal}" for signal in signals],
        groupby="stimulus_seed",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
        grid=True,
    )
    for signal in signals:
        plots *= df.hvplot.area(
            x="time_bin",
            y=f"ci_lower_{signal}",
            y2=f"ci_upper_{signal}",
            groupby="stimulus_seed",
            alpha=0.2,
            line_width=0,
            grid=True,
        )

    return plots


def prepare_confidence_intervals(
    df: pl.DataFrame,
    signals: list[str],
    bin_size: int = BIN_SIZE,
) -> pl.DataFrame:
    """
    Prepare data with confidence intervals for visualization.
    """
    # Scale data for better visualization (mapped over trial_id)
    df = scale_min_max(
        df,
        exclude_additional_columns=[
            "rating",  # already normalized
            "temperature",  # already normalized
        ],
    )

    # Calculate confidence intervals
    df = aggregate_over_stimulus_seeds(df, signals, bin_size=bin_size)
    return add_confidence_interval(df, signals)


def aggregate_over_stimulus_seeds(
    df: pl.DataFrame,
    signals: list[str],
    bin_size: int = BIN_SIZE,
) -> pl.DataFrame:
    """Aggregate over stimulus seeds for each trial using group_by_dynamic."""
    # Note: without group_by_dynamic, this would be something like
    # ````
    # df.with_columns(
    #     [(col("zeroed_timestamp") // 1000).cast(pl.Int32).alias("time_bin")]
    #     )
    #     .group_by(["stimulus_seed", "time_bin"])
    # ````

    # Zero-based timestamp in milliseconds
    df = _zero_based_timestamps(df)
    # Add microsecond timestamp column for better precision as group_by_dynamic uses int
    df = add_timestamp_μs_column(df, "zeroed_timestamp")
    # Time binning
    return (
        (
            df.sort("zeroed_timestamp_µs")
            .group_by_dynamic(
                "zeroed_timestamp_µs",
                every=f"{int((1000 / (1/bin_size))*1000)}i",
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
        .with_columns((col("zeroed_timestamp_µs") / 1_000_000).alias("time_bin"))
        .sort("stimulus_seed", "time_bin")
        # remove measures at exactly 180s so that they don't get their own bin
        .filter(col("time_bin") < 180)
        .drop("zeroed_timestamp_µs")
    )


def _zero_based_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (col("timestamp") - col("timestamp").min().over("trial_id")).alias(
            "zeroed_timestamp"
        )
    )


def add_confidence_interval(
    df: pl.DataFrame,
    signals: str,
) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                col(f"avg_{signal}")
                - CONFIDENCE_LEVEL * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_lower_{signal}")
            for signal in signals
        ]
        + [
            (
                col(f"avg_{signal}")
                + CONFIDENCE_LEVEL * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_upper_{signal}")
            for signal in signals
        ]
    ).sort("stimulus_seed", "time_bin")
