import hvplot.polars  # noqa
import polars as pl
from polars import col

from src.data.database_manager import DatabaseManager
from src.features.resampling import add_timestamp_μs_column
from src.features.scaling import scale_min_max
from src.features.transforming import merge_data_dfs

MODALITY_MAP = {
    "stimulus": ["rating", "temperature"],
    "eda": ["eda_tonic", "eda_phasic"],
}


def plot_confidence_interval(modality: str):
    signals = MODALITY_MAP[modality]
    # As we plot for each stimulus seed, we need trial metadata first
    with DatabaseManager() as db:
        df = db.get_table("feature_" + modality)
        trials = db.get_table("trials")  # get trials for stimulus seeds
    df = merge_data_dfs(
        [df, trials],
        merge_on=["participant_id", "trial_id", "trial_number"],
    ).drop("duration", "skin_area", "timestamp_start", "timestamp_end", strict=False)

    #
    df = aggregate_over_seeds(df, modality, bin_size=1)
    # scale for better visualization, must come before adding confidence intervals
    df = scale_min_max(
        df, exclude_additional_columns=["time_bin", "rating", "temperature"]
    )
    df = add_confidence_interval(df, modality)

    # Create plot
    plots = df.plot(
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


def _zero_based_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (col("timestamp") - col("timestamp").min().over("trial_id")).alias(
            "zeroed_timestamp"
        )
    )


def aggregate_over_seeds(
    df: pl.DataFrame,
    modality: str,
    bin_size: int = 1,  # TODO
) -> pl.DataFrame:
    """Aggregate over seeds for each trial using group_by_dynamic."""
    # Note: without group_by_dynamic, this would be something like
    # >>> df.with_columns(
    # >>>     [(col("zeroed_timestamp") // 1000).cast(pl.Int32).alias("time_bin")]
    # >>>     )
    # >>>     .group_by(["stimulus_seed", "time_bin"])

    # Select signals for the given modality
    modality = modality.lower()
    signals = [signal for signal in MODALITY_MAP[modality] if signal in df.columns]

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
                # TODO find out why the sample sizes are not constant
            )
        )
        .with_columns((col("zeroed_timestamp_µs") / 1_000_000).alias("time_bin"))
        .sort("stimulus_seed", "time_bin")
        # remove measures at exactly 180s so that they don't get their own bin
        .filter(col("time_bin") < 180)
        .drop("zeroed_timestamp_µs")
    )


def add_confidence_interval(
    df: pl.DataFrame,
    modality: str,
) -> pl.DataFrame:
    return df.with_columns(
        [
            (
                col(f"avg_{signal}")
                - 1.96 * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_lower_{signal}")
            for signal in MODALITY_MAP[modality]
        ]
        + [
            (
                col(f"avg_{signal}")
                + 1.96 * (col(f"std_{signal}") / col("sample_size").sqrt())
            ).alias(f"ci_upper_{signal}")
            for signal in MODALITY_MAP[modality]
        ]
    ).sort("stimulus_seed", "time_bin")
