import logging

import numpy as np
import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def to_describe(
    column: str,
    prefix: str = "",
) -> list[pl.Expr]:
    """
    Polars helper function for having the describe expression for use in groupby-agg.
    From https://github.com/pola-rs/polars/issues/8066#issuecomment-1794144838

    Example:
    ````python
    out = stimuli.group_by("seed", maintain_order=True).agg(
        *to_describe("y", prefix="temperature_"),
        *to_describe("time"),  # using list unpacking to pass multiple expressions
    )
    ````
    """
    prefix = prefix or f"{column}_"
    return [
        col(column).count().alias(f"{prefix}count"),
        col(column).is_null().sum().alias(f"{prefix}null_count"),
        col(column).mean().alias(f"{prefix}mean"),
        col(column).std().alias(f"{prefix}std"),
        col(column).min().alias(f"{prefix}min"),
        col(column).quantile(0.25).alias(f"{prefix}25%"),
        col(column).quantile(0.5).alias(f"{prefix}50%"),
        col(column).quantile(0.75).alias(f"{prefix}75%"),
        col(column).max().alias(f"{prefix}max"),
    ]


def check_sample_rate(
    df: pl.DataFrame,
    unique_timestamp: bool = False,
) -> None:
    r"""coefficient of variation between trials
    # $ CV=\frac{\sigma}{\mu} $
    """
    df = df.filter(col("trial_id") > 0)
    if unique_timestamp:
        df = df.unique("timestamp").sort("timestamp")
        logger.info("Checking sample rate for unique timestamps.")

    timestamp_start = (
        df.group_by("trial_id", maintain_order=True)
        .agg(pl.first("timestamp"))
        .sort("trial_id")
        .select("timestamp")
    )
    timestamp_end = (
        df.group_by("trial_id", maintain_order=True)
        .agg(pl.last("timestamp"))
        .sort("trial_id")
        .select("timestamp")
    )

    duration_in_s = (timestamp_end - timestamp_start) / 1000

    samples = (
        df.group_by("trial_id", maintain_order=True)
        .agg(pl.count("timestamp"))
        .sort("trial_id")
        .select("timestamp")
    )

    sample_rate_per_trial = samples / duration_in_s
    sample_rate_mean = (sample_rate_per_trial).mean().item()
    coeff_of_variation = (
        (sample_rate_per_trial).std() / (sample_rate_per_trial).mean() * 100
    ).item()

    logger.debug(
        "Sample rate per trial: %s",
        np.round(sample_rate_per_trial.to_numpy().flatten(), 2),
    )
    logger.info(f"The mean sample rate is {(sample_rate_mean):.2f}.")
    if coeff_of_variation and coeff_of_variation > 0.5:
        logger.warning(
            "Sample rate varies more than 0.5% between trials: "
            f"{coeff_of_variation:.2f}% (coefficient of variation)."
        )
