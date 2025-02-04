import numpy as np
import polars as pl
from polars import col


def create_samples(
    df: pl.DataFrame,
    intervals: dict[str, str] = {
        "increases": "strictly_increasing_intervals_without_plateaus",
        "decreases": "decreasing_intervals",
    },
    length_ms: int = 5000,
):
    """
    Create samples from a DataFrame with for decreasing and increasing intervals.
    Only the first 5 seconds of each interval are kept.

    Note: Needs a column "normalized_timestamp" in the DataFrame.
    """
    samples = _cap_samples(df, intervals, length_ms)
    samples = _generate_sample_ids(samples)

    return samples


def _cap_samples(
    df: pl.DataFrame,
    intervals: dict[str, str],
    length_ms: int,
):
    """
    Cap samples to the first 5 seconds of each interval.
    """
    # Add time counter for each relevant interval
    condition = [
        pl.when(col(interval_col) != 0)
        .then(
            # don't use plain timestamp, will lead to floating point weirdness
            col("normalized_timestamp")
            - col("normalized_timestamp").min().over(interval_col)
        )
        .otherwise(None)
        .alias(f"normalized_timestamp_{name}")
        for name, interval_col in intervals.items()
    ]
    samples = df.with_columns(condition)

    # Only use the first 5 seconds
    # Need two separate filters because else we would keep timestamps that are not
    # in the first 5 seconds but are in the first 5 seconds of the other interval
    condition = [
        pl.when(col(f"normalized_timestamp_{name}") < length_ms)
        .then(col(f"normalized_timestamp_{name}"))
        .otherwise(None)
        .alias(f"normalized_timestamp_{name}")
        for name in intervals.keys()
    ]
    samples = samples.with_columns(condition).filter(
        (col("normalized_timestamp_increases") < length_ms)
        | (col("normalized_timestamp_decreases") < length_ms)
    )
    # We also need to filter out all interval values similar to the step above
    condition = [
        pl.when(col(f"normalized_timestamp_{name}").is_null())
        .then(0)
        .otherwise(col(interval_col))
        .alias(interval_col)
        for name, interval_col in intervals.items()
    ]
    samples = samples.with_columns(condition)
    return samples


def _generate_sample_ids(
    samples: pl.DataFrame,
):
    # Split data into decreasing and increasing intervals to add labels and sample ids
    decreases = samples.filter(
        col("normalized_timestamp_decreases").is_not_null()
    ).with_columns(
        pl.lit(1).alias("label").cast(pl.UInt8),
        col("decreasing_intervals").alias("sample_id"),
    )
    increases = samples.filter(
        col("normalized_timestamp_increases").is_not_null()
    ).with_columns(
        pl.lit(0).alias("label").cast(pl.UInt8),
        (
            col("strictly_increasing_intervals")
            + (
                decreases.select(pl.last("decreasing_intervals"))
            )  # continue from decreases
        ).alias("sample_id"),
    )
    # Join the two tables back together
    samples = decreases.vstack(increases).sort("sample_id", "timestamp")

    # Make sure we kept equidistant sampling
    assert (
        not samples.filter(col("normalized_timestamp") % 10 != 0)
        .get_column("normalized_timestamp")
        .len()
    )

    # Join the two tables back together
    samples = decreases.vstack(increases).sort("sample_id", "timestamp")
    return samples
