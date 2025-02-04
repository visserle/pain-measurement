import polars as pl
from polars import col


def create_samples(
    df: pl.DataFrame,
    from_intervals: dict[str, str] = {
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
    samples = _cap_intervals_to_sample_length(df, from_intervals, length_ms)
    samples = _generate_sample_ids(samples, from_intervals)

    # Make sure we kept equidistant sampling with a sampling rate of 10 Hz
    assert (
        not samples.filter(col("normalized_timestamp") % 10 != 0)
        .get_column("normalized_timestamp")
        .len()
    )

    return samples


def _cap_intervals_to_sample_length(
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

    # Only use the first x milliseconds of each interval
    # Need two separate filters because else we would keep timestamps that are not
    # in the first x milliseconds but are in the first x milliseconds of the other
    # interval
    condition = [
        pl.when(col(f"normalized_timestamp_{name}") <= length_ms)
        .then(col(f"normalized_timestamp_{name}"))
        .otherwise(None)
        .alias(f"normalized_timestamp_{name}")
        for name in intervals.keys()
    ]
    samples = samples.with_columns(condition).filter(
        (col("normalized_timestamp_increases") <= length_ms)
        | (col("normalized_timestamp_decreases") <= length_ms)
    )
    # We also need to filter out all wrong interval values similar to the step above
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
    intervals: dict[str, str],
    label_mapping: dict[str, int] | None = None,
):
    """
    Generate consecutive sample IDs across any number of interval types.

    Args:
        samples: DataFrame containing interval data
        intervals: Dictionary mapping interval names to their column names
        label_mapping: Dictionary mapping interval names to their label values.
            If None, uses default mapping: {"increases": 0, "decreases": 1}
    """
    # Use default mapping if none provided
    if label_mapping is None:
        label_mapping = {"increases": 0, "decreases": 1}

    sample_dfs = []
    cumulative_count = 0

    # Process each interval type
    for name, interval_col in intervals.items():
        if name not in label_mapping:
            raise ValueError(f"No label mapping provided for interval type: {name}")

        # Filter samples for current interval type
        interval_df = samples.filter(
            col(f"normalized_timestamp_{name}").is_not_null()
        ).with_columns(
            [
                pl.lit(label_mapping[name]).alias("label").cast(pl.UInt8),
                # Add cumulative count to maintain consecutive IDs
                (col(interval_col) + cumulative_count).alias("sample_id"),
            ]
        )

        # Update cumulative count for next interval type
        if not interval_df.is_empty():
            cumulative_count = interval_df.select(pl.max("sample_id")).item()

        sample_dfs.append(interval_df)

    # Combine all interval DataFrames and sort
    samples = pl.concat(sample_dfs).sort("sample_id", "timestamp")
    return samples
