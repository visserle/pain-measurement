import logging
import operator
from functools import reduce

import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def create_samples(
    df: pl.DataFrame,
    intervals: dict[str, str],
    label_mapping: dict[str, int],
    length_ms: int = 5000,
    offsets_ms: dict[str, int] | None = None,
):
    """
    Create samples from a DataFrame with for different stimulus intervals (see labeling
    section of the StimulusGenerator).
    Only the first x milliseconds of each interval are used, with optional offsets.

    Note: Needs a column "normalized_timestamp" in the DataFrame.

    Args:
        df: DataFrame with stimulus data
        intervals: Dictionary mapping interval names to column names
        length_ms: Length of each sample in milliseconds
        label_mapping: Optional mapping of interval names to labels
        offsets: Dictionary mapping interval names to offset values in milliseconds.
                If None, no offset is applied. If an interval name is not in the dictionary,
                no offset is applied for that interval.
    """
    if "normalized_timestamp" not in df.columns:
        raise ValueError(
            "DataFrame must contain a column 'normalized_timestamp "
            "(e.g. src.features.resampling.add_normalized_timestamp)'"
        )

    if len(intervals) < 2:
        raise ValueError(
            "At least two interval types are required for sample creation."
        )

    # Initialize offsets with default values (0) if not provided
    if offsets_ms is None:
        offsets_ms = {name: 0 for name in intervals.keys()}
    else:
        # Ensure all intervals have an offset value (default 0)
        for name in intervals.keys():
            if name not in offsets_ms:
                offsets_ms[name] = 0

    samples = _cap_intervals_to_sample_length(df, intervals, length_ms, offsets_ms)
    samples = _generate_sample_ids(samples, intervals, label_mapping)
    # samples = _remove_not_matching_samples(samples)

    # Make sure we kept equidistant sampling with a sampling rate of 10 Hz
    if (
        samples.filter(col("normalized_timestamp") % 10 != 0)
        .get_column("normalized_timestamp")
        .len()
    ):
        logger.warning("Sampling rate is not equidistant with 10 Hz.")

    return samples


def _cap_intervals_to_sample_length(
    df: pl.DataFrame,
    intervals: dict[str, str],
    length_ms: int,
    offsets_ms: dict[str, int],
):
    """
    Cap samples to the first x milliseconds of each interval, with optional offsets.

    Uses a half-open interval [offset, offset + length) for each interval.

    Args:
        df: DataFrame with stimulus data
        intervals: Dictionary mapping interval names to column names
        length_ms: Length of each sample in milliseconds
        offsets_ms: Dictionary mapping interval names to offset values in milliseconds
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

    # Apply offsets and only use the length_ms duration of each interval
    condition = [
        pl.when(
            (col(f"normalized_timestamp_{name}") >= offsets_ms[name])
            & (col(f"normalized_timestamp_{name}") < offsets_ms[name] + length_ms)
        )
        .then(col(f"normalized_timestamp_{name}"))
        .otherwise(None)
        .alias(f"normalized_timestamp_{name}")
        for name in intervals.keys()
    ]
    samples = samples.with_columns(condition)

    # Filter out all data points that are not in the desired range
    condition = reduce(
        operator.or_,
        [
            (col(f"normalized_timestamp_{name}") >= offsets_ms[name])
            & (col(f"normalized_timestamp_{name}") < offsets_ms[name] + length_ms)
            for name in intervals.keys()
        ],
    )
    samples = samples.filter(condition)

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
    label_mapping: dict[str, int],
):
    """
    Generate consecutive sample IDs across any number of interval types.

    Args:
        samples: DataFrame containing interval data
        intervals: Dictionary mapping interval names to their column names
        label_mapping: Dictionary mapping interval names to their label values.
    """
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


def _remove_not_matching_samples(
    samples: pl.DataFrame,
):
    """
    Remove samples that do not match the criteria. E.g. samples that are too short.

    This is important only if the our sample length is greater than 5000 ms as not all
    intervals are longer than 5000 ms. In this case, we need to remove samples that are
    shorter than 5000 ms.
    """
    pass  # not implemented yet as we work with 5000 ms samples only


def make_sample_set_balanced(
    samples: pl.DataFrame,
    random_seed: int,
):
    """
    Make a sample set balanced by downsampling the majority class in the dataset to the
    size of the minority class to prevent the model from inclining towards the majority
    class. Shuffles data before downsampling to ensure random selection.
    """
    sample_length = samples.filter(
        col("sample_id") == samples.get_column("sample_id").first()
    ).height

    # Calculate samples per label and find minimum
    label_counts = samples.get_column("label").value_counts().sort("label")
    samples_per_label = label_counts.with_columns(col("count") // sample_length).sort(
        "label"
    )
    min_label_count = samples_per_label.get_column("count").min()

    # Calculate how many samples to remove from each group
    samples_to_remove = samples_per_label.with_columns(col("count") - min_label_count)

    # Balance the dataset by reducing larger groups to match smallest group
    balanced_groups = []
    for remove_count, (label, group) in zip(
        samples_to_remove.get_column("count"), samples.group_by("label")
    ):
        if remove_count == 0:
            # Keep group as is if it's already at the minimum size
            balanced_groups.append(group)
        else:
            # Shuffle the group and then reduce size to match the smallest group
            shuffled_group = group.sample(fraction=1.0, seed=random_seed)
            reduced_group = shuffled_group.limit(min_label_count * sample_length)
            balanced_groups.append(reduced_group)

    # Combine all balanced groups and sort by sample_id
    return pl.concat(balanced_groups).sort("sample_id")
