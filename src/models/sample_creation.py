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
) -> pl.DataFrame:
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
    samples = _remove_samples_that_are_too_short(samples, length_ms)
    samples = _remove_samples_from_stimulus_start(samples, label_mapping)

    return samples


def _cap_intervals_to_sample_length(
    df: pl.DataFrame,
    intervals: dict[str, str],
    length_ms: int,
    offsets_ms: dict[str, int],
) -> pl.DataFrame:
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
) -> pl.DataFrame:
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


def _remove_samples_that_are_too_short(
    samples: pl.DataFrame,
    length_ms: int = 5000,
) -> pl.DataFrame:
    is_equidistant = not (
        samples.filter(col("sample_id") == col("sample_id").first())
        .select(col("normalized_timestamp").diff().diff().cast(pl.Boolean))
        .select(pl.any("normalized_timestamp"))
        .item()
    )

    # Only EEG data is not perfectly equidistant and needs special handling
    # NOTE: This is all hardcoded to work with 5000 ms samples.
    if not is_equidistant:
        assert length_ms == 5000, (
            "Only 5000 ms samples are supported for EEG data as of now. "
            "Please adjust the code if you want to use different sample lengths."
        )
        logger.warning("Sampling rate is not equidistant with 10 Hz.")
        # Fix EEG samples to ensure consistent length
        new = []
        for sample in samples.group_by("sample_id", maintain_order=True):
            sample = sample[1]
            sample = sample.head(750)  # 250 * 3 = 750
            while sample.height < 750:
                sample = pl.concat([sample, sample.tail(1)])
            new.append(sample)

        return pl.concat(new)

    # Calculate the minimum length of samples by subtracting the sampling distance
    # from the length (else we would filter out all samples)
    sampling_distance_ms = (
        samples.filter(col("sample_id") == col("sample_id").first())
        .select(col("normalized_timestamp").diff().last())
        .item()
    )
    min_length_ms = length_ms - sampling_distance_ms

    # Group by sample_id to calculate the duration of each sample
    sample_durations = samples.group_by("sample_id").agg(
        (pl.max("normalized_timestamp") - pl.min("normalized_timestamp")).alias(
            "duration"
        )
    )

    # Get sample_ids that meet the minimum length requirement
    valid_sample_ids = sample_durations.filter(
        pl.col("duration") >= min_length_ms
    ).get_column("sample_id")

    # Filter the original samples DataFrame to keep only valid samples
    filtered_samples = samples.filter(
        pl.col("sample_id").is_in(valid_sample_ids.to_list())
    )

    removed_count = samples.get_column("sample_id").n_unique() - valid_sample_ids.len()
    if removed_count > 0:
        logger.debug(
            f"Removed {removed_count} samples with less than {int(min_length_ms / 100)} data points."
        )

    return filtered_samples


def _remove_samples_from_stimulus_start(
    samples: pl.DataFrame,
    label_mapping: dict[str, int],
) -> pl.DataFrame:
    keep_ids = []
    # Stimulus always starts with an increase
    for group in samples.filter(label=label_mapping["increases"]).group_by("trial_id"):
        group = group[1]
        sample_ids = group.get_column("sample_id").unique().sort()
        # only keep samples that do not start the trial (normalized timestamp != 0)
        keep_ids_non_start = (
            group.filter(col("normalized_timestamp") == 0)
            .get_column("sample_id")
            .unique()
        )
        group = group.filter(~col("sample_id").is_in(keep_ids_non_start.implode()))
        sample_ids = group.get_column("sample_id").unique().sort()

        keep_ids.extend(sample_ids)
    keep_ids += (
        samples.filter(label=label_mapping["decreases"])
        .get_column("sample_id")
        .unique()
        .to_list()
    )
    samples = samples.filter(col("sample_id").is_in(keep_ids))
    return samples


def make_sample_set_balanced(
    samples: pl.DataFrame,
    random_seed: int = 42,
) -> pl.DataFrame:
    """
    Make a sample set balanced by downsampling the majority class in the dataset to the
    size of the minority class to prevent the model from inclining towards the majority
    class. Functions ensures random sample selection.
    """
    sample_ids = (
        samples.group_by("sample_id").agg(pl.all().first()).select("sample_id", "label")
    )
    sample_ids_count = sample_ids.get_column("label").value_counts()

    majority_class_label = (
        sample_ids_count.filter(col("count") == col("count").max())
        .get_column("label")
        .item()
    )
    minority_class_label = (
        sample_ids_count.filter(col("count") == col("count").min())
        .get_column("label")
        .item()
    )
    minority_class_sample_n = (
        sample_ids_count.filter(col("count") == col("count").min())
        .get_column("count")
        .item()
    )

    keep_ids = (
        # subsample from majority class
        sample_ids.filter(col("label") == majority_class_label)
        .get_column("sample_id")
        .sample(minority_class_sample_n, seed=random_seed)
        # keep all samples from minority class
        .append(
            sample_ids.filter(col("label") == minority_class_label).get_column(
                "sample_id"
            )
        )
        .unique()
        .sort()
    )

    # Sanity check
    assert len(keep_ids) == minority_class_sample_n * 2, (
        "Sample set is not balanced. "
        f"Expected {minority_class_sample_n * 2} samples, got {len(keep_ids)}"
    )

    return samples.filter(col("sample_id").is_in(keep_ids))
