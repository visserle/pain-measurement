import polars as pl
from polars import col


def add_normalized_timestamp(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    trial_column: str = "trial_id",
):
    return df.with_columns(
        [
            (col(time_column) - col(time_column).min().over(trial_column)).alias(
                "normalized_timestamp"
            )
        ]
    )


def prepare_multiline_hvplot(
    df: pl.DataFrame,
    time_column: str = "normalized_timestamp",
    trial_column: str = "trial_id",
):
    """
    Add NaN separators at the end of each trial to prepare plotting of multiple lines
    per category in hvplot using normalized timestamps (which start at 0 for each
    trial).

    The time column should be normalized to start at 0 for each trial as this is the
    column you also want the multiple lines to be plotted against.

    https://hvplot.holoviz.org/user_guide/Large_Timeseries.html#multiple-lines-per-category-example

    Note that this workaround prevents the use of time data types in the DataFrame as
    NaN values are not supported for time data types in Polars.
    """
    # Sanity check if time column is normalized if we are looking at intervals
    if (
        df.group_by(trial_column)
        .agg(pl.col(time_column).min().alias("min_time"))
        .select(pl.sum("min_time"))
        .item()
        != 0
    ):
        raise ValueError(
            "Time column is not normalized to start at 0 for each trial. "
            "Please normalize the time column before calling this function "
            "using add_normalized_timestamp."
        )

    # Define columns where we don't want to add NaN separators
    info_columns = [
        "trial_id",
        "trial_number",
        "participant_id",
        "stimulus_seed",
        "trial_specific_interval_id",  # TODO: remove
        "continuous_interval_id",
        "decreasing_intervals",
        "major_decreasing_intervals",
        "increasing_intervals",
        "strictly_increasing",
        "plateau_intervals",
        "prolonged_minima_intervals",
    ]
    # Create a new DataFrame with NaN rows
    group_by_columns = [x for x in df.columns if x in info_columns]

    nan_df = df.group_by(group_by_columns).agg(
        [
            pl.lit(float("nan")).alias(col)  # following hvplot docs, not using pl.Null
            for col in df.columns
            if col not in info_columns
        ]
    )

    # Add a temporary column for sorting to add NaN separators at the end of each trial
    df = df.with_columns(pl.lit(0).alias("temp_sort"))
    nan_df = nan_df.with_columns(pl.lit(1).alias("temp_sort"))

    # Concatenate the original DataFrame with the NaN DataFrame and sort
    result = (
        pl.concat([df, nan_df], how="diagonal_relaxed", rechunk=True)
        .sort([trial_column, "temp_sort", time_column])
        .drop("temp_sort")
    )

    return result


def aggregate_correlations_fisher_z(
    df, correlation_column, group_by_column, include_ci=False
):
    """
    Perform Fisher z-transformation on correlations and return mean correlation and
    confidence intervals

    Parameters:
    -----------
    df : polars.DataFrame
        Input dataframe containing correlations
    correlation_column : str
        Name of column containing correlation values
    group_by_column : str
        Name of column to group by
    include_ci : bool, default=True
        Whether to calculate confidence intervals

    Returns:
    --------
    polars.DataFrame
        DataFrame with mean correlations and optionally confidence intervals
    """
    result = (
        df.with_columns(
            [
                pl.col(correlation_column)
                .clip(-0.9999, 0.9999)  # Clip values to avoid arctanh infinity
                .arctanh()
                .alias("z_transform")
            ]
        )
        .group_by(group_by_column, maintain_order=True)
        .agg(
            [
                pl.col("z_transform").mean().alias("mean_z"),
                (
                    pl.col("z_transform").std() / pl.col("z_transform").count().sqrt()
                ).alias("se_z"),
            ]
        )
        .with_columns(
            [
                pl.col("mean_z")
                .tanh()
                .alias(f"{group_by_column}_{correlation_column}_mean")
            ]
        )
    )

    if include_ci:
        result = (
            result.with_columns(
                [
                    (pl.col("mean_z") - 1.96 * pl.col("se_z")).alias("temp_lower_z"),
                    (pl.col("mean_z") + 1.96 * pl.col("se_z")).alias("temp_upper_z"),
                ]
            )
            .with_columns(
                [
                    pl.col("temp_lower_z")
                    .tanh()
                    .alias(f"{group_by_column}_{correlation_column}_ci_lower"),
                    pl.col("temp_upper_z")
                    .tanh()
                    .alias(f"{group_by_column}_{correlation_column}_ci_upper"),
                ]
            )
            .drop("temp_lower_z", "temp_upper_z")
        )

    return result.sort(group_by_column).drop("mean_z", "se_z")
