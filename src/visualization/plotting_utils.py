import polars as pl


def prepare_multiline_hvplot(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    trial_column: str = "trial_id",
):
    """
    Add NaN separators at the end of each trial to prepare plotting of multiple lines
    per category in hvplot.

    https://hvplot.holoviz.org/user_guide/Large_Timeseries.html#multiple-lines-per-category-example
    """

    # Define columns where we don't want to add NaN separators
    info_columns = [
        "trial_id",
        "trial_number",
        "participant_id",
        "stimulus_seed",
        "trial_specific_interval_id",
        "continuous_interval_id",
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

    # Concatenate the original DataFrame with the NaN DataFrame
    result = pl.concat(
        [df, nan_df],
        how="diagonal_relaxed",  # FIXME: maybe we can do better than just ignoring the schema
        rechunk=True,
    ).sort([trial_column, "temp_sort", time_column])

    # Remove the temporary sorting column and return the result
    return result.drop("temp_sort")
