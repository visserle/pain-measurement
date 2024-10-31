"""
This module contains functions to calculate and visualize correlations between
two columns at trial and participant level.

Example usage:
```python
# Columns to be correlated
col1 = "pupil"
col2 = "rating"

corr_by_trial = calculate_correlations_by_trial(df, col1, col2)
corr_by_participant = aggregate_correlations_fisher_z(
    corr_by_trial, "pupil_rating_corr", "participant_id", include_ci=True
)

plot_correlations_by_trial(corr_by_trial, "pupil_rating_corr")
# or
plot_correlations_by_participant(corr_by_participant, "pupil_rating_corr")
```

Note that the correlation of time series violates the assumption of independence
and can lead to spurious results. Use with caution.
"""

import altair as alt
import polars as pl
from polars import col


def calculate_correlations_by_trial(
    df: pl.DataFrame,
    col1: str,
    col2: str,
    trial_column: str = "trial_id",
):
    """
    Calculate correlations between two columns at trial and participant level

    Args:
        df: Polars DataFrame
        col1: First column name to correlate
        col2: Second column name to correlate

    Returns:
        DataFrame with participant-level correlations and confidence intervals

    Note:
        To aggregate the correlations to participant level, use the
        `aggregate_correlations_fisher_z` function.
    """
    # Create correlation column name
    corr_col = f"{col1}_{col2}_corr"

    # Calculate correlation between columns for each trial_id
    corr_by_trial = df.group_by("trial_id", maintain_order=True).agg(
        pl.corr(col1, col2).alias(corr_col),
        pl.first("participant_id"),
    )

    return corr_by_trial


def aggregate_correlations_fisher_z(
    df: pl.DataFrame,
    correlation_column: str,
    group_by_column: str,
    include_ci=False,
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


def plot_correlations_by_trial(
    df: pl.DataFrame,
    correlation_column: str,
    trial_column: str = "trial_id",
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 800,
    height: int = 400,
    point_size: int = 60,
):
    """
    Create an Altair chart showing correlations by trial, grouped by participant

    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing trial-level correlation data
    correlation_column : str
        Name of correlation column
    trial_column : str
        Name of trial ID column
    participant_column : str
        Name of participant ID column
    title : str, optional
        Chart title. If None, auto-generated from correlation column name
    width : int
        Chart width in pixels
    height : int
        Chart height in pixels
    point_size : int
        Size of scatter points

    Returns:
    --------
    altair.Chart
        Scatter plot with connected lines showing trial correlations by participant
    """

    # Auto-generate title if not provided
    if title is None:
        title = (
            f"{correlation_column.replace('_', ' ').title()} by Trial and Participant"
        )

    plot = (
        alt.Chart(df)
        .mark_line(opacity=0.3)
        .encode(
            x=alt.X(f"{trial_column}:Q", axis=alt.Axis(title="Trial Number")),
            y=alt.Y(
                f"{correlation_column}:Q",
                axis=alt.Axis(title="Correlation"),
            ),
            color=alt.Color(
                f"{participant_column}:N", legend=alt.Legend(title="Participant ID")
            ),
        )
    )

    # Add points on top of lines
    points = (
        alt.Chart(df)
        .mark_circle(size=point_size)
        .encode(
            x=f"{trial_column}:Q",
            y=f"{correlation_column}:Q",
            color=f"{participant_column}:N",
            tooltip=[
                f"{participant_column}:N",
                f"{trial_column}:Q",
                alt.Tooltip(f"{correlation_column}:Q", format=".3f"),
            ],
        )
    )

    # Combine layers
    combined_plot = (
        (plot + points)
        .properties(width=width, height=height, title=title)
        .configure_axis(grid=True, gridColor="#ededed")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=16, anchor="middle")
        .interactive()  # Makes the plot interactive for zooming/panning
    )

    return combined_plot


def plot_correlations_by_participant(
    df: pl.DataFrame,
    correlation_column: str,
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 600,
    height: int = 400,
    y_domain: tuple = (-1, 1),
):
    """
    Create an Altair chart showing correlations by participant with error bars

    Parameters:
    -----------
    df : polars.DataFrame
        DataFrame containing correlation data with mean and CI columns
    correlation_column : str
        Base name of correlation columns (without _mean/_ci suffixes)
    participant_column : str
        Name of participant ID column
    title : str, optional
        Chart title. If None, auto-generated from correlation column name
    width : int
        Chart width in pixels
    height : int
        Chart height in pixels
    y_domain : tuple
        (min, max) values for y-axis domain

    Returns:
    --------
    altair.Chart
        Combined error bar and point chart
    """

    # Auto-generate title if not provided
    if title is None:
        title = f"Mean {correlation_column} Correlations by Participant with 95% CI"

    # Create column names
    mean_col = f"{participant_column}_{correlation_column}_mean"
    ci_lower = f"{participant_column}_{correlation_column}_ci_lower"
    ci_upper = f"{participant_column}_{correlation_column}_ci_upper"

    # Create error bars
    error_bars = (
        alt.Chart(df)
        .mark_rule()
        .encode(
            x=alt.X(f"{participant_column}:O", axis=alt.Axis(title="Participant ID")),
            y=alt.Y(
                f"{ci_lower}:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(
                    title=correlation_column.replace("_", " ").title() + " Correlation"
                ),
            ),
            y2=f"{ci_upper}:Q",
        )
    )

    # Create scatter points
    points = (
        alt.Chart(df)
        .mark_circle(size=100, color="#1f77b4")
        .encode(x=f"{participant_column}:O", y=f"{mean_col}:Q")
    )

    # Combine layers
    plot = (
        (error_bars + points)
        .properties(
            width=width,
            height=height,
            title=title,
        )
        .configure_axis(grid=True, gridColor="#ededed")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=16, anchor="middle")
    )

    return plot
