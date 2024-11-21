"""
This module contains functions to calculate and visualize correlations between
two columns at trial and participant level.

Example usage:
```python
col1, col2 = "pupil", "temperature"

corr_by_trial = calculate_correlations_by_trial(df, col1, col2)
corr_by_participant = aggregate_correlations_fisher_z(
    corr_by_trial, f"{col1}_{col2}_corr", "participant_id", include_ci=True
)
plot_correlations_by_trial(corr_by_trial, f"{col1}_{col2}_corr")
# or
plot_correlations_by_participant(corr_by_participant, f"{col1}_{col2}_corr")
```

Note that the correlation of time series violates the assumption of independence
and can lead to spurious results. Use with caution.
"""

import logging

import altair as alt
import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


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
    # Remove nan correlations (can happen if one variable is constant)
    # This way we don't lose a whole group if one correlation is nan
    if df.filter(col(correlation_column) == float("nan")).height > 0:
        logger.debug("Removing NaN correlations")

    df = df.filter(col(correlation_column) != float("nan"))

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
    y_domain: tuple = (-1, 1),
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
    y_domain : tuple
        (min, max) values for y-axis domain

    Returns:
    --------
    altair.Chart
        Scatter plot with connected lines showing trial correlations by participant
    """

    if title is None:
        title = (
            f"{correlation_column.replace('_', ' ').title()} by Trial and Participant"
        )

    base = alt.Chart(df).encode(
        x=alt.X(f"{trial_column}:Q", axis=alt.Axis(title="Trial ID")),
        y=alt.Y(
            f"{correlation_column}:Q",
            axis=alt.Axis(title="Correlation"),
            scale=alt.Scale(domain=y_domain),
        ),
        color=alt.Color(
            f"{participant_column}:N", legend=alt.Legend(title="Participant ID")
        ),
    )

    lines = base.mark_line(opacity=0.5)

    points = base.mark_circle(size=point_size).encode(
        tooltip=[
            f"{participant_column}:N",
            f"{trial_column}:Q",
            alt.Tooltip(f"{correlation_column}:Q", format=".3f"),
        ]
    )

    chart_config = _get_base_chart_config(width, height, title)

    return (lines + points).properties(**chart_config).interactive()


def plot_correlations_by_participant(
    df: pl.DataFrame,
    correlation_column: str,
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 800,
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

    if title is None:
        title = f"Mean {correlation_column.replace('_', ' ').title()} by Participant with 95% CI"

    # Create column names
    mean_col = f"{participant_column}_{correlation_column}_mean"
    ci_lower = f"{participant_column}_{correlation_column}_ci_lower"
    ci_upper = f"{participant_column}_{correlation_column}_ci_upper"

    base = alt.Chart(df).encode(
        x=alt.X(f"{participant_column}:O", axis=alt.Axis(title="Participant ID")),
        y=alt.Y(
            f"{ci_lower}:Q",
            scale=alt.Scale(domain=y_domain),
            axis=alt.Axis(title="Correlation"),
        ),
    )

    error_bars = base.mark_rule().encode(y2=f"{ci_upper}:Q")
    points = base.mark_circle(size=100, color="#1f77b4").encode(
        y=f"{mean_col}:Q",
        tooltip=[
            alt.Tooltip(f"{participant_column}:N", title="Participant"),
            alt.Tooltip(f"{mean_col}:Q", title="Mean Correlation", format=".3f"),
            alt.Tooltip(f"{ci_lower}:Q", title="CI Lower", format=".3f"),
            alt.Tooltip(f"{ci_upper}:Q", title="CI Upper", format=".3f"),
        ],
    )
    chart_config = _get_base_chart_config(width, height, title)

    return (error_bars + points).properties(**chart_config)


def _get_base_chart_config(
    width: int,
    height: int,
    title: str,
):
    """Helper function to provide consistent chart configuration"""
    return {
        "width": width,
        "height": height,
        "title": title,
        "config": {
            "axis": {"grid": True, "gridColor": "#ededed"},
            "view": {"strokeWidth": 0},
            "title": {"fontSize": 16, "anchor": "middle"},
        },
    }
