"""
This module contains functions to calculate and visualize correlations between
two columns at trial and participant level.

Example usage:
```python
col1, col2 = "pupil", "temperature"

corr_by_trial = calculate_correlations_by_trial(df, col1, col2)
corr_by_participant = aggregate_correlations_fisher_z(
    corr_by_trial, col1, col2, "participant_id", include_ci=True
)
plot_correlations_by_trial(corr_by_trial, col1, col2)
# or
plot_correlations_by_participant(corr_by_participant, col1, col2)
```

Note that the correlation of time series violates the assumption of independence
and can lead to spurious results. Use with caution.
"""

import logging

import altair as alt
import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


COLORS = {
    "temperature_rating_corr": "red",
    "temperature_pupil_corr": "#ff7f0e",
    "temperature_eda_tonic_corr": "#2ca02c",
    "temperature_eda_phasic_corr": "#d62728",
    "temperature_heartrate_corr": "#9467bd",
    "temperature_brow_furrow_corr": "red",
    "temperature_cheek_raise_corr": "#2ca02c",
    "temperature_mouth_open_corr": "#d62728",
    "temperature_upper_lip_raise_corr": "#9467bd",
    "temperature_nose_wrinkle_corr": "#ff7f0e",
}


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
    corr_column = _create_corr_column_name(col1, col2)

    # Calculate correlation between columns for each trial_id
    corr_by_trial = (
        df.group_by("trial_id", maintain_order=True)
        .agg(
            pl.first(["trial_number", "participant_id", "stimulus_seed"]),
            pl.corr(col1, col2).alias(corr_column),
        )
        .sort(["trial_id"])
    )

    return corr_by_trial


def _create_corr_column_name(col1: str, col2: str):
    if col1 is None or col2 is None:
        raise ValueError("Please provide column names to correlate")
    return col1, col2


def plot_correlations_by_trial(
    corr_by_trial: pl.DataFrame,
    col1: str = None,
    col2: str = None,
    trial_column: str = "trial_id",
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 800,
    height: int = 400,
    y_domain: tuple = (-1, 1),
):
    """
    Create an Altair chart showing correlations by trial, grouped by participant
    """
    # Create correlation column name
    corr_column = _create_corr_column_name(col1, col2)

    # Set default title if none provided
    if title is None:
        title = f"{corr_column.replace('_', ' ').title()} by Trial and Participant"

    # Create base chart with shared encoding
    base = (
        alt.Chart(corr_by_trial)
        .encode(
            x=alt.X(f"{trial_column}:Q", axis=alt.Axis(title="Trial ID")),
            y=alt.Y(
                f"{corr_column}:Q",
                axis=alt.Axis(title="Correlation"),
                scale=alt.Scale(domain=y_domain),
            ),
            color=alt.Color(
                f"{participant_column}:N", legend=alt.Legend(title="Participant ID")
            ),
        )
        .properties(
            width=width,
            height=height,
            title=title,
        )
    )

    lines = base.mark_line(opacity=0.5)
    points = base.mark_circle(size=60).encode(
        tooltip=[
            f"{participant_column}:N",
            f"{trial_column}:Q",
            alt.Tooltip(f"{corr_column}:Q", format=".3f"),
        ]
    )

    return (
        alt.layer(lines, points)
        .configure_axis(grid=True, gridColor="#ededed")
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=16, anchor="middle")
        .interactive()
    )


def aggregate_correlations_fisher_z(
    df: pl.DataFrame,
    col1: str,
    col2: str,
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
    corr_column : str
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
    # Create correlation column name
    corr_column = _create_corr_column_name(col1, col2)

    # Remove nan correlations (can happen if e.g. one variable is constant)
    # This way we don't lose a whole group if one correlation is nan
    if df.filter(col(corr_column) == float("nan")).height > 0:
        logger.debug("Removing NaN correlations")

    df = df.filter(col(corr_column) != float("nan"))

    result = (
        df.with_columns(
            [
                pl.col(corr_column)
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
            [pl.col("mean_z").tanh().alias(f"{group_by_column}_{corr_column}_mean")]
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
                    .tanh()  # transform back to correlation space
                    .alias(f"{group_by_column}_{corr_column}_ci_lower"),
                    pl.col("temp_upper_z")
                    .tanh()
                    .alias(f"{group_by_column}_{corr_column}_ci_upper"),
                ]
            )
            .drop("temp_lower_z", "temp_upper_z")
        )

    return result.sort(group_by_column).drop("mean_z", "se_z")


def plot_mean_correlations_by_participant(
    corr_by_participant: pl.DataFrame,
    col1: str,
    col2: str,
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 800,
    height: int = 400,
    y_domain: tuple = (-1, 1),
    with_config: bool = True,  # for layered charts there must be no config
    color_map: dict = COLORS,
):
    """
    Create an Altair chart showing correlations by participant, grouped by trial

    For layered charts, you have to:
    1. Set with_config=False (configure the chart outside of this function)
    2. Specify the color_map for the different correlation types.
    3. Add plots using the + operator.

    (Another more elegant layering approach using color="independent" results in
    multiple subheadings in the legend, which is not desired.)
    """
    # Create correlation column name
    corr_column = _create_corr_column_name(col1, col2)

    # Add correlation type to column name for legend
    corr_by_participant = corr_by_participant.with_columns(
        pl.lit(corr_column).alias("correlation_type")
    )

    # Set default title if none provided
    if title is None:
        title = (
            f"Mean {corr_column.replace('_', ' ').title()} by Participant with 95% CI"
        )

    # Get column names
    mean_col = f"{participant_column}_{corr_column}_mean"
    ci_lower = f"{participant_column}_{corr_column}_ci_lower"
    ci_upper = f"{participant_column}_{corr_column}_ci_upper"

    # Create base chart
    base = (
        alt.Chart(
            corr_by_participant,
            width=width,
            height=height,
        )
        .encode(
            x=alt.X(f"{participant_column}:O", axis=alt.Axis(title="Participant ID")),
            y=alt.Y(
                f"{ci_lower}:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(title="Correlation"),
            ),
            color=alt.Color(
                "correlation_type:N",
                scale=alt.Scale(
                    domain=list(COLORS.keys()), range=list(COLORS.values())
                ),
                legend=alt.Legend(
                    title="Correlation Type",
                    labelExpr="replace(replace(datum.label, '_corr', ''), '_', ' ')",
                ),
            ),
        )
        .properties(title=title, width=width, height=height)
    )

    error_bars = base.mark_rule().encode(y2=f"{ci_upper}:Q")
    points = base.mark_circle(
        size=100,
    ).encode(
        y=f"{mean_col}:Q",
        tooltip=[
            alt.Tooltip(f"{participant_column}:N", title="Participant"),
            alt.Tooltip(f"{mean_col}:Q", title="Mean Correlation", format=".3f"),
            alt.Tooltip(f"{ci_lower}:Q", title="CI Lower", format=".3f"),
            alt.Tooltip(f"{ci_upper}:Q", title="CI Upper", format=".3f"),
        ],
    )

    if with_config:
        return (
            alt.layer(error_bars, points)
            .configure_axis(grid=True, gridColor="#ededed")
            .configure_view(strokeWidth=0)
            .configure_title(fontSize=16, anchor="middle")
        )

    return alt.layer(error_bars, points)


def plot_max_correlations_by_participant(
    corr_by_trial: pl.DataFrame,
    col1: str,
    col2: str,
    participant_column: str = "participant_id",
    title: str = None,
    width: int = 800,
    height: int = 400,
    y_domain: tuple = (-0.5, 1),
    with_config: bool = True,
    color_map: dict = COLORS,
):
    # Create correlation column name
    corr_column = _create_corr_column_name(col1, col2)

    # Set default title if none provided
    if title is None:
        title = f"Maximum {corr_column.replace('_', ' ').title()} by Participant"

    # Calculate the maximum correlation for each participant
    max_corr_df = corr_by_trial.group_by(participant_column, maintain_order=True).agg(
        pl.col(corr_column).max().alias(f"{corr_column}_max")
    )

    # Create a copy of the dataframe with a new column for the correlation type
    max_corr_df = max_corr_df.with_columns(
        pl.lit(corr_column).alias("correlation_type")
    )

    # The maximum correlation column name
    max_corr_col = f"{corr_column}_max"

    # Create base chart
    base = (
        alt.Chart(
            max_corr_df,
            width=width,
            height=height,
        )
        .encode(
            x=alt.X(f"{participant_column}:O", axis=alt.Axis(title="Participant ID")),
            y=alt.Y(
                f"{max_corr_col}:Q",
                scale=alt.Scale(domain=y_domain),
                axis=alt.Axis(title="Maximum Correlation"),
            ),
            # Use the correlation_type for color encoding to create the legend
            color=alt.Color(
                "correlation_type:N",
                scale=alt.Scale(
                    domain=list(color_map.keys()), range=list(color_map.values())
                ),
                legend=alt.Legend(
                    title="Correlation Type",
                    labelExpr="replace(replace(datum.label, '_corr', ''), '_', ' ')",
                ),
            ),
        )
        .properties(title=title, width=width, height=height)
    )

    circles = base.mark_circle(size=100).encode(
        tooltip=[
            alt.Tooltip(f"{participant_column}:N", title="Participant"),
            alt.Tooltip(f"{max_corr_col}:Q", title="Max Correlation", format=".3f"),
        ],
    )

    if with_config:
        return (
            circles.configure_axis(grid=True, gridColor="#ededed")
            .configure_view(strokeWidth=0)
            .configure_title(fontSize=16, anchor="middle")
        )

    return circles.properties(title=title)
