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
