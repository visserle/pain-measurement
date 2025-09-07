import logging

import altair as alt
import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def _create_corr_column_name(col1: str, col2: str):
    if col1 is None or col2 is None:
        raise ValueError("Please provide column names to correlate")
    return f"{col1}_{col2}_corr"


def calculate_correlations_by_trial(
    df: pl.DataFrame,
    reference: str,
    targets: str | list[str],
    trial_column: str = "trial_id",
):
    """
    Calculate correlations between a reference column and one or more target columns at trial level

    Args:
        df: Polars DataFrame
        reference: Reference column name to correlate against
        targets: Target column name(s) to correlate - can be a string or list of strings
        trial_column: Column name for trial identifier (default: "trial_id")

    Returns:
        DataFrame with trial-level correlations

    Note:
        To aggregate the correlations to participant level, use the
        `aggregate_correlations_fisher_z` function.
    """
    # Convert single string to list for uniform processing
    if isinstance(targets, str):
        targets = [targets]

    # Build aggregation expressions
    agg_exprs = [
        pl.first(["trial_number", "participant_id", "stimulus_seed"]),
        # Single n_obs column for the reference (assuming all targets have same non-null pattern)
        pl.col(reference).is_not_null().sum().alias("n_obs"),
    ]

    # Add correlation for each target column
    for target_col in targets:
        # Create correlation column name
        corr_column = _create_corr_column_name(reference, target_col)
        agg_exprs.append(pl.corr(reference, target_col).alias(corr_column))

    # Calculate correlations for each trial_id
    corr_by_trial = (
        df.group_by(trial_column, maintain_order=True)
        .agg(agg_exprs)
        .sort([trial_column])
    )

    return corr_by_trial
