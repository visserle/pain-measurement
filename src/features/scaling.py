# NOTE: explicit is better than implicit -> the scaling functions could potentially
# improved by using column names directly instead of the exclude additional columns
# keyword argument


import polars as pl

from src.features.transforming import map_trials

EXCLUDE_COLUMNS = [
    "trial_id",
    "trial_number",
    "participant_id",
    "stimulus_seed",
    "skin_area",
    "timestamp",
    "normalized_timestamp",
    "samplenumber",
]


@map_trials
def scale_min_max(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scales Float64 columns to the range [0, 1] for each trial.

    Note: Should be used with caution in ML pipelines to avoid **data leakage**.
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_min_max_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


@map_trials
def scale_standard(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scale Float64 columns to have mean 0 and standard deviation 1 for each trial.

    Note: Should be used with caution in ML pipelines to avoid **data leakage**.
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_standard_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


@map_trials
def scale_robust_standard(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scale Float64 columns to have median 0 and median absolute deviation 1 for each trial.

    Note: Should be used with caution in ML pipelines to avoid **data leakage**.
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_robust_standard_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


# does not need to be mapped since it's a simple operation
def scale_percent_to_decimal(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scales Float64 columns that are in percentage format to decimal format.

    In addition to the default columns to exclude, you can pass a list of additional
    columns to exclude from scaling.
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_percent_to_decimal_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


def _scale_min_max_col(column: pl.Expr) -> pl.Expr:
    return (column - column.min()) / (column.max() - column.min())


def _scale_standard_col(column: pl.Expr) -> pl.Expr:
    return (column - column.mean()) / column.std()


def _scale_robust_standard_col(column: pl.Expr) -> pl.Expr:
    return (column - column.median()) / (column.quantile(0.75) - column.quantile(0.25))


def _scale_percent_to_decimal_col(column: pl.Expr) -> pl.Expr:
    return (column / 100).round(5)  # round to avoid floating point weirdness
