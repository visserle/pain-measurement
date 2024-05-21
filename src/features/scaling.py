import polars as pl

from src.features.transformations import map_trials

EXCLUDE_COLUMNS = [
    "Timestamp",
    "SampleNumber",
    "Trial",
    "Stimulus_Seed",
    "Participant",
]


def _scale_min_max_col(col: pl.Expr) -> pl.Expr:
    return (col - col.min()) / (col.max() - col.min())


def _scale_standard_col(col: pl.Expr) -> pl.Expr:
    return (col - col.mean()) / col.std()


def _scale_percent_to_decimal_col(col: pl.Expr) -> pl.Expr:
    return (col / 100).round(5)


@map_trials
def scale_min_max(
    df: pl.DataFrame,
    exclude_columns: list[str] = EXCLUDE_COLUMNS,
) -> pl.DataFrame:
    """NOTE: Not for usage in ML pipeline (data leakage)."""
    # TODO: trial shouldn't even be float64
    return df.with_columns(
        _scale_min_max_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


@map_trials
def scale_standard(
    df: pl.DataFrame,
    exclude_columns: list[str] = EXCLUDE_COLUMNS,
) -> pl.DataFrame:
    """NOTE: Not for usage in ML pipeline (data leakage)."""
    # TODO: trial shouldn't even be float64
    return df.with_columns(
        _scale_standard_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


# does not need to be mapped by trial since it's a simple operation
def scale_percent_to_decimal(
    df: pl.DataFrame,
    exclude_columns: list[str] = EXCLUDE_COLUMNS,
) -> pl.DataFrame:
    """Scale percentage columns to decimal."""
    # TODO: trial shouldn't even be float64
    return df.with_columns(
        _scale_percent_to_decimal_col(pl.col(pl.Float64).exclude(exclude_columns))
    )
