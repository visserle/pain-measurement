import polars as pl

from src.features.transformations import map_trials

# As long as data types are not set correctly, we need to exclude these columns
# from scaling to avoid NaNs.
# TODO FIXME: switch to parquet and set schema
EXCLUDE_COLUMNS = [
    "Timestamp",
    "SampleNumber",
    "Trial",
    "Stimulus_Seed",
    "Participant",
    "Skin_Area",
]


def _scale_min_max_col(col: pl.Expr) -> pl.Expr:
    return (col - col.min()) / (col.max() - col.min())


def _scale_standard_col(col: pl.Expr) -> pl.Expr:
    return (col - col.mean()) / col.std()


def _scale_percent_to_decimal_col(col: pl.Expr) -> pl.Expr:
    return (col / 100).round(5)  # round to avoid floating point weirdness


@map_trials
def scale_min_max(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scales Float64 columns to the range [0, 1].

    NOTE: Not for usage in ML pipeline (data leakage).
    """
    # TODO: trial shouldn't even be float64
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
    Scale Float64 columns to have mean 0 and standard deviation 1.

    NOTE: Not for usage in ML pipeline (data leakage).
    """
    # TODO: trial shouldn't even be float64
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_standard_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


# does not need to be mapped by trial since it's a simple operation
def scale_percent_to_decimal(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scales Float64 columns that are in percentage format to decimal format.

    In addition to the default columns to exclude, you can pass a list of additional
    columns to exclude from scaling.
    """
    # TODO: trial shouldn't even be float64
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_percent_to_decimal_col(pl.col(pl.Float64).exclude(exclude_columns))
    )
