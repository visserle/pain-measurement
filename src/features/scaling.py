import polars as pl

from src.features.transformations import map_participants, map_trials

# As long as we are not using a schema, we need to exclude these columns from scaling
EXCLUDE_COLUMNS = [
    "Timestamp",
    "SampleNumber",
    "Participant",
    "Trial",
    "Stimulus_Seed",
    "Skin_Area",
]


def _scale_min_max_col(col: pl.Expr) -> pl.Expr:
    return (col - col.min()) / (col.max() - col.min())


def _scale_standard_col(col: pl.Expr) -> pl.Expr:
    return (col - col.mean()) / col.std()


def _scale_percent_to_decimal_col(col: pl.Expr) -> pl.Expr:
    return (col / 100).round(5)  # round to avoid floating point weirdness


@map_participants
@map_trials
def scale_min_max(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scales Float64 columns to the range [0, 1].

    NOTE: For exploratory analysis only, not for usage in ML pipeline (data leakage).
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_min_max_col(pl.col(pl.Float64).exclude(exclude_columns))
    )


@map_participants
@map_trials
def scale_standard(
    df: pl.DataFrame,
    exclude_additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """
    Scale Float64 columns to have mean 0 and standard deviation 1.

    NOTE: For exploratory analysis only, not for usage in ML pipeline (data leakage).
    """
    exclude_columns = EXCLUDE_COLUMNS + (exclude_additional_columns or [])
    return df.with_columns(
        _scale_standard_col(pl.col(pl.Float64).exclude(exclude_columns))
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
