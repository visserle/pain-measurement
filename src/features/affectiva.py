# Note: domain range is 0-1, scale accordingly
import polars as pl

from src.features.transformations import map_trials

EXCLUDE_COLUMNS = [
    "Timestamp",
    "SampleNumber",
    "Trial",
    "Stimulus_Seed",
    "Participant",
    "Trial",
]


@map_trials
def process_affectiva(
    df: pl.DataFrame,
    exclude_columns: list[str] = EXCLUDE_COLUMNS,
) -> pl.DataFrame:
    """Normalize each affectiva feature to the range [0, 1] by dividing by 100."""
    return df.with_columns(
        _scale_100_col(pl.col(pl.Float64).exclude(exclude_columns))
    )  # TODO: trial shouldn't even be float64


def _scale_100_col(col: pl.Expr) -> pl.Expr:
    return col / 100
