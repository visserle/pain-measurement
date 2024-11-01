import polars as pl
from polars import col


def to_describe(
    col: str,
    prefix: str = "",
) -> list[pl.Expr]:
    """
    Polars helper function for having the describe expression for use in groupby-agg.
    From https://github.com/pola-rs/polars/issues/8066#issuecomment-1794144838

    Example:
    ````python
    out = stimuli.group_by("seed", maintain_order=True).agg(
        *to_describe("y", prefix="temperature_"),
        *to_describe("time"),  # using list unpacking to pass multiple expressions
    )
    ````
    """
    prefix = prefix or f"{col}_"
    return [
        pl.col(col).count().alias(f"{prefix}count"),
        pl.col(col).is_null().sum().alias(f"{prefix}null_count"),
        pl.col(col).mean().alias(f"{prefix}mean"),
        pl.col(col).std().alias(f"{prefix}std"),
        pl.col(col).min().alias(f"{prefix}min"),
        pl.col(col).quantile(0.25).alias(f"{prefix}25%"),
        pl.col(col).quantile(0.5).alias(f"{prefix}50%"),
        pl.col(col).quantile(0.75).alias(f"{prefix}75%"),
        pl.col(col).max().alias(f"{prefix}max"),
    ]


def add_time_column(
    df: pl.DataFrame,
    time_column: str = "timestamp",
    time_unit: str = "ms",
    new_column_name: str = "time",
) -> pl.DataFrame:
    """
    Create a new column that contains the time from Timestamp in ms.

    Note: This datatype is not fully supported in Polars and DuckDB yet.
    Use with caution. https://github.com/pola-rs/polars/issues/13560
    """
    df = df.with_columns(
        col(time_column).cast(pl.Duration(time_unit=time_unit)).alias(new_column_name)
    )
    return df


def add_timestamp_µs_column(
    df: pl.DataFrame,
    time_column: str = "timestamp",
) -> pl.DataFrame:
    """
    Create a new column that contains the timestamp in microseconds (µs).

    Casts the datatype to Int64 which allow group_by_dynamic operations.
    """
    return df.with_columns(
        (col(time_column) * 1000).cast(pl.Int64).alias(time_column + "_µs")
    )
