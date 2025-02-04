import polars as pl
from polars import col


def to_describe(
    column: str,
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
    prefix = prefix or f"{column}_"
    return [
        col(column).count().alias(f"{prefix}count"),
        col(column).is_null().sum().alias(f"{prefix}null_count"),
        col(column).mean().alias(f"{prefix}mean"),
        col(column).std().alias(f"{prefix}std"),
        col(column).min().alias(f"{prefix}min"),
        col(column).quantile(0.25).alias(f"{prefix}25%"),
        col(column).quantile(0.5).alias(f"{prefix}50%"),
        col(column).quantile(0.75).alias(f"{prefix}75%"),
        col(column).max().alias(f"{prefix}max"),
    ]
