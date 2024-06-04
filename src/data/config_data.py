from dataclasses import dataclass, field

import polars as pl

# NOTE:
# Some out-of-the-box alternatives to this simple approach:
# Snakemake: https://snakemake.github.io/
# Kohesio: https://engineering.nike.com/koheesio/latest/tutorials/onboarding.html


class Transformation:
    """Use a function as a object with kwargs in the transformation pipeline."""

    def __init__(self, func: callable, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.func(df, **self.kwargs)

    def __repr__(self):
        return (
            f"{self.func.__name__}"
            + f"({', '.join(f'{k}={v}' for k, v in self.kwargs.items())})"
        )


@dataclass(kw_only=True)
class DataConfigBase:
    """
    Base class for data configuration.
    """

    transformations: list[Transformation] = field(default_factory=list)
