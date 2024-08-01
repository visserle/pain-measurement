import logging
from dataclasses import dataclass, field
from functools import reduce, wraps

import numpy as np
import polars as pl

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


# NOTE: square brackets can be used to access columns in polars but do not allow lazy
# evaluation -> use select, take, etc for performance
# (for lazy evaluation, one should also use pipe, etc.)


def map_trials(func: callable) -> callable:
    """Decorator to apply a function to each trial in a pl.DataFrame."""

    @wraps(func)
    def wrapper(
        df: pl.DataFrame,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        # Apply the function to each trial if "Trial" exists
        if "Trial" in df.columns:
            if len(df["Trial"].unique()) == 1:
                logger.debug(
                    "Only one trial found, applying function to the whole DataFrame."
                )
            result = df.group_by("Trial", maintain_order=True).map_groups(
                lambda group: func(group, *args, **kwargs)
            )
        # Else apply the function to the whole DataFrame
        else:
            logger.warning(
                f"No 'Trial' column found, applying function {func.__name__} "
                "to the whole DataFrame instead."
            )
            logger.info(
                f"Use {func.__name__}.__wrapped__() to access the function without the "
                "map_trials decorator."
            )
            result = func(df, *args, **kwargs)
        return result

    return wrapper


def add_timedelta_column(
    df: pl.DataFrame,
    timestamp_column="Timestamp",
    timedelta_column="Time",
    time_unit="ms",
) -> pl.DataFrame:
    """Create a new column that contains the time from Timestamp in ms."""
    # NOTE: saving timedelta to csv runs into problems, maybe we can do without it for now / just use it for debugging
    df = df.with_columns(
        pl.col(timestamp_column)
        .cast(pl.Duration(time_unit=time_unit))
        .alias(timedelta_column)
    )
    return df


# we do not need to map over participants here, because start and end points are defined
# by the marker and there is no risk of changing values that should not be changed
@map_trials
def interpolate(df: pl.DataFrame) -> pl.DataFrame:
    """Linearly interpolates the whole DataFrame"""
    return df.interpolate()


def resample(df: pl.DataFrame, ms: int) -> pl.DataFrame:  # FIXME
    if "Trial" in df.index.names:
        df = df.groupby("Trial").resample(f"{ms}ms", level="Time").mean()
    else:
        df = df.resample(f"{ms}ms", level="Time").mean()
    return df


def resample_to_500hz(df):  # FIXME
    if "Time" not in df.index.names:
        raise ValueError("Index must contain 'Time' for resampling.")
    if "Trial" in df.index.names:
        df = df.groupby("Trial").resample("2ms", level="Time").mean()
    else:
        df = df.resample("2ms", level="Time").mean()
    return df


"""
NOTE: NEW: scikit-learn’s transformers now support polars output with the set_output API.

import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pl.DataFrame(
    {"height": [120, 140, 150, 110, 100], "pet": ["dog", "cat", "dog", "cat", "cat"]}
)
preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), ["height"]),
        ("categorical", OneHotEncoder(sparse_output=False), ["pet"]),
    ],
    verbose_feature_names_out=False,
)
preprocessor.set_output(transform="polars")

df_out = preprocessor.fit_transform(df)
df_out
"""
