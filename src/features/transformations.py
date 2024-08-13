import logging
from functools import wraps

import polars as pl

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

info_columns = [
    "trial_id",
    "trial_number",
    "participant_id",
    "stimulus_seed",
    "trial_specific_interval_id",
    "continuous_interval_id",
]


def map_trials(func: callable):
    """
    Decorator to apply a function to each trial in a pl.DataFrame.
    """

    @wraps(func)
    def wrapper(
        df: pl.DataFrame,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        return df.group_by("trial_id", maintain_order=True).map_groups(
            lambda group: func(group, *args, **kwargs)
        )

    return wrapper


def remove_dulpicate_timestamps(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Remove duplicate timestamps from the DataFrame.

    For instance, the Shimmer3 GSR+ unit collects ca. 128 samples per second but with
    only 100 unique timestamps. This function removes the duplicates.
    """
    return df.unique("timestamp")


def add_time_column(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Create a new column that contains the time from Timestamp in ms.

    Note: This datatype is not fully implemented in Polars and DuckDB yet and is not
    recommended for saving to a database.
    """
    df = df.with_columns(
        pl.col("timestamp").cast(pl.Duration(time_unit="ms")).alias("time")
    )
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
