import logging
from functools import reduce, wraps

import polars as pl
from polars.exceptions import ColumnNotFoundError

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
        try:
            df = df.group_by("trial_id", maintain_order=True).map_groups(
                lambda group: func(group, *args, **kwargs)
            )
        except ColumnNotFoundError:
            logger.warning(
                f"No 'trial_id' column found. Applying function {func.__name__} to the "
                "entire DataFrame."
            )
            df = func(df, *args, **kwargs)
        return df

    return wrapper


def map_participants(func: callable):
    """
    Decorator to apply a function to each participant in a pl.DataFrame.
    """

    @wraps(func)
    def wrapper(
        df: pl.DataFrame,
        *args,
        **kwargs,
    ) -> pl.DataFrame:
        try:
            df = df.group_by("participant_id", maintain_order=True).map_groups(
                lambda group: func(group, *args, **kwargs)
            )
        except ColumnNotFoundError:
            logger.warning(
                f"No 'participant_id' column found. Applying function {func.__name__} "
                "to the entire DataFrame."
            )
            df = func(df, *args, **kwargs)
        return df

    return wrapper


def merge_data_dfs(
    dfs: list[pl.DataFrame],
    merge_on: list[str] = ["participant_id", "trial_id", "trial_number", "timestamp"],
    sort_by: list[str] = ["trial_id", "timestamp"],
) -> pl.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame.

    Default merge_on and sort_by columns are set for data DataFrames (raw, preprocess,
    feature).
    """
    if len(dfs) < 2:
        return dfs[0]

    df = reduce(
        lambda left, right: left.join(
            right,
            on=merge_on,
            how="full",
            coalesce=True,
        )
        .sort(sort_by)
        .drop(["rownumber_right", "samplenumber_right"], strict=False),
        dfs,
    )
    return df


@map_trials
def interpolate_and_fill_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Interpolate and fill null values in a DataFrame.
    """
    # interpolate() casts every column to Float64 so we use pl.selectors for int columns
    return (
        df.with_columns(df.select(pl.selectors.by_dtype(pl.Float64)).interpolate())
        .fill_null(strategy="forward")
        .fill_null(strategy="backward")
    )


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
