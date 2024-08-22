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
        return df.group_by("participant_id", maintain_order=True).map_groups(
            lambda group: func(group, *args, **kwargs)
        )

    return wrapper


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
