import logging
from functools import reduce, wraps
from typing import Iterable, List

import numpy as np
import polars as pl

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


# NOTE: square brackets can be used to access columns in polars but do not allow lazy evaluation -> use select, take, etc for performance
# although brackets can be faster for small dataframes
# therre are issues on github reporting eager is faster than lazy
# TODO: add **kwars to apply_func_participant and data config
# TODO: cast data types for more performance
# TODO: Round all timestamps to remove floating point weirdness
# 447,929.23030000000006, ...
# 448,929.2894, ...
# -> add as function to transformations of raw data, even imotions data?


def map_participant_datasets(func, participant):
    """Utility function for debugging, will be removed in the future, see process_data.py."""
    #TODO: use map instead, e.g.:
    # dict(zip(a, map(f, a.values())))
    # dict(map(lambda item: (item[0], f(item[1])), my_dictionary.items()
    for data in participant.datasets:
        participant.datasets[data].dataset = func(participant.datasets[data].dataset)
    return participant


def map_trials(func, trial_column='Trial'):
    """Decorator to apply a function to each trial in a DataFrame."""
    @wraps(func)
    def wrapper(df, *args, **kwargs):
        # Check if 'df' is a pl.DataFrame
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input must be a Polars DataFrame.")
        # Check if DataFrame has the specified trial column
        if trial_column in df.columns:
            # Warning if only one trial is found
            if len(df[trial_column].unique()) == 1:
                logger.warning("Only one trial found, applying function to the whole DataFrame.")
            # Apply the function to each trial
            result = df.group_by(trial_column, maintain_order=True).map_groups(lambda group: func(group, *args, **kwargs))
            logger.critical(f"Applying function {func.__name__} to each trial. We still have to find out if order is maintained.")
        else:
            # Apply the function to the whole DataFrame
            logger.warning(f"No '{trial_column}' column found, applying function {func.__name__} to the whole DataFrame instead.")
            logger.info(f"Use {func.__name__}.__wrapped__() to access the function without the map_trials decorator.")
            result = func(df, *args, **kwargs)
        return result
    return wrapper

def create_trials(
        df: pl.DataFrame,
        marker_column='Stimuli_Seed',
        trial_column='Trial') -> pl.DataFrame:
    """Create a trial column based on the marker column which originally saves the stimuli seed only once at the start and end of each trial."""
    # TODO: maybe we need to interpolate here for the nan at the start and end of each trial
    # TODO: Check if all trials are complete
    # Forward fill and backward fill columns
    ffill = df[marker_column].fill_null(strategy='forward') 
    bfill = df[marker_column].fill_null(strategy='backward')
    # Where forward fill and backward fill are equal, replace the NaNs in the original Stimuli_Seed
    # this is the same as np.where(ffill == bfill, ffill, df[marker_column])
    df = df.with_columns(
        pl.when(ffill == bfill)
        .then(ffill)
        .otherwise(df[marker_column])
        .alias(marker_column)
    )
    assert df['Timestamp'].is_sorted(descending=False)
    # Only keep rows where the Stimuli_Seed is not NaN
    df = df.filter(df[marker_column].is_not_null())
    # Create a new column that contains the trial number
    df = df.with_columns(
        pl.col(marker_column)
        .diff()                 # Calculate differences
        .fill_null(value=0)     # Replace initial null with 0 because the first trial is always 0
        .ne(0)                  # Check for non-zero differences
        .cum_sum()              # Cumulative sum of boolean values
        .cast(pl.UInt8)         # Cast to integer data type between 0 and 255
        .alias(trial_column)    # Rename the series
    )
    return df


def add_timedelta_column(
        df: pl.DataFrame, 
        timestamp_column='Timestamp', 
        timedelta_column='Time',
        time_unit='ms') -> pl.DataFrame:
    """Create a new column that contains the time from Timestamp in ms."""
    # NOTE: saving timedelta to csv runs into problems, maybe we can do without it for now / just use it for debuggings
    df = df.with_columns(
        pl.col(timestamp_column)
        .cast(pl.Duration(time_unit=time_unit))
        .alias(timedelta_column)
    )
    return df


@map_trials
def interpolate_to_marker_timestamps(
        df: pl.DataFrame,
        timestamp_column='Timestamp') -> pl.DataFrame:
    # Define a custom function for the transformation
    # TODO;NOTE: maybe there is a better way to do this: 
    # - https://docs.pola.rs/user-guide/expressions/null/#filling-missing-data
    # - especially https://docs.pola.rs/user-guide/expressions/null/#fill-with-interpolation
    """
    We define the timestamp where the marker was send as the first measurement timestamp 
    of the device to have the exact same trial duration for each modality 
    (different devices have different sampling rates). This shifts the data by about 5 ms
    and could be interpreted as an interpolation.
    """
    # Only do if there are nulls in the df which means that there are still the empty marker events were no data was recorded
    # Else we could change values that are not supposed to be changed
    if sum(df.null_count()).item() == 0:
        return df

    # Get the first and last timestamp of the group
    # TODO: NOTE: there is a difference between using the integer indexing and boolean indexing below
    # - we should decide depending on how duplicate timestamps are handled
    # especially in what order duplicate timestamps are removed
    first_timestamp = df[timestamp_column][0]
    second_timestamp = df[timestamp_column][1]
    second_to_last_timestamp = df[timestamp_column][-2]
    last_timestamp = df[timestamp_column][-1]

    # Replace the second and second-to-last timestamps
    return df.with_columns(
        pl.when(pl.col(timestamp_column) == df[timestamp_column][1])
        .then(first_timestamp)
        .when(pl.col(timestamp_column) == df[timestamp_column][-2])
        .then(last_timestamp)
        .otherwise(pl.col(timestamp_column))
        .alias(timestamp_column)
    ).drop_nulls()


@map_trials
def interpolate(df) -> pl.DataFrame:
    """Linearly interpolates the whole DataFrame."""
    return df.interpolate()


def resample(df: pl.DataFrame, ms: int) -> pl.DataFrame:
    if 'Trial' in df.index.names:
        df = df.groupby('Trial').resample(f'{ms}ms', level='Time').mean()
    else:
        df = df.resample(f'{ms}ms', level='Time').mean()
    return df

def resample_to_500hz(df):
    if 'Time' not in df.index.names:
        raise ValueError("Index must contain 'Time' for resampling.")
    if 'Trial' in df.index.names:
        df = df.groupby('Trial').resample('2ms', level='Time').mean()
    else:
        df = df.resample('2ms', level='Time').mean()
    return df


@map_trials
def scale_min_max(
        df: pl.DataFrame,
        exclude_columns=['Timestamp', 'Trial']) -> pl.DataFrame:
    return df.with_columns(
        _scale_min_max_col(
            pl.col(pl.Float64)
            .exclude(exclude_columns))) # TODO: trial shouldn't even be float64

def _scale_min_max_col(col: pl.Expr) -> pl.Expr:
    return (col - col.min()) / (col.max() - col.min())


@map_trials
def scale_standard(
        df: pl.DataFrame, 
        exclude_columns=['Timestamp', 'Trial']) -> pl.DataFrame:  # TODO: trial shouldn't even be float64
    return df.with_columns(
        _scale_standard_col(
            pl.col(pl.Float64)
            .exclude(exclude_columns)))

def _scale_standard_col(col: pl.Expr) -> pl.Expr:
    return (col - col.mean()) / col.std()


def merge_dfs(
        *dfs: pl.DataFrame | List[pl.DataFrame],
        merge_on=['Timestamp', 'Trial'],
        sort_by=['Timestamp']) -> pl.DataFrame:
    """
    Merge multiple DataFrames on the 'Timestamp' and 'Trial' columns. 
    Function accepts both a list of DataFrames or multiple DataFrames as arguments.
    """
    # Flatten in case of a single argument which is an iterable of DataFrames
    if len(dfs) == 1 and isinstance(dfs[0], Iterable):
        dfs = list(dfs[0])

    df = reduce(
        lambda left, right: 
            left.join(right, on=merge_on, how='outer_coalesce').sort(sort_by),
            dfs)
    return df


"""
NOTE: TODO: NEW: scikit-learn’s transformers now support polars output with the set_output API.

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
