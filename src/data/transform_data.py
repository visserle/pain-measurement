from functools import wraps
import logging

import numpy as np
import polars as pl
import pandas as pd

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


# NOTE: square brackets can be used to access columns in polars but do not allow lazy evaluation -> use select, take, etc for performance
# TODO: add **kwars to apply_func_participant
# TODO: cast data types for more performance
# TODO: Round all timestamps to remove floating point weirdness
# 447,929.23030000000006, ...
# 448,929.2894, ...
# -> add as function to transformations of raw data, even imotions data?


def map_trials(func):
    """Decorator to apply a function to each trial in a DataFrame."""
    @wraps(func)
    def wrapper(df, **kwargs):
        # Check if 'df' is a pl.DataFrame
        if not isinstance(df, pl.DataFrame):
            raise ValueError("Input must be a Polars DataFrame.")
        # Check if DataFrame has a 'Trial' column
        if 'Trial' in df.columns:
            # Apply the function to each trial
            result = df.group_by('Trial', maintain_order=True).map_groups(lambda group: func(group, **kwargs))
        else:
            # Apply the function to the whole DataFrame
            logger.warning("No 'Trial' column found, applying function %s to the whole DataFrame instead.", func.__name__)
            logger.info(f"Use {func.__name__}.__wrapped__() to access the function without the map_trials decorator.")
            result = func(df, **kwargs)
        return result
    return wrapper


def apply_func_participant(func, participant):
    """Utility function for debugging, will be removed in the future, see process_data.py."""
    #TODO: use map instead, e.g.:
    # dict(zip(a, map(f, a.values())))
    # dict(map(lambda item: (item[0], f(item[1])), my_dictionary.items()
    for data in participant.datasets:
        participant.datasets[data].dataset = func(participant.datasets[data].dataset)
    return participant


def create_trials(df: pl.DataFrame):
    # TODO: maybe we need to interpolate here for the nan at the start and end of each trial
    """Create a trial column based on the stimuli seed which is originally send only once at the start and end of each trial."""
    # TODO: Check if all trials are complete
    # Forward fill and backward fill columns
    ffill = df['Stimuli_Seed'].fill_null(strategy='forward')
    bfill = df['Stimuli_Seed'].fill_null(strategy='backward')
    # Where forward fill and backward fill are equal, replace the NaNs in the original Stimuli_Seed
    # this is the same as np.where(ffill == bfill, ffill, df['Stimuli_Seed'])
    df = df.with_columns(
        pl.when(ffill == bfill)
        .then(ffill)
        .otherwise(df['Stimuli_Seed'])
        .alias('Stimuli_Seed')
    )
    assert df['Timestamp'].is_sorted(descending=False)
    # Only keep rows where the Stimuli_Seed is not NaN
    df = df.filter(df['Stimuli_Seed'].is_not_null())
    # Create a new column that contains the trial number
    df = df.with_columns(
        pl.col('Stimuli_Seed')
        .diff()                 # Calculate differences
        .fill_null(value=0)     # Replace initial null with 0 because the first trial is always 0
        .ne(0)                  # Check for non-zero differences
        .cum_sum()              # Cumulative sum of boolean values
        .cast(pl.UInt8)         # Cast to integer data type between 0 and 255
        .alias('Trial')         # Rename the series to 'Trial'
    )
    return df


@map_trials
def interpolate_to_marker_timestamps(df):
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
    first_timestamp = df["Timestamp"][0]
    second_timestamp = df["Timestamp"][1]
    second_to_last_timestamp = df["Timestamp"][-2]
    last_timestamp = df["Timestamp"][-1]
    
    # Replace the second and second-to-last timestamps
    return df.with_columns(
        pl.when(pl.col("Timestamp") == df["Timestamp"][1])
        .then(first_timestamp)
        .when(pl.col("Timestamp") == df["Timestamp"][-2])
        .then(last_timestamp)
        .otherwise(pl.col("Timestamp"))
        .alias("Timestamp")
    ).drop_nulls()
    

def add_timedelta_column(df: pl.DataFrame):
    # NOTE: saving timedelta to csv runs into problems, maybe we can do without it for now / just use it for debuggings
    """Create a new column that contains the time from Timestamp in ms."""
    df = df.with_columns(
        pl.col('Timestamp')
        .cast(pl.Duration(time_unit='ms'))
        .alias('Time')
    )
    return df


def resample(df: pl.DataFrame, ms: int):
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


def interpolate(df, method='linear', limit_direction='both'):
    columns_to_interpolate = df.columns[(df.dtypes == float)]
    # When working with data that represents several trials, we need to interpolate each trial separately
    if 'Trial' in df.index.names:
        df[columns_to_interpolate] = df.groupby('Trial')[columns_to_interpolate].transform(lambda x: x.interpolate(method=method, limit_direction=limit_direction))
    else:
        df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method=method, limit_direction=limit_direction)
    return df





def min_max_scaler_col(col: pl.Expr) -> pl.Expr:
    return (col - col.min()) / (col.max() - col.min())

def standard_scaler_col(col: pl.Expr) -> pl.Expr:
    return (col - col.mean()) / col.std()

@map_trials
def min_max_scaler(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (min_max_scaler_col(pl.col(pl.Float64).exclude('Timestamp', 'Trial')))) # TODO: trial shouldn't even be float64

@map_trials
def standard_scaler(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        (standard_scaler_col(pl.col(pl.Float64).exclude('Timestamp', 'Trial')))) # TODO: trial shouldn't even be float64