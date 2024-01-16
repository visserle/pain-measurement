import numpy as np
import polars as pl
import pandas as pd

# NOTE: square brackets can be used to access columns in polars but do not allow lazy evaluation -> use select, take, etc for performance
# TODO: cast data types for more performance

def apply_func_participant(participant, func):
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


def interpolate_to_marker_timestamps(df):
    # Define a custom function for the transformation
    def replace_timestamps(group_df):
        """
        We define the timestamp where the marker was send as the first measurement timestamp 
        of the device to have the exact same trial duration for each modality 
        (different devices have different sampling rates). This shifts the data by about 5 ms
        and could be interpreted as an interpolation.
        """
        # Get the first and last timestamp of the group
        # TODO: NOTE: there is a difference between using the integer indexing and boolean indexing below
        # - we should decide depending on how duplicate timestamps are handled
        # especially in what order duplicate timestamps are removed
        first_timestamp = group_df["Timestamp"][0]
        second_timestamp = group_df["Timestamp"][1]
        second_to_last_timestamp = group_df["Timestamp"][-2]
        last_timestamp = group_df["Timestamp"][-1]
        
        # Replace the second and second-to-last timestamps
        return group_df.with_columns(
            pl.when(pl.col("Timestamp") == group_df["Timestamp"][1])
            .then(first_timestamp)
            .when(pl.col("Timestamp") == group_df["Timestamp"][-2])
            .then(last_timestamp)
            .otherwise(pl.col("Timestamp"))
            .alias("Timestamp")
        ).drop_nulls()

    # Only if there are nulls in the df
    if sum(df.null_count()).item() == 0:
        return df
    # Apply the custom function to each group
    return df.group_by("Trial", maintain_order=True).map_groups(replace_timestamps)
    

def add_timedelta_column(df: pl.DataFrame):
    # NOTE: saving timedelta to csv runs into problems, maybe we can do without it for now / just use it for debuggings
    """Create a new column that contains the time from Timestamp in ms."""
    df = df.with_columns(
        df['Timestamp']
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


def standardize(df):
    # Exclude 'Timestamp' from the columns to be standardized
    columns_to_standardize = df.columns[(df.dtypes == float) & (df.columns != 'Timestamp')]
    if 'Trial' in df.index.names:
        df[columns_to_standardize] = df.groupby('Trial')[columns_to_standardize].transform(lambda x: (x - x.mean()) / x.std())
    else:
        df[columns_to_standardize] = (df[columns_to_standardize] - df[columns_to_standardize].mean()) / df[columns_to_standardize].std()
    return df


def normalize(df):
    # Exclude 'Timestamp' from the columns to be standardized
    columns_to_normalize = df.columns[(df.dtypes == float) & (df.columns != 'Timestamp')]
    if 'Trial' in df.index.names:
        df[columns_to_normalize] = df.groupby('Trial')[columns_to_normalize].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())
    return df
