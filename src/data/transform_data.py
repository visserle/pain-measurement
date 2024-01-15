import pandas as pd
import numpy as np


def apply_func_participant(participant, func):
    """Wierd utility function, will be removed in the future, see process_data.py."""
    #TODO: use map instead, e.g.:
    # dict(zip(a, map(f, a.values())))
    # dict(map(lambda item: (item[0], f(item[1])), my_dictionary.items()
    for data in participant.datasets:
        participant.datasets[data].dataset = func(participant.datasets[data].dataset)
    return participant


def create_trial_index(df):
    """Create a trial index based on the stimuli seed which is originally send once at the start and end of each trial."""
    # Check if the trial index already exists
    if 'Trial' in df.index.names:
        return df
    # 0. Check if all trials are complete  # TODO
    # 1. Forward fill and backward fill columns
    ffill = df['Stimuli_Seed'].ffill()
    bfill = df['Stimuli_Seed'].bfill()
    # 2. Where forward fill and backward fill are equal, replace the NaNs in the original Stimuli_Seed
    df['Stimuli_Seed'] = np.where(ffill == bfill, ffill, df['Stimuli_Seed'])
    # 3. Only keep rows where the Stimuli_Seed is not NaN
    df = df[df['Stimuli_Seed'].notna()]
    df['Stimuli_Seed'] = df['Stimuli_Seed'].astype(int)
    # Create a new column that contains the trial number
    df['Trial'] = df.Stimuli_Seed.diff().ne(0).cumsum()
    # Add Trial to the index
    df.set_index('Trial', append=True if 'Time' in df.index.names else False, inplace=True)
    return df

def create_timedelta_index(df):
    """Convert the time stamp to time delta and set it as index."""
    # just casting to timedelta64[ms] is faster but less accurate
    df["Time"] = pd.to_timedelta(df["Timestamp"], unit='ms').round('ms').astype('timedelta64[ms]')
    df.set_index("Time", append=True if 'Trial' in df.index.names else False, inplace=True)
    # Remove duplicate index
    df = df[~df.index.duplicated(keep='first')]
    return df

def reorder_multiindex(df):
    if ('Trial' in df.index.names) and ('Time' in df.index.names):
        df = df.reorder_levels(['Trial', 'Time'])
    return df


def resample(df, ms):
    if 'Time' not in df.index.names:
        raise ValueError("Index must contain 'Time'.")
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
