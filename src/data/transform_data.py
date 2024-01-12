import pandas as pd
import numpy as np


def create_timedelta_index(df):
    """Convert the time stamp to time delta and set it as index."""
    # just casting to timedelta64[ms] is faster but less accurate
    df["Time"] = pd.to_timedelta(df["Timestamp"], unit='ms').round('ms').astype('timedelta64[ms]')
    df.set_index("Time", append=True if 'Trial' in df.index.names else False, inplace=True)
    # Remove duplicate index
    df = df[~df.index.duplicated(keep='first')]
    return df


def create_trial_index(df):
    """Create a trial index based on the MarkerDescription which contains the stimulus seed and is originally send once at the start and end of each trial."""
    # 0. Check if all trials are complete
    # TODO
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


def interpolate(df, method='linear'):
    columns_to_interpolate = df.columns[(df.dtypes == float)]
    if 'Trial' in df.index.names:
        df[columns_to_interpolate] = df.groupby('Trial')[columns_to_interpolate].transform(lambda x: x.interpolate(method=method))
    else:
        df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method=method)
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