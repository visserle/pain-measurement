# TODO;
# read https://neuraldatascience.io/7-eeg/erp_filtering.html for filter_eeg function
# Frequency-based analysis of EEG data
# find out why after lowcut=0.5 the data is centered around 0
# maybe improve code quality: https://stackoverflow.com/questions/75057003/how-to-apply-scipy-filter-in-polars-dataframe

import numpy as np
import polars as pl
from scipy import signal

from src.features.filtering import filter_butterworth
from src.features.resampling import downsample
from src.features.transforming import map_trials

SAMPLE_RATE = 500


def preprocess_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = filter_eeg(df)
    return df


def feature_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df


# quick and dirty TODO: improve
@map_trials
def filter_eeg(
    df: pl.DataFrame,
    sample_rate: int = SAMPLE_RATE,
    channel_columns: list[str] = [
        "ch1",
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "ch7",
        "ch8",
    ],
) -> pl.DataFrame:
    for channel in channel_columns:
        data = filter_butterworth(
            df.get_column(channel),
            sample_rate,
            lowcut=1,
            highcut=30,
        )
        series = pl.Series(channel + "_filtered", data)
        df = df.with_columns(series)
    return df


# TODO
def bandpower(df: pl.DataFrame, band: list, fs: int = 500) -> pl.DataFrame:
    """
    Compute the average power of the signal in a specific frequency band.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame with the 'eeg_raw' column.
    band : list
        The frequency band of interest, e.g. [0.5, 4] for delta waves.
    fs : int
        The sampling rate of the signal in Hz.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional column 'bandpower' containing the average
        power of the signal in the specified frequency band.
    """
    # Compute the power spectral density of the signal
    f, Pxx = signal.welch(df["eeg_raw"], fs=fs, nperseg=fs)
    # Compute the average power in the specified frequency band
    bandpower = np.mean(Pxx[(f >= band[0]) & (f <= band[1])])
    df["bandpower"] = bandpower
    return df
