# TODO;
# Frequency-based analysis of EEG data

import numpy as np
import polars as pl
from scipy import signal

from src.features.resampling import downsample


# sample rate is 500Hz
def preprocess_eeg(df: pl.DataFrame) -> pl.DataFrame:
    # df = bandpower(df, [0.5, 4])
    return df


def feature_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
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
