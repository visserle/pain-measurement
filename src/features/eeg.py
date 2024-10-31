# TODO;
# read https://neuraldatascience.io/7-eeg/erp_filtering.html for filter_eeg function
# Frequency-based analysis of EEG data
# find out why after lowcut=0.5 the data is centered around 0
# maybe improve code quality: https://stackoverflow.com/questions/75057003/how-to-apply-scipy-filter-in-polars-dataframe


import numpy as np
import polars as pl
from polars import col
from scipy import signal

from src.features.filtering import butterworth_filter
from src.features.resampling import downsample
from src.features.transforming import map_trials

SAMPLE_RATE = 500
CHANNELS = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]


def preprocess_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = filter_eeg(df)
    return df


def feature_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df


@map_trials
def filter_eeg(
    df: pl.DataFrame,
    sample_rate: int = SAMPLE_RATE,
    channel_columns: list[str] = CHANNELS,
) -> pl.DataFrame:
    return df.with_columns(
        col(channel_columns).map_batches(
            lambda x: butterworth_filter(
                x.to_numpy(),
                sample_rate,
                lowcut=1,
                highcut=35,
            )
        )  # .name.suffix("_filtered") TODO naming convention
    )


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
