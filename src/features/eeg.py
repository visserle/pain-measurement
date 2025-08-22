import logging

import polars as pl
from scipy import signal

from src.features.transforming import map_participants

SAMPLE_RATE = 500  # original sample rate; we will decimate to 250 Hz
FINAL_SAMPLE_RATE = 250  # final sample rate after decimation
CHANNELS = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def feature_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate_eeg(df, factor=2)
    df = highpass_filter_eeg(df, cutoff=0.5, sfreq=FINAL_SAMPLE_RATE)
    df = line_noise_filter(df, sfreq=FINAL_SAMPLE_RATE)
    return df


@map_participants
def decimate_eeg(
    df: pl.DataFrame,
    factor: int,
    channels: list[str] = CHANNELS,
) -> pl.DataFrame:
    """Causal decimation using anti-aliasing filter followed by downsampling."""

    # Design anti-aliasing filter (lowpass at Nyquist/factor)
    cutoff_fraction = 0.6 / factor
    b, a = signal.butter(6, cutoff_fraction, btype="low")

    """
    Nyquist frequency = 250 Hz / 2 = 125 Hz
    Cutoff frequency = 0.6 * 125 Hz = 75 Hz
    """

    # Get EEG data and apply anti-aliasing filter
    eeg_data = df.select(channels).to_numpy().T
    filtered_eeg = signal.lfilter(b, a, eeg_data, axis=1)

    # Downsample by taking every nth sample
    decimated_eeg = filtered_eeg[:, ::factor]

    # Create new DataFrame with decimated EEG data
    decimated_df = pl.DataFrame(
        {channel: decimated_eeg[i] for i, channel in enumerate(channels)}
    )

    # Handle other columns by gathering every nth row
    other_columns = [column for column in df.columns if column not in channels]
    if other_columns:
        gathered_df = df.select(other_columns).gather_every(factor)

        # Ensure consistent heights
        if gathered_df.height != decimated_df.height:
            gathered_df = gathered_df.head(decimated_df.height)

        return pl.concat([gathered_df, decimated_df], how="horizontal")
    else:
        return decimated_df


@map_participants
def highpass_filter_eeg(
    df: pl.DataFrame,
    cutoff: float,
    sfreq: int,
    channels: list = CHANNELS,
) -> pl.DataFrame:
    """Causal highpass filter using forward-only IIR filtering."""

    # Design Butterworth highpass filter
    nyquist = sfreq / 2
    normalized_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype="high", analog=False)

    # Get EEG data and apply causal filter
    eeg_data = df.select(channels).to_numpy().T
    filtered_eeg = signal.lfilter(b, a, eeg_data, axis=1)

    # Reconstruct DataFrame
    filtered_df = pl.DataFrame(
        {channel: filtered_eeg[i] for i, channel in enumerate(channels)}
    )

    info_columns = [column for column in df.columns if column not in channels]
    info_df = df.select(info_columns)

    return pl.concat([info_df, filtered_df], how="horizontal")


@map_participants
def line_noise_filter(
    df: pl.DataFrame,
    sfreq: int,
    notch_freq: int = 50,
    quality_factor: int = 30,
    channels: list = CHANNELS,
) -> pl.DataFrame:
    """Causal notch filter for line noise removal."""

    # Design notch filters for fundamental and harmonic
    nyquist = sfreq / 2

    # Primary notch filter
    freq_norm_1 = notch_freq / nyquist
    b1, a1 = signal.iirnotch(freq_norm_1, quality_factor)

    # Harmonic notch filter (2x frequency)
    freq_norm_2 = (notch_freq * 2) / nyquist
    b2, a2 = signal.iirnotch(freq_norm_2, quality_factor)

    # Get EEG data
    eeg_data = df.select(channels).to_numpy().T

    # Apply both notch filters causally
    temp_filtered = signal.lfilter(b1, a1, eeg_data, axis=1)  # Remove fundamental
    filtered_eeg = signal.lfilter(b2, a2, temp_filtered, axis=1)  # Remove harmonic

    # Reconstruct DataFrame
    filtered_df = pl.DataFrame(
        {channel: filtered_eeg[i] for i, channel in enumerate(channels)}
    )

    info_columns = [column for column in df.columns if column not in channels]
    info_df = df.select(info_columns)

    return pl.concat([info_df, filtered_df], how="horizontal")
