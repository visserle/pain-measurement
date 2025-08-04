import logging

import mne
import polars as pl

from src.features.transforming import map_trials

SAMPLE_RATE = 500  # original sample rate; we will decimate to 250 Hz
CHANNELS = ["f3", "f4", "c3", "cz", "c4", "p3", "p4", "oz"]

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

mne.set_log_level(verbose=False)


def preprocess_eeg(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate_eeg(df, factor=2)
    df = highpass_filter_eeg(df, cutoff=0.5, sfreq=250)
    df = remove_line_noise(df, sfreq=250)
    return df


def feature_eeg(df: pl.DataFrame) -> pl.DataFrame:
    return df


@map_trials
def decimate_eeg(
    df: pl.DataFrame,
    factor: int,
    channels: list[str] = CHANNELS,
) -> pl.DataFrame:
    """Decimate EEG channels using mne.filter.resample and gather other columns.

    This function applies mne.filter.resample to EEG channels as a matrix and
    gathers every nth row for all other columns, where n is the decimation factor.

    Note that this code does not use polars-native map-batches here as it does not work
    well with MNE.
    """
    # Sanity check for time columns
    if sum(s.count("time") for s in df.columns) > 1:
        logger.warning(
            "More than one time column found. The additional time columns will be "
            "decimated which may lead to unexpected results."
        )

    # Resample EEG channels
    eeg_data = df.select(channels).to_numpy().T
    resampled_eeg = mne.filter.resample(eeg_data, down=factor, method="polyphase")

    # Create a new DataFrame with resampled EEG data
    resampled_df = pl.DataFrame(
        {channel: resampled_eeg[i] for i, channel in enumerate(channels)}
    )

    # Gather every nth row for non-EEG columns
    other_columns = [column for column in df.columns if column not in channels]
    gathered_df = df.select(other_columns).gather_every(factor)

    # Ensure that the gathered DataFrame has the same height as the resampled DataFrame
    # (polars and MNE work in this regard)
    if gathered_df.height != resampled_df.height:
        gathered_df = gathered_df.head(resampled_df.height)

    # Combine resampled EEG data with gathered non-EEG columns
    return pl.concat([gathered_df, resampled_df], how="horizontal")


@map_trials
def highpass_filter_eeg(
    df: pl.DataFrame,
    cutoff: float,
    sfreq: int,
    channels: list = CHANNELS,
) -> pl.DataFrame:
    # Filter EEG channels
    eeg_data = df.select(channels).to_numpy().T

    filtered_eeg = mne.filter.filter_data(
        eeg_data,
        sfreq,
        l_freq=cutoff,
        h_freq=None,
    )

    # Create a new DataFrame with filtered EEG data
    filtered_df = pl.DataFrame(
        {channel: filtered_eeg[i] for i, channel in enumerate(channels)}
    )

    info_columns = [column for column in df.columns if column not in channels]
    info_df = df.select(info_columns)

    # Combine filtered EEG data with non-EEG columns
    return pl.concat([info_df, filtered_df], how="horizontal")


@map_trials
def remove_line_noise(
    df: pl.DataFrame,
    sfreq: int,
    notch_freq: int = 50,
    channels: list = CHANNELS,
) -> pl.DataFrame:
    """ """
    # Filter EEG channels
    eeg_data = df.select(channels).to_numpy().T
    filtered_eeg = mne.filter.notch_filter(
        eeg_data, Fs=sfreq, freqs=[notch_freq, notch_freq * 2]
    )

    # Create a new DataFrame with filtered EEG data
    filtered_df = pl.DataFrame(
        {channel: filtered_eeg[i] for i, channel in enumerate(channels)}
    )

    info_columns = [column for column in df.columns if column not in channels]
    info_df = df.select(info_columns)

    # Combine filtered EEG data with non-EEG columns
    return pl.concat([info_df, filtered_df], how="horizontal")
