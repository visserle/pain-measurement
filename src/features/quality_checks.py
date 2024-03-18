import logging

import numpy as np
import polars as pl

# TODO: write results to participant config json
# TODO: add more quality checks based on non-trialized imotions data
# - packet loss
# - battery
# - Check stimuli length, must be >= stimuli.duration
# where we create a stimulus object from the config file with the right seed, etc.
# Also check after resampling, interpolation, etc.

# # search for nan values in raw data (trial + data)
# -> samplenumber, should be increasing by 1 all the time

# also check for floating point weirdness in timestamps -> bad sign

# check for number of imotions events = stim len * sample rate

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def check_sample_rate(df, unique_timestamp=False):
    if unique_timestamp:
        # actually slightly faster than maintain_order=True (but not lazy)
        df = df.unique("Timestamp").sort("Timestamp")
        logger.info("Checking sample rate for unique timestamps.")

    timestamp_start = (
        df.group_by("Trial")
        .agg(pl.first("Timestamp"))
        .sort("Trial")
        .select("Timestamp")
    )
    timestamp_end = (
        df.group_by("Trial").agg(pl.last("Timestamp")).sort("Trial").select("Timestamp")
    )

    duration_in_s = (timestamp_end - timestamp_start) / 1000

    samples = (
        df.group_by("Trial")
        .agg(pl.count("Timestamp"))
        .sort("Trial")
        .select("Timestamp")
    )

    sample_rate_per_trial = samples / duration_in_s
    sample_rate_mean = (sample_rate_per_trial).mean().item()
    coeff_of_variation = (
        (sample_rate_per_trial).std() / (sample_rate_per_trial).mean() * 100
    ).item()

    logger.debug(
        "Sample rate per trial: %s",
        np.round(sample_rate_per_trial.to_numpy().flatten(), 2),
    )
    logger.info(f"The mean sample rate is {(sample_rate_mean):.2f}.")
    if coeff_of_variation and coeff_of_variation > 0.5:
        logger.warning(
            f"Sample rate varies more than 0.5% between trials: {coeff_of_variation:.2f}% (coefficient of variation)."
        )


def check_stimuli_length(df):
    pass


def check_battery(df):
    pass


def check_packet_loss(df):
    pass
