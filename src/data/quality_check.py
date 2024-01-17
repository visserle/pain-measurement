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

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def check_sample_rate(df, unique_timestamp=False):
    if unique_timestamp:
        df = df.unique('Timestamp').sort('Timestamp') # actually slightly faster than maintain_order=True
    
    timestamp_start = df.group_by("Trial").agg([
        pl.first('Timestamp'),
    ]).sort('Trial').drop('Trial')
    timestamp_end = df.group_by("Trial").agg([
        pl.last('Timestamp'),
    ]).sort('Trial').drop('Trial')
    duration_in_s = (timestamp_end-timestamp_start) / 1000

    samples = df.group_by("Trial").agg([
        pl.count('Timestamp'),
    ]).sort('Trial').drop('Trial')

    sample_rate_per_trial = (samples/duration_in_s)
    sample_rate_mean = (sample_rate_per_trial).mean().item()
    coeff_of_variation = ((sample_rate_per_trial).std() / (sample_rate_per_trial).mean() * 100).item()
    
    logger.debug("Sample rate per trial: %s", np.round(sample_rate_per_trial.to_numpy().flatten(), 2))
    logger.info(f"The mean sample rate is {(sample_rate_mean):.2f}.")
    if coeff_of_variation > 0.5:
        logger.warning(f"Sample rate varies more than 0.5% between trials: {coeff_of_variation:.2f}%.")
