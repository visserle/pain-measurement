import math
import logging
from functools import wraps

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.signal as signal

from src.data.transform_data import map_trials
from src.log_config import configure_logging

configure_logging(stream_level=logging.DEBUG, ignore_libs=['matplotlib', 'Comm', 'bokeh', 'tornado', 'param'])


EYES = ['Pupillometry_R', 'Pupillometry_L']

def map_eyes(eye_columns=EYES):
    def decorator(func):
        @wraps(func)
        def wrapper(df, *args, **kwargs):
            for eye in eye_columns:
                df = func(df, eye, *args, **kwargs)
            return df
        return wrapper
    return decorator


@map_trials
def process_pupillometry(df, sampling_rate=60, cutoff_frequency=0.5):
    df = values_below_x_are_still_blinks(df)
    df = remove_periods_around_blinks(df)
    df = interolate_pupillometry(df)
    df = low_pass_filter_pupillometry(df, sampling_rate, cutoff_frequency)
    return df


@map_eyes()
def values_below_x_are_still_blinks(df, eye, x=2):
    return df.with_columns(
        pl.when(pl.col(eye) < x)
        .then(-1)
        .otherwise(pl.col(eye))
        .alias(eye)
    )

@map_eyes()
def remove_periods_around_blinks(df, eye, period=100, sampling_rate=60):
    # Remove periods of 100 ms before and after blinks
    # NOTE: Geuter 2014 used 100 ms cut-off before and after each blink

    neg_ones = df[eye] == -1

    # Shift the series to find the transitions
    start_conditions = neg_ones & ~neg_ones.shift(1)
    end_conditions = neg_ones & ~neg_ones.shift(-1)

    # Get the indices where the conditions are True
    start_indices = start_conditions.arg_true().to_list()
    end_indices = end_conditions.arg_true().to_list()

    # Explicitly check for edge cases where the first or last value is -1
    if df[eye][0] == -1:
        start_indices.insert(0, 0)
    if df[eye][-1] == -1:
        end_indices.append(len(df) - 1) # if df you could also use df.height


    # Enlargen each segment (defined by start and end indices) by a given amount
    enlargen_amount = math.ceil(period/(1000/sampling_rate)) # 100 ms
    start_indices = [index - enlargen_amount for index in start_indices]
    end_indices = [index + enlargen_amount for index in end_indices]

    # Ensure that the indices are within the bounds of the dataframe
    start_indices = [max(index, 0) for index in start_indices]
    end_indices = [min(index, len(df) - 1) for index in end_indices]

    # Initialize a condition (false just means rows selected here)
    condition = pl.lit(False)

    # Loop over each segment and update the condition
    for start, end in zip(start_indices, end_indices):
        condition = condition | (pl.arange(0, df.height).is_between(start, end))

    # Using with_column to modify the eye column
    return df.with_columns(
        pl.when(condition)
        .then(None)
        .otherwise(df[eye])
        .alias(eye)
    )


@map_eyes()
def interolate_pupillometry(df, eye):
    # Linearly interpolate and fill edge cases when the first or last value is null
    # NOTE: cubic spline?
    return df.with_columns([
        pl.col(eye)
        .interpolate()
        .forward_fill()
        .backward_fill()
        .alias(eye)
    ])

@map_eyes()
def low_pass_filter_pupillometry(df, eye, sampling_rate, cutoff_frequency=2.0):
    pupil_smoothed = low_pass_filter(df[eye], sampling_rate, cutoff_frequency=0.5)
    return df.with_columns(pl.Series(eye, pupil_smoothed))


def low_pass_filter(data, sampling_rate, cutoff_frequency=2.0):
    """
    Apply a low-pass filter to smooth the data.
    - data: numpy array of pupil size measurements.
    - sampling_rate: rate at which data was sampled (Hz).
    - cutoff_frequency: frequency at which to cut off the high-frequency signal.
    Returns the filtered data.
    """
    # Normalizing the frequency
    normalized_cutoff = cutoff_frequency / (0.5 * sampling_rate)

    # Creating the filter
    b, a = signal.butter(N=2, Wn=normalized_cutoff, btype='low', analog=False)

    # Applying the filter
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


