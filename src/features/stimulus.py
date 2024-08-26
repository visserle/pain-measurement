import polars as pl

from src.features.scaling import scale_min_max, scale_percent_to_decimal


def preprocess_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_rating(df)
    df = scale_temperature(df)
    return df


def feature_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    # no need to downsample, as the stimulus data is already at 10 Hz
    return df


def scale_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'rating' column to the range [0, 1] by dividing by 100."""
    return scale_percent_to_decimal(df, exclude_additional_columns=["temperature"])


def scale_temperature(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'temperature' column using min-max scaling (for each trial).

    NOTE: This function should not be used in the ML pipeline due to data leakage of
    the 'temperature' column. However, we don't yet if want to use 'temperature' as a
    target, so this function is included for now. Also the data leakeage is not
    significant. TODO: actually the range is known beforehand, so it's not a problem.

    NOTE: scale_min_max automatically maps to trials.

    TODO, BUG: the actual max values should be retrieved from the calibration
    results, not from the data itself. (low priority, bc temperature is not used as a
    target). Trial max =/= global max (inaccuaracy of about 5 %).
    """
    return scale_min_max(df, exclude_additional_columns=["rating"])
