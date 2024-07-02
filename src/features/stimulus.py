import polars as pl

from src.features.aggregation import calculate_corr
from src.features.scaling import scale_min_max, scale_percent_to_decimal


def add_skin_area(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reconstructs the applied skin areas on the left forearm from the participant id and
    trial number.

    The skin area distribution is as follows:

    |---|---|
    | 1 | 4 |
    | 5 | 2 |
    | 3 | 6 |
    |---|---|

    Each skin area is stimulated twice, where one trial is 3 minutes long.
    For particpants with an even id the stimulation order is:
    6 -> 5 -> 4 -> 3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1 -> end.
    For participants with an odd id the stimulation order is:
    1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> end.
    """
    return df.with_columns(
        pl.when(pl.col("Participant") % 2 == 1)
        .then((pl.col("Trial") % 6) + 1)
        .otherwise(6 - (pl.col("Trial") % 6))
        .alias("Skin_Area")
        .cast(pl.Int8)
    )


def process_stimulus(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the 'Stimulus' data by scaling the 'Rating' and 'Temperature' columns to
    [0, 1].

    NOTE: This function should not be used in the ML pipeline due to data leakage of
    the 'Temperature' column. However, we don't yet if want to use 'Temperature' as a
    target, so this function is included for now. Also the data leakeage is not
    significant. TODO
    """
    df = scale_rating(df)
    df = scale_temperature(df)
    return df


def scale_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'Rating' column to the range [0, 1] by dividing by 100."""
    return scale_percent_to_decimal(df, exclude_additional_columns=["Temperature"])


def scale_temperature(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize the 'Temperature' column using min-max scaling (for each trial).

    NOTE: This function should not be used in the ML pipeline due to data leakage of
    the 'Temperature' column. However, we don't yet if want to use 'Temperature' as a
    target, so this function is included for now. Also the data leakeage is not
    significant. TODO"""
    return scale_min_max(df, exclude_additional_columns=["Rating"])


def corr_temperature_rating(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate the correlation between 'Temperature' and 'Rating' for each trial."""
    return calculate_corr(df, "Temperature", "Rating")
