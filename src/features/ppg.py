"""Heartrate data is directly taken from the Shimmer device."""

import polars as pl
from polars import col

from src.features.filtering import (
    adaptive_ema_smooth,
    butterworth_filter_non_causal,
)
from src.features.resampling import decimate, interpolate_and_fill_nulls
from src.features.transforming import map_trials

SAMPLE_RATE = 100
MAX_HEARTRATE = 120
# 20 bpm above the maximum normal heart_rate to account for pain and stress


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_heart_rate_nulls(df)
    df = low_pass_filter_heart_rate_non_causal(df)
    # order matters, since we reuse the heart_rate column
    df = ema_smooth_heart_rate(df)
    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate(df, factor=10)
    return df


def remove_heart_rate_nulls(
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.with_columns(
        col("ppg_heart_rate_shimmer").cast(pl.Float64),
        pl.when(col("ppg_heart_rate_shimmer") > MAX_HEARTRATE)
        .then(None)
        .when(col("ppg_heart_rate_shimmer") == -1)
        .then(None)
        .otherwise(col("ppg_heart_rate_shimmer"))
        .alias("heart_rate"),
    ).drop("ppg_heart_rate_shimmer")
    # note that the interpolate function already has the map_trials decorator
    # so we don't need to add it at the top of this function
    return interpolate_and_fill_nulls(df, ["heart_rate"])


@map_trials
def ema_smooth_heart_rate(
    df: pl.DataFrame,
    heart_rate_column: str = "heart_rate",
) -> pl.DataFrame:
    """Causal median filter on heart_rate column."""
    return df.with_columns(
        col(heart_rate_column)
        .map_batches(
            lambda x: adaptive_ema_smooth(
                x,
                fast_alpha=0.06,
                slow_alpha=0.006,
                threshold=0.05,
            )
        )
        .alias(heart_rate_column + "_smooth")
    )


@map_trials
def low_pass_filter_heart_rate_non_causal(
    df: pl.DataFrame,
    sample_rate: int = SAMPLE_RATE,
    lowcut: float = 0,
    highcut: float = 0.8,
    order: int = 2,
    heart_rate_column: list[str] = ["heart_rate"],
) -> pl.DataFrame:
    """Low-pass filter the heart_rate data using a butterworth filter. Non-causal."""
    return df.with_columns(
        col(heart_rate_column)
        .map_batches(
            # map_batches to apply the filter to each column
            lambda x: butterworth_filter_non_causal(
                x,
                sample_rate,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
        .name.suffix("_exploratory")
    )


# For neurokit2 library to process the PPG signal, use the following function:

# import neurokit2 as nk


# @map_trials
# def nk_process_ppg(
#     df: pl.DataFrame,
#     sampling_rate: int = SAMPLE_RATE,
# ) -> pl.DataFrame:
#     """
#     Process the raw PPG signal using NeuroKit2 and the "elgendi" method.

#     Creates the following columns:
#     - ppg_clean
#     - ppg_rate
#     - ppg_quality
#     - ppg_peaks

#     Note that neurokit approach is non-causal, i.e. it uses future data to calculate
#     the signal.
#     """

#     return (
#         df.with_columns(
#             col("ppg_raw")
#             .map_batches(
#                 lambda x: pl.from_pandas(
#                     nk.ppg_process(  # returns a tuple, we only need the pd.DataFrame
#                         ppg_signal=x.to_numpy(),
#                         sampling_rate=sampling_rate,
#                         method="elgendi",
#                     )[0].drop("PPG_Raw", axis=1)
#                 ).to_struct()
#             )
#             .alias("ppg_components")
#         )
#         .unnest("ppg_components")
#         .select(pl.all().name.to_lowercase())
#     )
