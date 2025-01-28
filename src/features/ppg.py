"""Heartrate data is directly taken from the Shimmer device.
(See below for a neurokit2 approach for processing the PPG signal.)"""

import polars as pl
from polars import col

from src.features.filtering import butterworth_filter
from src.features.resampling import decimate, interpolate_and_fill_nulls
from src.features.transforming import map_trials

SAMPLE_RATE = 100
MAX_HEARTRATE = 120
# 20 bpm above the maximum normal heartrate to account for pain and stress


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = remove_heartrate_nulls(df)
    df = low_pass_filter_ppg(df)
    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = decimate(df, factor=10)
    return df


def remove_heartrate_nulls(
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(col("ppg_heartrate_shimmer") > MAX_HEARTRATE)
        .then(None)
        .when(col("ppg_heartrate_shimmer") == -1)
        .then(None)
        .otherwise(col("ppg_heartrate_shimmer"))
        .alias("heartrate")
    )
    # note that the interpolate function already has the map_trials decorator
    # so we don't need to add it at the top of this function
    return interpolate_and_fill_nulls(df, ["heartrate"])


@map_trials
def low_pass_filter_ppg(
    df: pl.DataFrame,
    sample_rate: float = SAMPLE_RATE,
    lowcut: float = 0,
    highcut: float = 0.8,
    order: int = 2,
    heartrate_column: list[str] = ["heartrate"],
) -> pl.DataFrame:
    """Low-pass filter the heartrate data using a butterworth filter.
    This filter has the function to turn the stepwise signal into a smooth one.
    (Not physically motivated, just to make sure that the linear interpolated data
    from the previous step plus the original stepwise data is not too far off.)
    """
    return df.with_columns(
        col(heartrate_column).map_batches(
            # map_batches to apply the filter to each column
            lambda x: butterworth_filter(
                x,
                SAMPLE_RATE,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
    )


# If you want to use the neurokit2 library to process the PPG signal, you can use the
# following function:

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
