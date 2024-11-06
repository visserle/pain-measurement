# NOTE: neurokit approach is non-causal, i.e. it uses future data to calculate the signal. TODO
# Note that the device intern signal processing seems to be causal (ppg_heartrate from
# the raw table)

import neurokit2 as nk
import polars as pl

from src.features.filtering import butterworth_filter
from src.features.resampling import decimate, interpolate_and_fill_nulls
from src.features.transforming import map_trials

SAMPLE_RATE = 100
MAX_HEARTRATE = 100  # TODO: check if this is a good value


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = nk_process_ppg(df)
    df = remove_heartrate_nulls(df)
    df = low_pass_filter_ppg(df)
    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    # df = decimate(df, factor=10)  # TODO test if this is a good idea
    return df


@map_trials
def nk_process_ppg(
    df: pl.DataFrame,
    sampling_rate: int = SAMPLE_RATE,
) -> pl.DataFrame:
    """
    Process the raw PPG signal using NeuroKit2 and the "elgendi" method.

    Creates the following columns:
    - ppg_clean
    - ppg_rate
    - ppg_quality
    - ppg_peaks
    """
    return (
        df.with_columns(
            pl.col("ppg_raw")
            .map_batches(
                lambda x: pl.from_pandas(
                    nk.ppg_process(  # returns a tuple, we only need the pd.DataFrame
                        ppg_signal=x.to_numpy(),
                        sampling_rate=sampling_rate,
                        method="elgendi",
                    )[0].drop("PPG_Raw", axis=1)
                ).to_struct()
            )
            .alias("ppg_components")
        )
        .unnest("ppg_components")
        .select(pl.all().name.to_lowercase())
    )


def remove_heartrate_nulls(
    df: pl.DataFrame,
) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("ppg_heartrate") > MAX_HEARTRATE)
        .then(100)
        .when(pl.col("ppg_heartrate") == -1)
        .then(None)
        .otherwise(pl.col("ppg_heartrate"))
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
    pupil_columns: list[str] = ["heartrate"],
) -> pl.DataFrame:
    return df.with_columns(
        pl.col(
            pupil_columns
        ).map_batches(  # map_batches to apply the filter to each column
            lambda x: butterworth_filter(
                x,
                SAMPLE_RATE,
                lowcut=lowcut,
                highcut=highcut,
                order=order,
            )
        )
    )
