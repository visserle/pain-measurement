# TODO
# maybe we first should do:
# - smooth the signal
# - resample with equidistant timestamps
# NOTE: this approach is non-causal, i.e. it uses future data to smooth the signal. TODO

import neurokit2 as nk
import polars as pl

from src.features.resampling import downsample
from src.features.transforming import map_trials


def preprocess_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = nk_process_ppg(df)
    return df


def feature_ppg(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df


@map_trials
def nk_process_ppg(
    df: pl.DataFrame,
    sampling_rate: int = 100,
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
                    nk.ppg_process(  # returns a tuple, we only need the dataframe
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
