# NOTE: GSR data collected at 100 Hz can be safely downsampled to 10 Hz or even less.
# TODO: drop irrelevant columns, downsample, etc.

import neurokit2 as nk
import polars as pl
from polars import col

from src.features.resampling import downsample
from src.features.transforming import map_trials

SAMPLE_RATE = 100


def preprocess_eda(df: pl.DataFrame) -> pl.DataFrame:
    df = nk_process_eda(df)
    return df


def feature_eda(df: pl.DataFrame) -> pl.DataFrame:
    df = downsample(df, new_sample_rate=10)
    return df


@map_trials
def nk_process_eda(
    df: pl.DataFrame,
    sample_rate: int = 100,
    method: str = "neurokit",
) -> pl.DataFrame:
    """
    Transform the raw EDA signal into phasic and tonic components using NeuroKit2.

    The default method "neurokit" is based on a high-pass filter of 0.05 Hz as used in
    the BIOPAC algorithm.

    https://www.biopac.com/knowledge-base/phasic-eda-issue/,
    https://github.com/neuropsychology/NeuroKit/blob/1aa8deee392f8098df4fd77a23f696c2ff2d29db/neurokit2/eda/eda_phasic.py#L141
    """
    return (
        df.with_columns(
            col("eda_raw")
            .map_batches(
                lambda x: pl.from_pandas(
                    nk.eda_phasic(
                        eda_signal=x.to_numpy(),
                        sampling_rate=sample_rate,
                        method=method,
                    )
                ).to_struct()
            )
            .alias("eda_components")
        )
        .unnest("eda_components")
        .select(pl.all().name.to_lowercase())
    )
