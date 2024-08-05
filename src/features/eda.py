# NOTE: GSR data collected at 100 Hz can be safely downsampled to 10 Hz or even less.

import neurokit2 as nk
import pandas as pd
import polars as pl

from src.features.transformations import map_trials


def clean_eda(df: pl.DataFrame) -> pl.DataFrame:
    # TODO: drop irrelevant columns, downsample, etc.
    return df


def feature_eda(df: pl.DataFrame) -> pl.DataFrame:
    df = nk_process_eda(df)
    return df


@map_trials
def nk_process_eda(
    df: pl.DataFrame,
    sampling_rate: int = 100,
) -> pl.DataFrame:
    eda_raw = df.select("eda_raw").to_numpy().flatten()
    eda_processed: pd.DataFrame = nk.eda_phasic(
        eda_signal=eda_raw,
        sampling_rate=sampling_rate,
        method="neurokit",
    )  # this returns EDA_Phasic and EDA_Tonic columns
    return df.hstack(pl.from_pandas(eda_processed))
