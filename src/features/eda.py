# f your GSR data was collected at 100 Hz (100 samples per second), you can safely downsample it to 10 Hz (10 samples per second) or even less.

import neurokit2 as nk
import pandas as pd
import polars as pl

from src.features.transformations import map_trials

EDA_RAW_COLUMN = "EDA_RAW"


@map_trials
def process_eda(
    df: pl.DataFrame,
    sampling_rate: int = 100,  # TODO FIXME hardcoded
) -> pl.DataFrame:
    eda_raw = df.select(EDA_RAW_COLUMN).to_numpy().flatten()
    eda_processed: pd.DataFrame = nk.eda_phasic(
        eda_signal=eda_raw,
        sampling_rate=sampling_rate,
        method="neurokit",
    )  # returns EDA_Phasic and EDA_Tonic columns
    return df.hstack(pl.from_pandas(eda_processed))
