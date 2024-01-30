import neurokit2 as nk
import polars as pl

from src.data.transform_data import map_trials


@map_trials
def eda_process(df: pl.DataFrame, sampling_rate) -> pl.DataFrame:
    eda_raw_np = df.select("EDA_RAW").to_numpy().flatten()
    # Returns EDA_Phasic and EDA_Tonic as pd.DataFrame
    eda_processed_pd = nk.eda_phasic(eda_raw_np, sampling_rate=sampling_rate, method="neurokit")
    return df.hstack(pl.from_pandas(eda_processed_pd))
