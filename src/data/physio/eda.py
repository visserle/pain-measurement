import polars as pl
import neurokit2 as nk

from src.data.transform_data import map_trials

@map_trials
def eda_process(df: pl.DataFrame, sampling_rate) -> pl.DataFrame:
    eda_raw_np = df.select('EDA_RAW').to_numpy().flatten()
    eda_processed_pd = nk.eda_phasic(eda_raw_np, sampling_rate=sampling_rate, method='neurokit')
    return df.hstack(pl.from_pandas(eda_processed_pd))
