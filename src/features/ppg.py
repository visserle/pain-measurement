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
    ppg_raw = df.select("ppg_raw").to_numpy().flatten()
    ppg_processed, _ = nk.ppg_process(
        ppg_raw,
        sampling_rate=sampling_rate,
        method="elgendi",
    )
    # the neurokit functions returns clean, rate, quality and binary peak columns
    df = df.hstack(pl.from_pandas(ppg_processed).drop("PPG_Raw"))
    return df.select(pl.all().name.to_lowercase())
