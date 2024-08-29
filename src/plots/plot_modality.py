import polars as pl
from icecream import ic
from polars import col

from src.data.database_manager import DatabaseManager
from src.features.resampling import add_timestamp_μs_column
from src.features.scaling import scale_min_max, scale_standard
from src.features.transforming import merge_data_dfs

info_columns = [
    "trial_number",
    "participant_id",
    "stimulus_seed",
    "rownumber",
    "samplenumber",
]


def plot_modality_over_trials(
    modality: str,
    processing_step: str = None,
    signals: list[str] = None,
    normalize: bool = True,
) -> pl.DataFrame:
    processing_step = processing_step or "feature"

    with DatabaseManager() as db:
        df = db.get_table(processing_step + "_" + modality)
    df = df.drop(
        info_columns,
        strict=False,
    )

    if normalize:
        df = scale_min_max(
            df,
            exclude_additional_columns=[
                "time_bin",
                "rating",
                "temperature",
                "ppg_quality",
            ],
        )

    return df.plot(
        x="timestamp",
        y=signals,
        groupby="trial_id",
        kind="line",
        xlabel="Time (s)",
        ylabel="Normalized value",
    )
