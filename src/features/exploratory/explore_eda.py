import neurokit2 as nk
import polars as pl
from polars import col
from scipy.signal import detrend

from src.features.exploratory.explore_filtering import non_causal_butterworth_filter
from src.features.exploratory.explore_resampling import non_causal_decimate
from src.features.transforming import map_participants

SAMPLE_RATE = 100


def explore_eda(df: pl.DataFrame) -> pl.DataFrame:
    df = nk_process_eda(df, sample_rate=SAMPLE_RATE)
    df = detrend_tonic_component(df)
    df = non_causal_decimate(df, factor=10)
    return df


@map_participants
def nk_process_eda(
    df: pl.DataFrame,
    sample_rate: int = 100,
    method: str = "neurokit",
) -> pl.DataFrame:
    """
    Transform the raw EDA signal into phasic and tonic components using NeuroKit2 (non-causal).

    The default method "neurokit" is based on a high-pass filter of 0.05 Hz as used in
    the BIOPAC algorithm.

    https://www.biopac.com/knowledge-base/phasic-eda-issue/,
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


@map_participants
def butterworth_eda_decomposition(
    df: pl.DataFrame,
    sample_rate: int = SAMPLE_RATE,
    lowcut: float = 0.05,
    highcut: float = 0,  # 0 means no high-pass filtering for tonic
    order: int = 2,
) -> pl.DataFrame:
    """
    Transform the raw EDA signal into phasic and tonic components using a *causal*
    butterworth high-pass filter of 0.05 Hz for phasic component.
    """
    return df.with_columns(
        [
            # Phasic component: high-pass filtered signal (>= 0.05 Hz)
            col("eda_raw")
            .map_batches(
                lambda x: non_causal_butterworth_filter(
                    x,
                    sample_rate,
                    lowcut=lowcut,
                    highcut=0,  # High-pass filter
                    order=order,
                )
            )
            .alias("eda_phasic"),
            # Tonic component: low-pass filtered signal (< 0.05 Hz)
            col("eda_raw")
            .map_batches(
                lambda x: non_causal_butterworth_filter(
                    x,
                    sample_rate,
                    lowcut=0,
                    highcut=lowcut,  # Low-pass filter
                    order=order,
                )
            )
            .alias("eda_tonic"),
        ]
    )


@map_participants
def detrend_tonic_component(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Detrend the tonic component of the EDA signal.
    This is a non-causal filter. Do not use it for real-time processing.
    """

    return df.with_columns(
        col("eda_tonic")
        .map_batches(
            lambda x: detrend(
                x.to_numpy(),
            )
        )
        .name.suffix("_detrended")
    )
