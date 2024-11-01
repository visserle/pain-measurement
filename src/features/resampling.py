# TODO: add interpolation with zero-stuffing for up-sampling, polars does this by default using upsample
# TODO: use butterworth filter for low-pass filtering in the decimate function to avoid
# ripple in the passband


import logging

import polars as pl
import scipy.signal as signal
from polars import col

from src.features.transforming import map_trials

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


@map_trials
def decimate(
    df: pl.DataFrame,
    factor: int,
) -> pl.DataFrame:
    """Decimate all float columns using scipy.signal.decimate (order 8 Chebyshev type I
    filter).

    This function applies scipy.signal.decimate to all float columns in the DataFrame
    (except the 'timestamp' column) and gathers every 'factor' rows.
    """
    if sum(s.count("time") for s in df.columns) > 1:
        logger.warning(
            "More than one time column found. The additional time columns will be "
            "low-pass filtered via the decimate function which may lead to unexpected "
            "results."
        )

    def decimate_column(column: pl.Series) -> pl.Series:
        if column.dtype in [pl.Float32, pl.Float64] and column.name != "timestamp":
            return pl.from_numpy(
                signal.decimate(
                    x=column.to_numpy(),
                    q=factor,
                    ftype="iir",
                    zero_phase=True,
                )
            ).to_series()
        else:
            return column.gather_every(factor)

    return df.select(pl.all().map_batches(decimate_column))
