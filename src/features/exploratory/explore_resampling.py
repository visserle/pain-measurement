# Note that time-columns are mere floats as pl.Duration is not fully supported in Polars
# and DuckDB yet (end of 2024).

import logging

import polars as pl
import scipy.signal as signal
from polars import col
from polars.datatypes.group import FLOAT_DTYPES

from src.features.transforming import map_trials

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


@map_trials
def non_causal_decimate(
    df: pl.DataFrame,
    factor: int,
    time_column: str = "timestamp",
) -> pl.DataFrame:
    """Decimate float columns using scipy.signal.decimate (order 8 Chebyshev type I
    filter) and downsample integer columns by gathering every 'factor'.

    Note that the 'timestamp' column is not decimated, but only downsampled.
    """
    if sum(s.count("time") for s in df.columns) > 1:
        logger.warning(
            "More than one time column found. The additional time columns will be "
            "low-pass filtered via the decimate function which may lead to unexpected "
            "results."
        )

    def decimate_column(column: pl.Series) -> pl.Series:
        if column.dtype in FLOAT_DTYPES and column.name != time_column:
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

    return df.select(
        pl.all().map_batches(decimate_column)
    )  # no need to add aliases here
