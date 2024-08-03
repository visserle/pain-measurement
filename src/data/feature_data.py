import logging

import duckdb
import polars as pl

from src.data.data_config import DataConfig

DB_FILE = DataConfig.DB_FILE

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def create_feature_data_dfs() -> dict[str, pl.DataFrame]:
    pass
