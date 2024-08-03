import logging

import duckdb
import polars as pl

from src.data.data_config import DataConfig
from src.features.eda import preprocess_eda

DB_FILE = DataConfig.DB_FILE

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

# TODO: go trough the modalities in a for loop and check if it already exists in the database
# before preprocessing it


def create_preprocessed_data_dfs() -> dict[str, pl.DataFrame]:
    with duckdb.connect(DB_FILE.as_posix()) as conn:
        dfs = {}

        # Preprocessed EDA
        df = conn.query("from Raw_EDA").pl()
        dfs["Preprocessed_EDA"] = preprocess_eda(df)

        # Preprocessed Stimulus
        pass

        return dfs
