import logging
from pathlib import Path

import duckdb
import polars as pl

from src.data.imotions_data import (
    create_raw_data_dfs,
    create_trials_df,
    load_imotions_data,
)
from src.data.imotions_data_config import data_config
from src.data.seeds_data import seed_data  # noqa (used in query)
from src.data.utils import pl_schema_to_duckdb_schema

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

# Paths
DB_FILE = Path("experiment.duckdb")
participant_data = Path("runs/experiments/participants.csv")
data = Path("data/imotions")


class DatabaseSchema:
    @staticmethod
    def create_participants_table():
        pass

    @staticmethod
    def create_trials_table(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        conn.execute("""
            CREATE SEQUENCE IF NOT EXISTS seq_trials_trial_id START 1;
            CREATE TABLE IF NOT EXISTS Trials (
                trial_id USMALLINT NOT NULL DEFAULT NEXTVAL('seq_trials_trial_id'),
                trial_number USMALLINT,
                participant_id USMALLINT,
                stimulus_seed USMALLINT,
                timestamp_start DOUBLE,
                timestamp_end DOUBLE,
                duration DOUBLE,
                skin_area USMALLINT,
                UNIQUE (trial_number, participant_id)
            );
        """)

    @staticmethod
    def create_seeds_table(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        # Load directly into the database
        conn.execute("CREATE OR REPLACE TABLE Seeds AS SELECT * FROM seed_data")

    @staticmethod
    def create_raw_data_table(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        schema: pl.Schema,
    ) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                trial_id USMALLINT,
                {pl_schema_to_duckdb_schema(schema)},
                UNIQUE (trial_id, rownumber)
            );
        """)
        logger.info(f"Created table '{name}' in the database.")


class DatabaseManager:
    def __init__(
        self,
        db_file: str | Path = DB_FILE,
        auto_connect: bool = True,
    ):
        self.db_file = db_file
        self.conn = None
        if auto_connect:  # no need to call connect() explicitly or via context manager
            self.connect()

    def __enter__(self):
        if not self.conn:
            self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        self.conn = duckdb.connect(str(self.db_file))
        # DatabaseSchema.create_participants_table(self.conn)  # TODO
        DatabaseSchema.create_trials_table(self.conn)
        DatabaseSchema.create_seeds_table(self.conn)

    def disconnect(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def execute(self, query: str):
        """Execute a query on the database. Does not close the connection."""
        try:
            return self.conn.execute(query)
        except AttributeError:
            self.connect()
            return self.execute(query)

    def load_trials(
        self,
        trials_df: pl.DataFrame,
    ):
        columns = ", ".join(trials_df.columns)
        try:
            self.conn.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM trials_df")
        except duckdb.ConstraintException as e:
            logger.warning(f"Trial data already exists in the database: {e}")

    def load_raw_data(
        self,
        name: str,
        raw_data_df: pl.DataFrame,
    ):
        DatabaseSchema.create_raw_data_table(self.conn, name, raw_data_df.schema)
        self.conn.execute(f"""
            INSERT INTO {name}
            SELECT t.trial_id, r.*
            FROM raw_data_df AS r
            JOIN Trials AS t ON r.trial_number = t.trial_number AND r.participant_id = t.participant_id
            ORDER BY r.rownumber;
        """)


def main():
    db = DatabaseManager()
    with db:
        for participant_id in range(1, 24):
            # Load polars dataframes into dictionary from config
            load_imotions_data(data_config, data, participant_id=participant_id)

            # Create metadata table
            trials = create_trials_df(data_config, participant_id=participant_id)
            db.load_trials(trials)

            # Create raw data tables
            raw_data_dfs = create_raw_data_dfs(data_config, trials)
            for key, df in raw_data_dfs.items():
                db.load_raw_data(key, df)


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")
