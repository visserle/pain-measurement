# TODO:
# - check if modality already exists in the database before preprocessing it
# - add labels function for feature data
# - remove duplcate timestamps from shimmer sensor data (maybe this also exists in eeg data)
# - add database for quality control (e.g. if the number of rows in the raw data is the same as in the preprocess data)
# - performance https://docs.pola.rs/user-guide/expressions/user-defined-functions/ (maybe)
# - add participant data to the database
# - add questionnaires to the database
# - add calibration data to the database
# - add measurement data to the database
# - add excluded data information to the database and check in the preprocessing process if the data is excluded

import logging

import duckdb
import polars as pl
from icecream import ic

from src.data.data_config import DataConfig
from src.data.data_processing import (
    create_feature_data_df,
    create_preprocess_data_df,
    create_raw_data_df,
    create_trials_df,
)
from src.data.database_schema import DatabaseSchema
from src.data.imotions_data import load_imotions_data_df

MODALITIES = DataConfig.MODALITIES
NUM_PARTICIPANTS = DataConfig.NUM_PARTICIPANTS
DB_FILE = DataConfig.DB_FILE


logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DatabaseManager:
    """
    Database manager for the experiment data.

    Note that DuckDB and Polars share the same memory space, so it is possible to
    pass Polars DataFrames to DuckDB queries. As the data is highly compressed by Apache
    Arrow, for each processing step, all modalities are loaded into memory at once,
    processed, and then inserted into the database.

    DuckDB allows only one connection at a time, so the usage as a context manager is
    recommended.

    Example usage:
    >>> db = DatabaseManager()
    >>> with db:
    >>>    df = db.execute("SELECT * FROM Trials").pl()  # .pl() for Polars DataFrame
    >>> # alternatively
    >>>    df = db.read_table("Trials")
    >>> df.head()
    """

    def __init__(self):
        self.conn = None
        self._initialize_tables()

    @staticmethod
    def _initialize_tables():
        with duckdb.connect(DB_FILE.as_posix()) as conn:
            # DatabaseSchema.create_participants_table(self.conn)  # TODO
            DatabaseSchema.create_trials_table(conn)
            DatabaseSchema.create_seeds_table(conn)

    def connect(self) -> None:
        if not self.conn:
            self.conn = duckdb.connect(DB_FILE.as_posix())

    def disconnect(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def execute(self, query: str):
        """Execute a SQL query."""
        if self.conn is None:
            raise ConnectionError(
                "Database not connected. Use 'with' statement or call connect() first."
            )
        return self.conn.execute(query)

    def sql(self, query: str):
        """Run a SQL query. If it is a SELECT statement, create a relation object from
        the given SQL query, otherwise run the query as-is. (see DuckDB docs)
        """
        if self.conn is None:
            raise ConnectionError(
                "Database not connected. Use 'with' statement or call connect() first."
            )
        return self.conn.sql(query)

    def read_table(
        self,
        table_name: str,
        remove_invalid: bool = False,
    ) -> pl.DataFrame:
        """Return the data from a table as a Polars DataFrame."""
        if remove_invalid:
            pass  # TODO
        return self.execute(f"SELECT * FROM {table_name}").pl()

    def table_exists(
        self,
        name: str,
    ) -> bool:
        return (
            self.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables
                WHERE table_name = '{name}'
                """).fetchone()[0]
            > 0
        )

    def participant_exists(
        self,
        participant_id: int,
        table_name: str = "Trials",
    ) -> bool:
        """Check if a participant exists in the database.

        If they exist in Trials, they exist in all raw data tables.
        """
        if "Raw" in table_name:
            table_name = "Trials"
        if not self.table_exists(table_name):
            return False
        result = self.execute(
            f"SELECT True FROM {table_name} WHERE participant_id = {participant_id} LIMIT 1"
        ).fetchone()
        return bool(result)

    def insert_trials(
        self,
        trials_df: pl.DataFrame,
    ) -> None:
        columns = ", ".join(trials_df.columns)
        try:
            self.conn.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM trials_df")
        except duckdb.ConstraintException as e:
            logger.warning(f"Trial data already exists in the database: {e}")

    def insert_raw_data(
        self,
        participant_id: int,
        table_name: str,
        raw_data_df: pl.DataFrame,
    ) -> None:
        DatabaseSchema.create_raw_data_table(
            self.conn,
            table_name,
            raw_data_df.schema,
        )
        self.conn.execute(f"""
            INSERT INTO {table_name}
            SELECT t.trial_id, r.*
            FROM raw_data_df AS r
            JOIN Trials AS t 
                ON r.trial_number = t.trial_number 
                AND r.participant_id = t.participant_id
            ORDER BY r.rownumber;
        """)

    # Note that in constrast to raw data, preprocessed and feature-engineered data is not
    # inserted into the database per participant, but per modality over all
    # participants.
    def insert_preprocess_data(
        self,
        table_name: str,
        preprocess_data_df: pl.DataFrame,
    ) -> None:
        DatabaseSchema.create_preprocess_data_table(
            self.conn,
            table_name,
            preprocess_data_df.schema,
        )
        self.conn.execute(f"""
            INSERT INTO {table_name}
            SELECT *
            FROM preprocess_data_df
            ORDER BY trial_id, timestamp;
        """)

    def insert_feature_data(
        self,
        table_name: str,
        feature_data_df: pl.DataFrame,
    ) -> None:
        # same as preprocess data for now TODO FIXME
        self.insert_preprocess_data(table_name, feature_data_df)


def main():
    with DatabaseManager() as db:
        # Raw data
        for participant_id in range(1, NUM_PARTICIPANTS + 1):
            if db.participant_exists(participant_id):
                logger.debug(
                    f"Raw data for participant {participant_id} already exists."
                )
                continue
            df = load_imotions_data_df(participant_id, "Trials")
            trials_df = create_trials_df(participant_id, df)
            db.insert_trials(trials_df)

            for modality in MODALITIES:
                df = load_imotions_data_df(participant_id, modality)
                df = create_raw_data_df(participant_id, df, trials_df)
                db.insert_raw_data(participant_id, "Raw_" + modality, df)
            logger.debug(f"Raw data for participant {participant_id} inserted.")
        logger.info("Raw data inserted.")

        # preprocessed data
        # no check for existing data as it will be overwritten
        for modality in MODALITIES:
            table_name = "Preprocess_" + modality
            df = db.read_table("Raw_" + modality)
            df = create_preprocess_data_df(table_name, df)
            db.insert_preprocess_data(table_name, df)
        logger.info("Data preprocessed.")

        # Feature-engineered data
        for modality in MODALITIES:
            table_name = f"Feature_{modality}"
            df = db.read_table(f"Preprocess_{modality}")
            df = create_feature_data_df(table_name, df)
            db.insert_feature_data(table_name, df)
        logger.info("Data feature-engineered.")

        logger.info("Database processing complete.")


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")
