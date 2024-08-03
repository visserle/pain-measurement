import logging

import duckdb
import polars as pl

from src.data.data_config import DataConfig
from src.data.feature_data import create_feature_data_dfs
from src.data.imotions_data_to_raw_data import (
    create_raw_data_dfs,
    create_trials_df,
    load_imotions_data_dfs,
)
from src.data.preprocessed_data import create_preprocessed_data_dfs
from src.data.seeds_data import seed_data  # noqa (used in db query)
from src.data.utils import pl_schema_to_duckdb_schema

NUM_PARTICIPANTS = DataConfig.NUM_PARTICIPANTS
DB_FILE = DataConfig.DB_FILE

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DatabaseSchema:
    """
    Database schema for the experiment data.

    Consists of:
        - metadata tables for participants, trials, seeds, etc. and
        - data tables for raw, preprocessed, and feature data.

    Note: participant_id and trial_number are denormalized columns to avoid joins.
    """

    @staticmethod
    def create_database_tables():
        pass

    @staticmethod
    def create_participants_table(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Participants (
                participant_id INTEGER PRIMARY KEY,
                age INTEGER,
                gender CHAR
            );
        """)
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
                -- FOREIGN KEY (participant_id) REFERENCES participants(participant_id) #  TODO
            );
        """)

    @staticmethod
    def create_seeds_table(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        # Load directly into the database (not from CSV)
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
        """)  # trial_id will be added via join with Trials
        logger.info(f"Created table '{name}' in the database.")

    @staticmethod
    def create_preprocessed_data_table(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        schema: pl.Schema,
    ) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
            {pl_schema_to_duckdb_schema(schema)},
            UNIQUE (trial_id, rownumber)
            );
        """)
        logger.info(f"Created table '{name}' in the database.")

    @staticmethod
    def create_feature_data_table(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        schema: pl.Schema,
    ) -> None:
        # same schema as preprocessed data
        return DatabaseManager.create_preprocessed_data_table(conn, name, schema)


class DatabaseManager:
    """
    Database manager for the experiment data.

    Note that DuckDB and Polars share the same memory space.

    Example usage:
    >>> db = DatabaseManager()
    >>> with db:
    >>>    df = db.execute("SELECT * FROM Trials").pl()
    >>>    df.head()

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
                "Database is not connected. Use 'with' statement or call connect() first."
            )
        return self.conn.execute(query)

    def sql(self, query: str):
        """Run a SQL query. If it is a SELECT statement, create a relation object from
        the given SQL query, otherwise run the query as-is.
        """
        if self.conn is None:
            raise ConnectionError(
                "Database is not connected. Use 'with' statement or call connect() first."
            )
        return self.conn.sql(query)

    def insert_trials(
        self,
        trials_df: pl.DataFrame,
    ) -> None:
        columns = ", ".join(trials_df.columns)
        try:
            self.conn.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM trials_df")
        except duckdb.ConstraintException as e:
            logger.warning(f"Trial data already exists in the database: {e}")

    def _participant_exists(
        self,
        participant_id: int,
    ) -> bool:
        result = self.execute(
            f"SELECT 1 FROM Trials WHERE participant_id = {participant_id} LIMIT 1"
        )
        return bool(result.fetchone())

    def insert_raw_data(
        self,
    ) -> None:
        """
        Load iMotions data from CSV files into the database.

        As the data extraction from the iMotions format is more complex, most of it is
        done in the `imotions_data_to_raw_data.py` module.
        """
        for participant_id in range(1, NUM_PARTICIPANTS + 1):
            # Check if participant exists
            if self._participant_exists(participant_id):
                logger.debug(
                    f"Participant {participant_id} already exists in the database."
                )
                continue

            # Load iMotions data
            imotions_data_dfs = load_imotions_data_dfs(participant_id)
            # Create trials DataFrame
            trials_df = create_trials_df(
                participant_id,
                imotions_data_dfs["iMotions_Marker"],
            )
            # Insert trials into database to get trial IDs
            self.insert_trials(trials_df)
            # Create raw data DataFrames
            raw_data_dfs = create_raw_data_dfs(imotions_data_dfs, trials_df)
            # Insert raw data into database
            for table_name, df in raw_data_dfs.items():
                DatabaseSchema.create_raw_data_table(
                    self.conn,
                    table_name,
                    df.schema,
                )
                self.conn.execute(f"""
                INSERT INTO {table_name}
                SELECT t.trial_id, r.*
                FROM {table_name} AS r
                JOIN Trials AS t ON r.trial_number = t.trial_number AND r.participant_id = t.participant_id
                ORDER BY r.rownumber;
                """)

    def insert_preprocessed_data(
        self,
        preprocessed_data_dfs: dict[str, pl.DataFrame],
    ) -> None:
        # TODO: add exclusion of participants and trials
        for table_name, df in preprocessed_data_dfs.items():
            DatabaseSchema.create_preprocessed_data_table(
                self.conn,
                table_name,
                df.schema,
            )
            try:
                self.conn.execute(f"""
                    INSERT INTO {table_name}
                    SELECT *
                    FROM {table_name}
                    ORDER BY trial_id, timestamp;
                """)
            except duckdb.ConstraintException:
                logger.debug(
                    f"Preprocessed data '{table_name}' already exists in the database."
                )

    def insert_feature_data(
        self,
        feature_data_dfs: dict[str, pl.DataFrame],
    ) -> None:
        pass


def main():
    with DatabaseManager() as db:
        # Insert raw participant data
        db.insert_raw_data()

        # Insert preprocessed data
        preprocessed_data_dfs = create_preprocessed_data_dfs()
        db.insert_preprocessed_data(preprocessed_data_dfs)

        # Insert feature data
        feature_data_dfs = create_feature_data_dfs()
        db.insert_feature_data(feature_data_dfs)


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG, stream_milliseconds=True)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")
