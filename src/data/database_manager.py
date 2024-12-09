# TODO: add database for quality control (e.g. if the number of rows in the raw data is the same as in the preprocess data)
# TODO: performance https://docs.pola.rs/user-guide/expressions/user-defined-functions/ (maybe)
# TODO: labeled data frame should be part of the database: label based on temperature and rating (not so sure abt the latter)
# - the labeled data should be equidistantly sampled with a sample rate of idk
# - label at the very end when merging all feature data
# - add anonymization to the database in the future

import logging

import duckdb
import polars as pl
from icecream import ic
from polars import col

from src.data.data_config import DataConfig
from src.data.data_processing import (
    create_calibration_results_df,
    create_feature_data_df,
    create_measurement_results_df,
    create_participants_df,
    create_preprocess_data_df,
    create_questionnaire_df,
    create_raw_data_df,
    create_trials_df,
    merge_feature_data_dfs,
)
from src.data.database_schema import DatabaseSchema
from src.data.imotions_data import load_imotions_data_df

DB_FILE = DataConfig.DB_FILE
NUM_PARTICIPANTS = DataConfig.NUM_PARTICIPANTS
QUESTIONNAIRES = DataConfig.QUESTIONNAIRES
MODALITIES = DataConfig.MODALITIES


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
    ```python
    db = DatabaseManager()
    with db:
        df = db.execute("SELECT * FROM Trials").pl()  # .pl() for Polars DataFrame
        # or alternatively
        df = db.get_table("Trials", exclude_invalid_data=False)
    df.head()
    ```
    """

    def __init__(self):
        self.conn = None
        self._initialize_tables()

    @staticmethod
    def _initialize_tables():
        with duckdb.connect(DB_FILE.as_posix()) as conn:
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

    def _ensure_connection(self):
        """Helper method to check connection status."""
        if self.conn is None:
            raise ConnectionError(
                "Database not connected. Use 'with' statement or call connect() first."
            )

    def execute(self, query: str):
        """Execute a SQL query.

        Note: Add a .pl() to the return value to get a Polars DataFrame, e.g.
        `db.execute("SELECT * FROM Feature_EDA").pl()`."""
        self._ensure_connection()
        return self.conn.execute(query)

    def sql(self, query: str):
        """Run a SQL query. If it is a SELECT statement, create a relation object from
        the given SQL query, otherwise run the query as-is. (see DuckDB docs)
        """
        self._ensure_connection()
        return self.conn.sql(query)

    def get_table(
        self,
        table_name: str,
        exclude_invalid_data: bool = True,
    ) -> pl.DataFrame:
        """Return the data from a table as a Polars DataFrame."""
        # TODO: make it more efficient by filtering out invalid trials in the query
        df = self.execute(f"SELECT * FROM {table_name}").pl()
        if exclude_invalid_data:
            invalid_trials = DataConfig.load_invalid_trials()
            if ["participant_id", "trial_number"] in df.columns:
                # Note that not every participant has 12 trials, a naive trial_id filter
                # would remove the wrong trials
                df = df.filter(
                    ~pl.struct(["participant_id", "trial_number"]).is_in(
                        invalid_trials.select(
                            ["participant_id", "trial_number"]
                        ).unique()
                    )
                )
            else:  # not all tables have trial information, e.g. questionnaires
                df = df.filter(
                    ~col("participant_id").is_in(
                        invalid_trials["participant_id"]
                        .value_counts()
                        # TODO: find a better way
                        .filter(col("count") == 12)["participant_id"]
                        .unique()
                    )
                )
                logging.debug(
                    "TODO: find criteria for filtering invalid participants in the exclude_invalid_data kw."
                    # maybe we also should rename it to remove_invalid_data
                )
        return df

    def get_final_feature_data(
        self,
        exclude_invalid_data: bool,
    ) -> pl.DataFrame:
        dfs = [
            self.get_table("Feature_" + modality, exclude_invalid_data)
            for modality in MODALITIES
        ]
        return merge_feature_data_dfs(dfs)

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

    def ctas(
        self,
        table_name: str,
        df: pl.DataFrame,
    ) -> None:
        """Create a table as select.

        Most convenient way to create a table from a df, but does neither support
        constraints nor altering the table afterwards.
        (That is why we need to create the table schema manually and insert the data
        afterwards for more complex tables, see DatabaseSchema.)
        """
        # DuckDB does not support hyphens in table names
        table_name = table_name.replace("-", "_")
        self.execute(
            f"CREATE OR REPLACE TABLE {table_name} AS FROM df"
        )  # df is taken from context by DuckDB

    def insert_trials(
        self,
        trials_df: pl.DataFrame,
    ) -> None:
        columns = ", ".join(trials_df.columns)
        try:
            self.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM trials_df")
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
        self.execute(f"""
            INSERT INTO {table_name}
            SELECT t.trial_id, r.*
            FROM raw_data_df AS r
            JOIN Trials AS t 
                ON r.trial_number = t.trial_number 
                AND r.participant_id = t.participant_id
            ORDER BY r.rownumber;
        """)

    # Note that in constrast to raw data, preprocessed and feature-engineered data is
    # not inserted into the database per participant, but per modality over all
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
        self.execute(f"""
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
        # TODO: FIXME same as preprocess data for now
        self.insert_preprocess_data(table_name, feature_data_df)

    def insert_labels_data(
        self,
        table_name: str,
        label_data_df: pl.DataFrame,
    ) -> None:
        # TODO: FIXME same as preprocess data for now
        self.insert_preprocess_data(table_name, label_data_df)


def main():
    # MODALITIES = ["Face"]
    with DatabaseManager() as db:
        # Participant list
        df = create_participants_df()
        db.ctas("Participants", df)

        # Experiment results (calibration and performance metrics)
        df = create_calibration_results_df()
        db.ctas("Calibration_Results", df)
        df = create_measurement_results_df()
        db.ctas("Measurement_Results", df)

        # Questionnaire data
        for questionnaire in QUESTIONNAIRES:
            df = create_questionnaire_df(questionnaire)
            db.ctas("Questionnaire_" + questionnaire.upper(), df)
        logger.info("Questionnaire data inserted.")

        # Raw data
        for participant_id in range(1, NUM_PARTICIPANTS + 1):
            if participant_id in DataConfig.MISSING_PARTICIPANTS:
                logger.debug(f"No data for participant {participant_id}.")
                continue
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

        # Preprocessed data
        # no check for existing data as it will be overwritten
        for modality in MODALITIES:
            table_name = "Preprocess_" + modality
            df = db.get_table("Raw_" + modality, exclude_invalid_data=False)
            df = create_preprocess_data_df(table_name, df)
            db.insert_preprocess_data(table_name, df)
        logger.info("Data preprocessed.")

        # Feature-engineered data
        for modality in MODALITIES:
            table_name = f"Feature_{modality}"
            df = db.get_table(f"Preprocess_{modality}", exclude_invalid_data=False)
            df = create_feature_data_df(table_name, df)
            db.insert_feature_data(table_name, df)
        logger.info("Data feature-engineered.")

        # TODO: Add labels
        #
        # add final merge of all feature data and label at the *very* end
        # using a separate function for this
        # something like this for collecting all feature data:
        # query = """
        # FROM sqlite_schema
        # select tbl_name
        # """
        # with db:
        #     tables = db.execute(query).pl()
        # t = tables.get_column("tbl_name").to_list()
        # list(filter(lambda x: x.startswith("Feature"), t))
        # -> the df function for label adding should be in data_processing.py

        logger.info("Data pipeline completed.")


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")
