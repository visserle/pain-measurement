import logging

import duckdb
import polars as pl
from polars import col

from src.data.anonymize import anonymize_db
from src.data.data_config import DataConfig
from src.data.data_processing import (
    create_calibration_results_df,
    create_feature_data_df,
    create_measurement_results_df,
    create_merged_and_labeled_data_df,
    create_participants_df,
    create_preprocess_data_df,
    create_questionnaire_df,
    create_raw_data_df,
    create_trials_df,
)
from src.data.database_schema import DatabaseSchema
from src.data.imotions_data import load_imotions_data_df

DB_FILE = DataConfig.DB_FILE
NUM_PARTICIPANTS = DataConfig.NUM_PARTICIPANTS
MODALITIES = DataConfig.MODALITIES
QUESTIONNAIRES = DataConfig.QUESTIONNAIRES


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
        df = db.get_table("Trials", exclude_trials_with_measurement_problems=False)
    df.head()
    ```
    """

    def __init__(self) -> None:
        self.conn = None
        self._initialize_tables()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

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

    def _ensure_connection(self) -> None:
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
        exclude_trials_with_measurement_problems: bool | str | list[str] = True,
    ) -> pl.DataFrame:
        """Return the data from a table as a Polars DataFrame.

        Args:
            table_name: The name of the table to retrieve.
            exclude_trials_with_measurement_problems:
                - If True, excludes all trials with measurement problems.
                - If a string, excludes only trials with problems in the specified modality (e.g., 'eeg').
                - If a list of strings, excludes trials with problems in any of the specified modalities.
                Modality matching is partial, so 'eeg' will match entries like 'eeg/eda'.
                - If False, includes all trials.
        """

        df = self.execute(
            f"SELECT * FROM {table_name}"
        ).pl()  # could be more efficient by filtering out invalid trials in the query

        invalid_trials = self.execute("SELECT * FROM Invalid_Trials").pl()
        # do not filter invalid trials from invalid_trials table, would be empty
        if table_name == "invalid_trials":
            return df

        if exclude_trials_with_measurement_problems:
            # Filter invalid trials by modality if a specific modality or list is specified
            filtered_invalid_trials = invalid_trials

            if isinstance(exclude_trials_with_measurement_problems, str):
                # Single modality case
                modality = exclude_trials_with_measurement_problems
                filtered_invalid_trials = invalid_trials.filter(
                    pl.col("modality").str.contains(modality)
                )
            elif isinstance(exclude_trials_with_measurement_problems, list):
                # Multiple modalities case - create a filter condition for each modality
                filter_conditions = [
                    pl.col("modality").str.contains(mod)
                    for mod in exclude_trials_with_measurement_problems
                ]
                # Combine conditions with logical OR
                if filter_conditions:
                    combined_filter = filter_conditions[0]
                    for condition in filter_conditions[1:]:
                        combined_filter = combined_filter | condition
                    filtered_invalid_trials = invalid_trials.filter(combined_filter)

            if "participant_id" in df.columns and "trial_number" in df.columns:
                # Note that not every participant has 12 trials, so a filter using the
                # trial_id would remove the wrong trials
                df = df.filter(
                    ~pl.struct(["participant_id", "trial_number"]).is_in(
                        filtered_invalid_trials.select(
                            ["participant_id", "trial_number"]
                        )
                        .unique()
                        .to_struct()
                    )
                )
            elif (
                "participants" in table_name.lower()
                or "questionnaire" in table_name.lower()
                or "result" in table_name.lower()
            ):
                # remove participants that only have invalid trials
                # (note that this is different from the invalid participants table)

                # If filtering by modality, only consider trials with that modality
                only_invalid_trials = (
                    filtered_invalid_trials.group_by("participant_id")
                    .agg(pl.len().alias("count"))
                    .filter(pl.col("count") == 12)
                    .get_column("participant_id")
                )
                df = df.filter(~pl.col("participant_id").is_in(only_invalid_trials))
            else:  # not all tables have trial information, e.g. questionnaires
                # no filtering necessary, invalid participants are already excluded
                pass
        return df

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

        # Register the DataFrame explicitly first
        self.conn.register("temp_df", df)
        self.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM temp_df")
        self.conn.unregister("temp_df")

    def insert_trials(
        self,
        trials_df: pl.DataFrame,
    ) -> None:
        columns = ", ".join(trials_df.columns)
        try:
            self.conn.register("trials_df", trials_df)
            self.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM trials_df")
            self.conn.unregister("trials_df")
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
        self.conn.register("raw_data_df", raw_data_df)
        self.execute(f"""
            INSERT INTO {table_name}
            SELECT t.trial_id, r.*
            FROM raw_data_df AS r
            JOIN Trials AS t 
                ON r.trial_number = t.trial_number 
                AND r.participant_id = t.participant_id
            ORDER BY r.rownumber;
        """)
        self.conn.unregister("raw_data_df")

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
        self.conn.register("preprocess_data_df", preprocess_data_df)
        self.execute(f"""
            INSERT INTO {table_name}
            SELECT *
            FROM preprocess_data_df
            ORDER BY trial_id, timestamp;
        """)
        self.conn.unregister("preprocess_data_df")

    def insert_feature_data(
        self,
        table_name: str,
        feature_data_df: pl.DataFrame,
    ) -> None:
        DatabaseSchema.create_feature_data_table(
            self.conn,
            table_name,
            feature_data_df.schema,
        )
        self.conn.register("feature_data_df", feature_data_df)
        self.execute(f"""
            INSERT INTO {table_name}
            SELECT *
            FROM feature_data_df
            ORDER BY trial_id, timestamp;
        """)
        self.conn.unregister("feature_data_df")


def main():
    """
    Main function for database creation.
    The first step of this pipeline needs the original deanonymized data.
    Later steps of this pipeline can be applied using the publicy availible anonymized
    data.
    """
    db = DatabaseManager()
    if not DB_FILE.exists():
        with db:
            # Participant data, experiment and questionnaire results
            db.ctas(
                "Invalid_Participants", DataConfig.load_invalid_participants_config()
            )
            db.ctas("Invalid_Trials", DataConfig.load_invalid_trials_config())
            db.ctas("Participants", create_participants_df())
            db.ctas("Calibration_Results", create_calibration_results_df())
            db.ctas("Measurement_Results", create_measurement_results_df())
            for questionnaire in QUESTIONNAIRES:
                df = create_questionnaire_df(questionnaire)
                db.ctas("Questionnaire_" + questionnaire.upper(), df)
            logger.info("Participant data inserted.")

            # Raw data
            for participant_id in range(1, NUM_PARTICIPANTS + 1):
                if participant_id in (
                    db.execute("SELECT participant_id FROM Invalid_Participants")
                    .pl()  # returns a df
                    .to_series()
                    .to_list()
                ):
                    logger.debug(f"Participant {participant_id} is invalid.")
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

    # Anonymize database
    anonymize_db(db)
    logger.info("Anonymized Database.")

    # NOTE: From here on, you can use the pipeline without possessing the original, de-
    # anonymized data.

    with db:
        # Preprocessed data
        # no check for existing data as it will be overwritten every time
        for modality in MODALITIES:
            table_name = "Preprocess_" + modality
            df = db.get_table(
                "Raw_" + modality,
                exclude_trials_with_measurement_problems=False,
            )
            df = create_preprocess_data_df(table_name, df)
            db.insert_preprocess_data(table_name, df)
        logger.info("Data preprocessed.")

        # Feature-engineered data
        for modality in MODALITIES:
            table_name = f"Feature_{modality}"
            df = db.get_table(
                f"Preprocess_{modality}",
                exclude_trials_with_measurement_problems=False,
            )
            df = create_feature_data_df(table_name, df)
            db.insert_feature_data(table_name, df)
        logger.info("Data feature-engineered.")

        # Merge feature data and add labels
        data_dfs = []
        for modality in MODALITIES:
            if modality == "EEG":
                continue  # we do not merge EEG data, as it has a different sampling rate
            data_dfs.append(
                db.get_table(
                    f"Feature_{modality}",
                    exclude_trials_with_measurement_problems=False,
                )
            )
        trials_df = db.get_table(
            "Trials", exclude_trials_with_measurement_problems=False
        )
        df = create_merged_and_labeled_data_df(data_dfs, trials_df)
        db.ctas("Merged_and_Labeled_Data", df)
        logger.info("Data merged and labeled.")

        logger.info("Data pipeline completed.")


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")
