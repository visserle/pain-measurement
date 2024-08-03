import logging

import duckdb
import polars as pl

from src.data.seeds_data import seed_data  # noqa (used in db query)

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
                {map_polars_schema_to_duckdb(schema)},
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
            {map_polars_schema_to_duckdb(schema)},
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
        return DatabaseSchema.create_preprocessed_data_table(conn, name, schema)


def map_polars_schema_to_duckdb(schema: pl.Schema) -> str:
    """
    Convert a Polars DataFrame schema to a DuckDB schema string. Also lowercases all
    column names.

    In an ideal world, this function would not be necessary, but DuckDB neither supports
    constraints in CREATE TABLE AS SELECT (CTAS) statements, nor does it support
    altering tables to add keys or constraints. Therefore, we need to create the table
    schema manually and insert the data afterwards for keys and constraints.

    NOTE: Not validated for all Polars data types, pl.Object is missing.
    """
    type_mapping = {
        pl.Int8: "TINYINT",
        pl.Int16: "SMALLINT",
        pl.Int32: "INTEGER",
        pl.Int64: "BIGINT",
        pl.UInt8: "UTINYINT",
        pl.UInt16: "USMALLINT",
        pl.UInt32: "UINTEGER",
        pl.UInt64: "UBIGINT",
        pl.Float32: "REAL",
        pl.Float64: "DOUBLE",
        pl.Boolean: "BOOLEAN",
        pl.Utf8: "VARCHAR",
        pl.Date: "DATE",
        pl.Datetime: "tIMESTAMP",
        pl.Duration: "INTERVAL",
        pl.Time: "TIME",
        pl.Categorical: "VARCHAR",
        pl.Binary: "BLOB",
    }

    def get_duckdb_type(polars_type):
        # Recursive function to handle nested types
        if isinstance(polars_type, pl.Decimal):
            precision = polars_type.precision
            scale = polars_type.scale
            return f"DECIMAL({precision}, {scale})"
        elif isinstance(polars_type, pl.List):
            inner_type = get_duckdb_type(polars_type.inner)
            return f"{inner_type}[]"
        elif isinstance(polars_type, pl.Struct):
            fields = [
                f"{field[0]} {get_duckdb_type(field[1])}"
                for field in polars_type.fields
            ]
            return f"STRUCT({', '.join(fields)})"
        # Base types
        else:
            duckdb_type = type_mapping.get(polars_type)
            if duckdb_type is None:
                raise ValueError(f"Unsupported Polars data type: {polars_type}")
            return duckdb_type

    duckdb_schema = []
    for column_name, polars_type in schema.items():
        duckdb_type = get_duckdb_type(polars_type)
        duckdb_schema.append(f"{column_name.lower()} {duckdb_type}")

    return ", ".join(duckdb_schema)
