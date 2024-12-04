import logging

import duckdb
import polars as pl

from src.data.data_processing import create_seeds_df

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class DatabaseSchema:
    """
    Database schema for the experiment data.

    Consists of:
        - metadata tables for participants, trials, seeds, etc. and
        - data tables for raw, preprocess, and feature data.

    Note: participant_id and trial_number are denormalized columns to avoid joins.
    """

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
                trial_number UTINYINT,
                participant_id UTINYINT,
                stimulus_seed USMALLINT,
                skin_area UTINYINT,
                timestamp_start DOUBLE,
                timestamp_end DOUBLE,
                duration DOUBLE,
                UNIQUE (trial_number, participant_id)
            );
        """)

    @staticmethod
    def create_seeds_table(
        conn: duckdb.DuckDBPyConnection,
    ) -> None:
        # check for existing table using custom function to avoid df creation
        if not DatabaseSchema.table_exists(conn, "Seeds"):
            # NOTE: we directly load the seed data into the table here, which is an
            # exception to the usual pattern (create table, insert data)
            # this makes sense because the seeds are static and not part of the
            # experiment data
            seeds_data = create_seeds_df()  # noqa : used in the string below
            conn.execute("CREATE TABLE Seeds AS SELECT * FROM seeds_data")

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
        """)
        logger.debug(f"Created table '{name}'.") if not DatabaseSchema.table_exists(
            conn, name
        ) else None

    @staticmethod
    def create_preprocess_data_table(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        schema: pl.Schema,
    ) -> None:
        conn.execute(f"""
            CREATE OR REPLACE TABLE {name} (
                {map_polars_schema_to_duckdb(schema)},
                UNIQUE (trial_id, rownumber)
            );
        """)
        logger.debug(f"Created table '{name}'.")

    @staticmethod
    def create_feature_data_table(
        conn: duckdb.DuckDBPyConnection,
        name: str,
        schema: pl.Schema,
    ) -> None:
        # same schema as preprocess data
        return DatabaseSchema.create_preprocess_data_table(conn, name, schema)

    @staticmethod
    def table_exists(
        conn: duckdb.DuckDBPyConnection,
        name: str,
    ) -> bool:
        return (
            conn.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables
                WHERE table_name = '{name}'
                """).fetchone()[0]
            > 0
        )


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
        pl.Datetime: "TIMESTAMP",
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
        # Base types and unsupported types
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
