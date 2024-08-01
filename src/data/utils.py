from functools import reduce

import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.make_dataset import load_dataset


def pl_schema_to_duckdb_schema(schema: pl.Schema) -> str:
    """
    Convert a Polars DataFrame schema to a DuckDB schema string. Also lowercases all
    column names.

    In an ideal world, this function would not be necessary, but DuckDB neither supports
    constraints in CREATE TABLE AS SELECT (CTAS) statements, nor does it support
    altering tables to add keys or constraints. Therefore, we need to create the table
    schema manually and insert the data afterwards for keys / constraints.

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
