from functools import reduce

import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_participant import ParticipantConfig
from src.data.make_dataset import load_dataset


def pl_schema_to_duckdb_schema(schema: pl.Schema) -> str:
    """
    Convert a Polars DataFrame schema to a DuckDB schema string. Also lowercases all
    column names.

    In an ideal world, this function would not be necessary, but DuckDB neihter does
    support constraints in CREATE TABLE AS SELECT (CTAS) statements, nor does it support
    and does not support
    altering tables to add primary keys. Therefore, we need to create the table schema
    manually and insert the data afterwards.

    NOTE: Not tested for all Polars data types (especially nested types).
    NOTE: pl.Object is not accounted for here.
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


def merge_datasets(
    dfs: list[pl.DataFrame],
    merge_on: list[str] = ["Timestamp", "Trial", "Participant"],
    sort_by: list[str] = ["Timestamp"],
) -> pl.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame.

    The default merge_on and sort_by columns are for merging different modalities of
    one participant.

    The function can also be used to merge multiple participants' modalities with
    a different merge_on and sort_by column.

    Examples:

    Merge two datasets of different modalities of one participant:
    >>> dfs = load_participant_datasets(PARTICIPANT_LIST[0], INTERIM_LIST)
    >>> eda_plus_rating = merge_datasets([dfs.eda, dfs.stimulus])


    Merge multiple participants' modalities:
    ````python
    # The load function loads one modality for multiple participants
    stimuli = load_modality_data(PARTICIPANT_LIST, INTERIM_DICT["stimulus"])
    eda = load_modality_data(PARTICIPANT_LIST, INTERIM_DICT["eda"])
    multiple_eda_plus_rating = merge_datasets(
        [stimuli, eda],
        merge_on=["Timestamp", "Participant", "Trial"],
        sort_by=["Participant", "Trial", "Timestamp"],
    )
    # Normalzing, plotting, etc.
    features = ["Temperature", "Rating", "EDA_Tonic"]
    multiple_eda_plus_rating = interpolate(multiple_eda_plus_rating)
    multiple_eda_plus_rating = scale_min_max(
        multiple_eda_plus_rating, exclude_additional_columns=["Temperature", "Rating"]
    )
    multiple_eda_plus_rating.hvplot(
        x="Timestamp",
        y=features,
        groupby=["Participant", "Trial"],
        kind="line",
        width=800,
        height=400,
        ylim=(0, 1),
    )
    ````
    """
    if len(dfs) < 2:
        return dfs[0]

    df = reduce(
        lambda left, right: left.join(
            right,
            on=merge_on,
            how="outer_coalesce",
        ).sort(sort_by),
        dfs,
    )
    return df


def load_modality_data(
    participants: list[ParticipantConfig],
    modality: DataConfigBase,
) -> pl.DataFrame:
    """
    Load data from multiple participants for a specific modality. Used for exploratory
    data analysis.
    """

    dfs = []
    for participant in participants:
        df = load_dataset(participant, modality).df
        dfs.append(df)

    return pl.concat(dfs)
