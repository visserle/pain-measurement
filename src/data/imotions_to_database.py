import logging
from pathlib import Path

import duckdb
import polars as pl

from src.data.imotions_data_config import data_config
from src.data.utils import pl_schema_to_duckdb_schema

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

# Paths
db_file = Path("test.duckdb")
participant_data_file = Path("runs/experiments/participants.csv")
data = Path("data/imotions")


def load_imotions_data(
    data_config: dict[str, dict[str, str]],
    data_path: Path,
    participant_id: int,
) -> dict[str, dict[str, str]]:
    """Add dataframes to the data_config dictionary (also lowers column names)."""
    for key, config in data_config.items():
        file_name = data_path / str(participant_id) / f"{config['file_name']}.csv"
        if not file_name.is_file():
            logger.error(f"File {file_name} does not exist.")
            continue
        start_index = _get_start_index_from_imotions_file(file_name)
        df = pl.read_csv(
            file_name,
            skip_rows=start_index,
            infer_schema_length=20000,  # FIXME TODO
            columns=config["load_columns"],
        )
        logger.debug(f"Loaded {key} data from {file_name}.")
        if "rename_columns" in config:
            df = df.rename(config["rename_columns"])
        # Lowercase column names
        df = df.select([pl.col(col).alias(col.lower()) for col in df.columns])
        # Remove spaces from column names
        df = df.select([pl.col(col).alias(col.replace(" ", "_")) for col in df.columns])
        data_config[key]["data"] = df
    return data_config


def _get_start_index_from_imotions_file(file_name: str) -> int:
    """Output files from iMotions have a header that needs to be skipped."""
    with open(file_name, "r") as file:
        lines = file.readlines(2**16)  # only read a few lines
    file_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
    return file_start_index


def create_trials_df(
    imotions_data_config: dict[str, dict[str, pl.DataFrame]],
    participant_id: str,
) -> pl.DataFrame:
    """
    Create a table with trial information (metadata for each measurement).
    """
    trials = imotions_data_config["iMotions_Marker"]["data"]
    trials = (
        # 'markerdescription' from imotions was used exclusively for the stimulus seed
        # drop all rows where the markerdescription is null
        trials.filter(pl.col("markerdescription").is_not_null())
        .select(
            [
                pl.col("markerdescription").alias("stimulus_seed").cast(pl.UInt16),
                pl.col("timestamp").alias("timestamp_start"),
            ]
        )
        # group by stimulus_seed to ensure one row per stimulus
        .group_by("stimulus_seed")
        # create columns for start and end of each stimulus
        .agg(
            [
                pl.col("timestamp_start").min().alias("timestamp_start"),
                pl.col("timestamp_start").max().alias("timestamp_end"),
            ]
        )
        .sort("timestamp_start")
    )
    # add column for duration of each stimulus
    trials = trials.with_columns(
        trials.select(
            (pl.col("timestamp_end") - pl.col("timestamp_start")).alias("duration")
        )
    )
    # add column for trial number
    trials = trials.with_columns(
        pl.arange(1, trials.height + 1).alias("trial_number").cast(pl.UInt8)
    )
    # add column for participant id
    trials = trials.with_columns(
        pl.lit(participant_id).alias("participant_id").cast(pl.UInt8)
    )
    # add column for skin area
    id_is_odd = int(participant_id) % 2
    skin_areas = list(range(1, 7)) if id_is_odd else list(range(6, 0, -1))
    trials = trials.with_columns(
        pl.Series(
            name="skin_area",
            values=skin_areas * (trials.height // len(skin_areas) + 1),  # repeat
        )
        .head(trials.height)
        .alias("skin_area")
        .cast(pl.UInt8)
    )
    # change order of columns
    trials = trials.select(
        pl.col(a := "trial_number"),
        pl.col(b := "participant_id"),
        pl.all().exclude(a, b),
    )
    return trials


def load_trials_into_db(
    df: pl.DataFrame,
    db_file: Path = db_file,
) -> None:
    con = duckdb.connect(str(db_file))

    # Check if the table already exists
    table_exists = (
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'Trials'"
        ).fetchone()[0]
        > 0
    )

    if not table_exists:
        # Create sequence and table
        query = f"""
            CREATE SEQUENCE IF NOT EXISTS seq_trials_trial_id START 1;
            CREATE OR REPLACE TABLE Trials (
                trial_id USMALLINT NOT NULL DEFAULT NEXTVAL('seq_trials_trial_id'),
                {pl_schema_to_duckdb_schema(df.schema)},
                -- PRIMARY KEY (trial_id),
                UNIQUE (trial_number, participant_id)
            )
        """
        con.execute(query)

    # Insert data using polars
    columns = ", ".join(df.columns)
    try:
        con.execute(f"INSERT INTO Trials ({columns}) SELECT * FROM df")
    except duckdb.ConstraintException as e:
        logger.warning(f"Error inserting data into Trials: {e}")
    con.close()


def create_raw_data_dfs(
    imotions_data_config: dict[str, dict[str, pl.DataFrame]],
    trials: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """
    Create raw data tables for each data type based on the trial information.
    (Only keep rows that are part of a trial.)
    """
    # Create a DataFrame with the trial information only
    trial_info = trials.select(
        pl.col("timestamp_start").alias("trial_start"),
        pl.col("timestamp_end").alias("trial_end"),
        pl.col("trial_number"),
        pl.col("participant_id"),
    )
    raw_data_dfs = {}
    for key, config in imotions_data_config.items():
        if key == "iMotions_Marker":  # skip trial data (metadata)
            continue

        # Perform an asof join to assign trial numbers to each stimulus
        df = config["data"]
        df = df.join_asof(
            trial_info,
            left_on="timestamp",
            right_on="trial_start",
            strategy="backward",
        )
        # Correct backwards join by checking for the end of each trial
        df = (
            df.with_columns(
                pl.when(
                    pl.col("timestamp").is_between(
                        pl.col("trial_start"),
                        pl.col("trial_end"),
                    )
                )
                .then(pl.col("trial_number"))
                .otherwise(None)
                .alias("trial_number")
            )
            .filter(pl.col("trial_number").is_not_null())  # drop non-trial rows
            .drop(["trial_start", "trial_end"])
        )
        # Change order of columns
        df = df.select(
            pl.col(a := "trial_number"),
            pl.col(b := "participant_id"),
            pl.all().exclude(a, b),
        )

        key = key.replace("iMotions_", "Raw_")
        raw_data_dfs[key] = df

    return raw_data_dfs


def load_raw_data_into_db(
    name: str,
    raw_data_df: pl.DataFrame,
    db_file: Path = db_file,
) -> None:
    con = duckdb.connect(str(db_file))

    con.execute("""
        CREATE TEMPORARY TABLE raw_data AS
        SELECT * FROM raw_data_df
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {name} (
        trial_id USMALLINT,
        {pl_schema_to_duckdb_schema(raw_data_df.schema)},
        -- FOREIGN KEY (trial_id) REFERENCES trials(trial_id)
    );
    """)

    # Join the raw data dataframe with the trials table to get the trial_id
    query = f"""
        INSERT INTO {name}
        SELECT t.trial_id, r.*
        FROM raw_data AS r
        JOIN trials AS t ON r.trial_number = t.trial_number AND r.participant_id = t.participant_id;
    """
    con.execute(query)

    # Drop the temporary table
    con.execute("""
        DROP TABLE raw_data
    """)

    con.close()


def main():
    for participant_id in range(1, 4):
        # Load polars dataframes into dictionary from config
        load_imotions_data(data_config, data, participant_id=participant_id)

        # Create metadata table
        trials = create_trials_df(data_config, participant_id=participant_id)
        load_trials_into_db(trials)

        # Create raw data tables
        raw_data_dfs = create_raw_data_dfs(data_config, trials)
        for key, df in raw_data_dfs.items():
            load_raw_data_into_db(key, df)


if __name__ == "__main__":
    import time

    from src.log_config import configure_logging

    configure_logging(stream_level=logging.DEBUG)

    start = time.time()
    main()
    end = time.time()
    print(f"Runtime: {end - start:.2f} seconds.")


# Insert participants data into database
# participants_table = pl.read_csv(participant_data_file)
# load_table_into_db(participants_table, "Participants")
#
# def load_table_into_db(
#     table: pl.DataFrame,
#     table_name: str,
#     db_file: Path = db_file,
# ) -> None:
#     """Load a polars DataFrame into a duckdb database."""
#     con = duckdb.connect(str(db_file))
#     temp_table_name = f"temp_{table_name}"
#     con.register(temp_table_name, table)

#     # Check if the table already exists
#     table_exists = (
#         con.execute(
#             f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
#         ).fetchone()[0]
#         > 0
#     )

#     # Create DuckDB table from DataFrame
#     # NOTE: workwaround because in duckdb for ctas constraits are not supported
#     # nor alter table add primary key

#     # con.execute(
#     #     "CREATE TABLE table_name AS SELECT * FROM temp_table_name, PRIMARY KEY (Participant, trial_number)"
#     # )
#     # # Insert data from DataFrame to DuckDB table
#     # con.execute("INSERT INTO my_table SELECT * FROM df_pd")

#     if table_exists:
#         # If the table exists, append the new data to it
#         con.execute(f"INSERT INTO {table_name} SELECT * FROM {temp_table_name}")
#         logger.debug(f"Appended data to table '{table_name}' in the database.")
#     else:
#         # If the table doesn't exist, create a new table with primary key constraint
#         # Create table from the selection
#         con.execute(
#             f"CREATE TABLE {table_name} (Participant INTEGER PRIMARY KEY) AS SELECT * FROM {temp_table_name}"
#         )

#         # Add primary key
#         con.execute(
#             f"ALTER TABLE {table_name} ADD PRIMARY KEY (Participant, trial_number)"
#         )

#         con.execute(
#             f"CREATE TABLE {table_name} AS SELECT * FROM {temp_table_name}, PRIMARY KEY (Participant, trial_number)"
#         )

#         logger.debug(
#             f"Table '{table_name}' created in the database with primary key (Participant, trial_number)."
#         )

#     con.close()
