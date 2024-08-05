from pathlib import Path

import polars as pl
import tomllib

from src.experiments.measurement.stimulus_generator import StimulusGenerator

STIMULUS_CONFIG_PATH = Path("src/experiments/measurement/measurement_config.toml")


def get_seeds_data():
    with open(STIMULUS_CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)["stimulus"]

    seeds = config["seeds"]
    return pl.DataFrame(
        {
            "seed": seeds,
            "major_decreasing_intervals": [
                StimulusGenerator(
                    config=config, seed=seed
                ).major_decreasing_intervals_ms
                for seed in seeds
            ],
        },
        schema={
            "seed": pl.UInt16,
            "major_decreasing_intervals": pl.List(pl.List(pl.UInt32)),
        },
    )


if __name__ == "__main__":
    # debugging
    from src.data.database_schema import map_polars_schema_to_duckdb

    seed_data = get_seeds_data()
    duckdb_schema = map_polars_schema_to_duckdb(seed_data.schema)
    print(seed_data)
    print(f"{duckdb_schema = }")
