from pathlib import Path

import polars as pl
import tomllib

from src.experiments.measurement.stimulus_generator import StimulusGenerator

config_path = Path("src/experiments/measurement/measurement_config.toml")
with open(config_path, "rb") as f:
    config = tomllib.load(f)["stimulus"]

seeds = config["seeds"]
seed_data = pl.DataFrame(
    {
        "seed": seeds,
        "major_decreasing_intervals": [
            StimulusGenerator(config=config, seed=seed).major_decreasing_intervals_ms
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
    from src.data.utils import pl_schema_to_duckdb_schema

    duckdb_schema = pl_schema_to_duckdb_schema(seed_data.schema)
    print(seed_data)
    print(f"{duckdb_schema = }")
