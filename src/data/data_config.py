from pathlib import Path

import polars as pl
import tomllib
import yaml


class DataConfig:
    MODALITIES = ["Stimulus", "EDA", "EEG", "PPG", "Pupil", "Face"]
    NUM_PARTICIPANTS = 28

    DB_FILE = Path("data/experiment.duckdb")
    PARTICIPANT_DATA_FILE = Path("runs/experiments/participants.csv")
    INVALID_TRIALS_FILE = Path("src/data/invalid_trials.csv")

    IMOTIONS_DATA_PATH = Path("data/imotions")
    IMOTIONS_DATA_CONFIG_FILE = Path("src/data/imotions_data_config.yaml")

    STIMULUS_CONFIG_PATH = Path("src/experiments/measurement/measurement_config.toml")

    @classmethod
    def load_imotions_config(cls):
        with open(cls.IMOTIONS_DATA_CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)

    @classmethod
    def load_stimulus_config(cls):
        with open(cls.STIMULUS_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)["stimulus"]

    @classmethod
    def load_invalid_trials(cls):
        return pl.read_csv(cls.INVALID_TRIALS_FILE, skip_rows=4)
