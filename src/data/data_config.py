from pathlib import Path

import polars as pl
import tomllib
import yaml


class DataConfig:
    DB_FILE = Path("data/pain-measurement.duckdb")

    MODALITIES = ["Stimulus", "EDA", "EEG", "PPG", "Pupil", "Face"]
    NUM_PARTICIPANTS = 45
    MISSING_PARTICIPANTS = [39]

    PARTICIPANT_DATA_FILE = Path("runs/experiments/participants.csv")
    INVALID_TRIALS_FILE = Path("src/data/invalid_trials.csv")

    CALIBRATION_RESULTS_FILE = Path("data/experiments/calibration_results.csv")
    MAESUREMENT_RESULTS_FILE = Path("data/experiments/measurement_results.csv")

    QUESTIONNAIRES = [
        # same as in questionnaires/app.py
        "general",
        "bdi-ii",
        "phq-15",
        "panas",
        "pcs",
        "pvaq",
        "stai-t-10",
        "maas",
    ]
    QUESTIONNAIRES_DATA_PATH = Path("data/experiments/questionnaires")

    IMOTIONS_DATA_PATH = Path("data/imotions")
    IMOTIONS_DATA_CONFIG_FILE = Path("src/data/imotions_data_config.yaml")
    STIMULUS_CONFIG_PATH = Path("src/experiments/measurement/measurement_config.toml")

    # Class methods to load config files
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
        return pl.read_csv(cls.INVALID_TRIALS_FILE, skip_rows=6)
