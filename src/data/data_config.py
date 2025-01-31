from pathlib import Path

import polars as pl
import tomllib
import yaml


class DataConfig:
    DB_FILE = Path("data/pain-measurement.duckdb")

    MODALITIES = ["Stimulus", "EDA", "EEG", "PPG", "Pupil", "Face"]

    NUM_PARTICIPANTS = 50
    PARTICIPANT_DATA_FILE = Path("runs/experiments/participants.csv")
    INVALID_PARTICIPANTS_FILE = Path("src/data/invalid_participants.csv")
    INVALID_TRIALS_FILE = Path("src/data/invalid_trials.csv")

    CALIBRATION_RESULTS_FILE = Path("data/experiments/calibration_results.csv")
    MAESUREMENT_RESULTS_FILE = Path("data/experiments/measurement_results.csv")

    QUESTIONNAIRES = [
        # same as in src/experiments/questionnaires/app.py
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

    @classmethod
    def load_imotions_config(cls):
        with open(cls.IMOTIONS_DATA_CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)

    @classmethod
    def load_stimulus_config(cls):
        with open(cls.STIMULUS_CONFIG_PATH, "rb") as f:
            return tomllib.load(f)["stimulus"]

    @classmethod
    def load_invalid_participants_config(cls):
        return pl.read_csv(
            cls.INVALID_PARTICIPANTS_FILE,
            schema_overrides=dict(participant_id=pl.UInt8),
        )

    @classmethod
    def load_invalid_trials_config(cls):
        return pl.read_csv(
            cls.INVALID_TRIALS_FILE,
            schema_overrides=dict(participant_id=pl.UInt8, trial_number=pl.UInt8),
        )
