from pathlib import Path

import yaml


class DataConfig:
    MODALITIES = ["Stimulus", "EDA", "EEG", "PPG", "Pupil", "Face"]
    METADATA = ["Trials"]
    NUM_PARTICIPANTS = 28

    DB_FILE = Path("experiment.duckdb")
    PARTICIPANT_DATA_FILE = Path("runs/experiments/participants.csv")

    IMOTIONS_DATA_PATH = Path("data/imotions")
    IMOTIONS_DATA_CONFIG_FILE = Path("src/data/imotions_data_config.yaml")

    @classmethod
    def load_imotions_config(cls):
        with open(cls.IMOTIONS_DATA_CONFIG_FILE, "r") as file:
            return yaml.safe_load(file)
