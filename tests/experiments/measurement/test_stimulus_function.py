from pathlib import Path

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.experiments.utils import load_configuration

config = load_configuration(Path("src/experiments/measurement/measurement_config.toml"))
config = config["stimulus"]
stimulus = StimulusGenerator(config)


def test_stimulus_duration():
    assert stimulus.duration == 180


def test_uniqueness_of_seeds():
    assert len(config["seeds"]) == len(set(config["seeds"]))


def test_number_of_seeds():
    assert len(config["seeds"]) == 12
