from pathlib import Path

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.experiments.utils import load_configuration

config = load_configuration(Path("src/experiments/measurement/measurement_config.toml"))
stimulus = StimulusGenerator(config["stimulus"])


def test_stimulus_duration():
    assert stimulus.duration == 180
