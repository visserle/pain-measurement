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


# # TODO: check all the seeds
# from src.experiments.thermoino import ThermoinoComplexTimeCourses, list_com_ports

# thermoino = ThermoinoComplexTimeCourses(
#     mms_baseline=28,  # has to be the same as in MMS
#     mms_rate_of_rise=1,  # has to be the same as in MMS
#     dummy=True,
# )
# thermoino.connect()
# thermoino.init_ctc(bin_size_ms=500)

# for seed in config["seeds"]:
#     stimulus = StimulusGenerator(config, seed)
#     thermoino.create_ctc(temp_course=stimulus.y, sample_rate=stimulus.sample_rate)
# # raises an error if rate of rise is too high for the given stimulus
