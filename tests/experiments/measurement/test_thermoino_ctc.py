from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tomllib

from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.experiments.thermoino import ThermoinoComplexTimeCourses


def load_configuration(file_path: str) -> dict:
    """Load configuration from a TOML file."""
    with open(file_path, "rb") as file:
        return tomllib.load(file)


current_dir = Path(__file__).resolve().parent
thermoino_config = load_configuration(
    (current_dir / "../../../src/experiments/thermoino_config.toml").resolve()
)
stimulus_config = load_configuration(
    (
        current_dir / "../../../src/experiments/measurement/measurement_config.toml"
    ).resolve()
)["stimulus"]


@dataclass
class DummyStimulusGenerator:
    sample_rate: int
    duration: int

    def __post_init__(self):
        self.y = (
            -np.cos(np.linspace(0, 2 * np.pi, (self.duration * self.sample_rate))) * 4
            + 40
        )


@pytest.fixture
def thermoino():
    # Initialize Thermoino with specific settings for testing complex time courses
    test_thermoino = ThermoinoComplexTimeCourses(
        mms_baseline=thermoino_config["mms_baseline"],
        mms_rate_of_rise=thermoino_config["mms_rate_of_rise"],
        dummy=True,
    )
    test_thermoino.connect()
    yield test_thermoino
    test_thermoino.close()


def test_create_ctc_inverse_transform(thermoino):
    # Define a simple sinusoidal temperature course and a sample rate
    sample_rate = stimulus_config["sample_rate"]
    bin_size_ms = thermoino_config["bin_size_ms"]

    # Uncomment for DummyStimulusGenerator
    # stimulus = DummyStimulusGenerator(
    #     sample_rate=sample_rate,
    #     duration=30,
    # )
    stimulus = StimulusGenerator()

    temp_course = stimulus.y

    # Initialize and create the CTC
    thermoino.init_ctc(bin_size_ms=bin_size_ms)
    thermoino.create_ctc(temp_course=temp_course, sample_rate=sample_rate)

    # Calculate the ms per bin rate of rise
    mms_rate_of_rise_ms = thermoino.mms_rate_of_rise / 1e3  # Convert to °C/ms
    temp_course_resampled_diff = thermoino.ctc * mms_rate_of_rise_ms

    # Reconstruct the temperature course
    temp_course_resampled = np.cumsum(temp_course_resampled_diff) + temp_course[0]

    # Interpolating manually with a stepwise 'ffill'
    original_length = int(stimulus.duration)
    bin_size_samples = int((bin_size_ms / 1000) * sample_rate)  # Convert ms to samples
    expanded_steps = np.repeat(temp_course_resampled, bin_size_samples)[
        : original_length * sample_rate
    ]

    # Plotting the reconstructed against the original temperature course
    plt.figure(figsize=(10, 5))
    plt.plot(expanded_steps, label="Reconstructed")
    plt.plot(temp_course, label="Original")
    plt.title("Comparison of Original and Reconstructed Temperature Profiles")
    plt.xlabel("Time (samples)")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(True)
    plt.xticks(
        np.arange(0, len(expanded_steps), sample_rate),
        np.arange(0, len(expanded_steps) / sample_rate),
    )
    plt.show()

    # Assert that the inversely transformed temperature course is close to the original
    assert np.allclose(expanded_steps, temp_course, atol=0.5), (
        "The reconstructed temperature does not closely match the original"
    )
