import pytest

from src.experiments.thermoino import Thermoino


@pytest.fixture
def thermoino():
    # Create a Thermoino object in dummy mode for testing
    test_thermoino = Thermoino(mms_baseline=25, mms_rate_of_rise=10, dummy=True)
    test_thermoino.connect()
    yield test_thermoino
    test_thermoino.close()


def test_set_temp_by_inverse_transform(thermoino):
    # Set initial temperature
    thermoino.temp = 25
    # Duration in seconds to reach the target temperature
    duration = 5
    # Convert the duration from seconds to microseconds for the MOVE command calculation
    move_time_us = duration * 1e6

    # Calculate the temperature change based on the move time and rate of rise
    temp_change = (move_time_us / 1e6) * thermoino.mms_rate_of_rise

    # Calculate the target temperature
    temp_target = thermoino.temp + temp_change

    # Perform the operation
    duration_returned, success = thermoino.set_temp(temp_target)

    # Check if the calculated duration and success flag are correct
    assert duration_returned == pytest.approx(duration)
    assert success is True
    # Check if the temperature was correctly updated in the Thermoino object
    assert thermoino.temp == temp_target
