# TODO
# formatting of text box
# vas picture

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

from expyriment import control, design, misc, stimuli, io

from src.expyriment.estimator import BayesianEstimatorVAS
from src.expyriment.thermoino import Thermoino
from src.expyriment.thermoino_dummy import ThermoinoDummy
from src.expyriment.utils import load_configuration, load_script, prepare_stimuli, warn_signal
from src.expyriment.participant_data import ask_for_participant_info, add_participant_info
from src.log_config import close_root_logging, configure_logging

# Check if the script is run from the command line
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Run the pain-calibration experiment.")
    parser.add_argument(
        "--dummy", action="store_true", help="Run in development mode."
    )
    args = parser.parse_args()
    DEVELOP_MODE = args.dummy
else:
    # Default mode when not using CLI arguments
    DEVELOP_MODE = True

# Constants
EXP_NAME = "pain-calibration"
CONFIG_PATH = Path("src/expyriment/calibration_config.toml")
SCRIPT_PATH = Path("src/expyriment/calibration_SCRIPT.yaml")
LOG_DIR = Path("runs/expyriment/calibration/")
PARTICIPANTS_EXCEL_PATH = LOG_DIR.parent / "participants.xlsx"

# Configure logging
log_file = LOG_DIR / datetime.now().strftime("%Y_%m_%d__%H_%M_%S.log")
configure_logging(stream_level=logging.DEBUG, file_path=log_file)


# Utility functions
def press_space():
    """Press space to continue."""
    exp.keyboard.wait(keys=misc.constants.K_SPACE)


def present_script_and_wait(script_key):
    """Present a script and wait for space key press."""
    SCRIPT[script_key].present()
    press_space()


# Load configurations and script
config = load_configuration(CONFIG_PATH)
SCRIPT = load_script(SCRIPT_PATH)

# Experiment settings
THERMOINO = config["thermoino"]
EXPERIMENT = config["experiment"]
ESTIMATOR = config["estimator"]
STIMULUS = config["stimulus"]
JITTER = random.randint(0, STIMULUS["iti_max_jitter"]) if not DEVELOP_MODE else 0

# Expyriment defaults
control.defaults.window_size = (800, 600)
design.defaults.experiment_background_colour = misc.constants.C_DARKGREY
stimuli.defaults.textline_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.textbox_text_colour = EXPERIMENT["element_color"]
io.defaults.eventfile_directory = (LOG_DIR / "events").as_posix()
io.defaults.datafile_directory = (LOG_DIR / "data").as_posix()
io.defaults.outputfile_time_stamp = True

# Development mode settings
if DEVELOP_MODE:
    Thermoino = ThermoinoDummy
    control.set_develop_mode(True)
    STIMULUS["iti_duration"] = 300
    STIMULUS["stimulus_duration"] = 200
else:
    participant_info = ask_for_participant_info()

participant_info = ask_for_participant_info() # to be removed TODO
# Experiment setup
exp = design.Experiment(name=EXP_NAME)
control.initialize(exp)
prepare_stimuli(SCRIPT)
cross_idle = stimuli.FixCross(colour=EXPERIMENT["element_color"])
cross_pain = stimuli.FixCross(colour=EXPERIMENT["cross_pain_color"])
cross_idle.preload()
cross_pain.preload()

# Initialize Thermoino
luigi = Thermoino(
    port=THERMOINO["port"],
    mms_baseline=THERMOINO["mms_baseline"],
    mms_rate_of_rise=THERMOINO["mms_rate_of_rise"],
)
luigi.connect()


# Trial functions
def run_preexposure_trials():
    """Run pre-exposure trials with different temperatures."""
    for idx, temp in enumerate(STIMULUS["preexposure_temperatures"]):
        cross_idle.present()
        iti_duration = STIMULUS["iti_duration"] if idx != 0 else STIMULUS["iti_duration_short"]
        misc.Clock().wait(iti_duration + JITTER)
        luigi.trigger()
        time_for_ramp_up, _ = luigi.set_temp(temp)
        cross_pain.present()
        misc.Clock().wait(STIMULUS["stimulus_duration"] + time_for_ramp_up)
        time_for_ramp_down, _ = luigi.set_temp(THERMOINO["mms_baseline"])
        cross_idle.present()
        misc.Clock().wait(time_for_ramp_down)


def run_estimation_trials(estimator: BayesianEstimatorVAS):
    """Run estimation trials and return the final estimate."""
    for trial in range(estimator.trials):
        cross_idle.present()
        misc.Clock().wait(STIMULUS["iti_duration"] + JITTER)
        luigi.trigger()
        time_for_ramp_up, _ = luigi.set_temp(estimator.get_estimate())
        cross_pain.present()
        misc.Clock().wait(STIMULUS["stimulus_duration"] + time_for_ramp_up)
        time_for_ramp_down, _ = luigi.set_temp(THERMOINO["mms_baseline"])
        cross_idle.present()
        misc.Clock().wait(time_for_ramp_down)

        SCRIPT[f"question_vas{estimator.vas_value}"].present()
        found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
        if found == misc.constants.K_y:
            estimator.conduct_trial(response="y", trial=trial)
            SCRIPT["answer_yes"].present()
        elif found == misc.constants.K_n:
            estimator.conduct_trial(response="n", trial=trial)
            SCRIPT["answer_no"].present()
        misc.Clock().wait(1000)
    # Additional warning tone if all steps of the calibration were in the same direction
    if estimator.check_steps():
        warn_signal()
    return estimator.get_estimate()


# Experiment procedure
def main():  
    # Start experiment
    control.start(skip_ready_screen=True)

    # Introduction
    present_script_and_wait("welcome_1")
    present_script_and_wait("welcome_2")
    present_script_and_wait("welcome_3")

    # Pre-exposure Trials
    present_script_and_wait("info_preexposure")
    run_preexposure_trials()

    # Pre-exposure Feedback
    SCRIPT["question_preexposure"].present()
    found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
    if found == misc.constants.K_y:
        ESTIMATOR["temp_start_vas70"] -= STIMULUS["preexposure_correction"]
        SCRIPT["answer_yes"].present()
        logging.info("Preexposure was painful.")
    elif found == misc.constants.K_n:
        SCRIPT["answer_no"].present()
        logging.info("Preexposure was not painful.")
    misc.Clock().wait(1000)

    # VAS 70 Estimation
    present_script_and_wait("info_vas70_1")
    present_script_and_wait("info_vas70_2")
    present_script_and_wait("info_vas70_3")

    estimator_vas70 = BayesianEstimatorVAS(
        vas_value=70,
        temp_start=ESTIMATOR["temp_start_vas70"],
        temp_std=ESTIMATOR["temp_std_vas70"],
        trials=ESTIMATOR["trials_vas70"],
    )
    participant_info["vas70"] = run_estimation_trials(estimator=estimator_vas70)

    # VAS 0 Estimation
    present_script_and_wait("info_vas0")
    estimator_vas0 = BayesianEstimatorVAS(
        vas_value=0,
        temp_start=estimator_vas70.get_estimate() - ESTIMATOR["temp_start_vas0_offset"],
        temp_std=ESTIMATOR["temp_std_vas0"],
        trials=ESTIMATOR["trials_vas0"],
    )
    participant_info["vas0"] = run_estimation_trials(estimator=estimator_vas0)

    # End of Experiment
    present_script_and_wait("bye")

    # Close and clean up
    control.end()
    add_participant_info(PARTICIPANTS_EXCEL_PATH, participant_info)
    luigi.close()
    close_root_logging()


if __name__ == "__main__":
    main()
