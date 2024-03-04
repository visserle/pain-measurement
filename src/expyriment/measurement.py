import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.expyriment.imotions import EventRecievingiMotions, RemoteControliMotions
from src.expyriment.participant_data import read_last_participant
from src.expyriment.pop_ups import (
    ask_for_eyetracker_calibration,
    ask_for_measurement_start,
)
from src.expyriment.stimulus_generator import StimulusGenerator
from src.expyriment.thermoino import ThermoinoComplexTimeCourses
from src.expyriment.utils import (
    load_configuration,
    load_script,
    prepare_script,
    scale_1d_value,
    scale_2d_tuple,
)
from src.expyriment.visual_analogue_scale import VisualAnalogueScale
from src.log_config import close_root_logging, configure_logging

# Constants
EXP_NAME = "pain-measurement"
SCRIPT_PATH = Path("src/expyriment/measurement_script.yaml")
CONFIG_PATH = Path("src/expyriment/measurement_config.toml")
THERMOINO_CONFIG_PATH = Path("src/expyriment/thermoino_config.toml")
LOG_DIR = Path("runs/expyriment/measurement/")
PARTICIPANTS_EXCEL_PATH = LOG_DIR.parent / "participants.xlsx"

# Configure logging
log_file = LOG_DIR / datetime.now().strftime("%Y_%m_%d__%H_%M_%S.log")
configure_logging(stream_level=logging.DEBUG, file_path=log_file)

# Load configurations and script
config = load_configuration(CONFIG_PATH)
SCRIPT = load_script(SCRIPT_PATH)
THERMOINO = load_configuration(THERMOINO_CONFIG_PATH)
EXPERIMENT = config["experiment"]
STIMULUS = config["stimulus"]
IMOTIONS = config["imotions"]
VAS = config["visual_analogue_scale"]

# Create an argument parser
parser = argparse.ArgumentParser(
    description="Run the pain-measurement experiment. Dry by default."
)
parser.add_argument("-a", "--all", action="store_true", help="Enable all features")
parser.add_argument(
    "-f", "--full_screen", action="store_true", help="Run in full screen mode"
)
parser.add_argument(
    "-s", "--full_stimulus", action="store_true", help="Use full stimulus duration"
)
parser.add_argument(
    "-p", "--participant", action="store_true", help="Use real participant data"
)
parser.add_argument(
    "-t", "--thermoino", action="store_true", help="Enable Thermoino device"
)
parser.add_argument(
    "-i", "--imotions", action="store_true", help="Enable iMotions integration"
)
args = parser.parse_args()

# Adjust settings
if args.all:
    logging.debug("Run full experiment.")
    for flag in vars(args).keys():
        setattr(args, flag, True)
if not args.full_screen:
    control.defaults.window_size = (800, 600)
    control.set_develop_mode(True)
if not args.full_stimulus:
    STIMULUS.update(config["dummy_stimulus"])
    logging.warning("Using dummy stimulus.")
if not args.participant:
    read_last_participant = lambda x: config["dummy_participant"]  # noqa: E731
    logging.warning("Using dummy participant data.")
if not args.imotions:
    ask_for_eyetracker_calibration = (  # noqa: E731
        lambda: logging.debug(
            "Skip asking for eye-tracker calibration because of dummy iMotions."
        )
        or True  # hack to return True
    )
    ask_for_measurement_start = lambda: logging.debug(  # noqa: E731
        "Skip asking for measurement start because of dummy iMotions."
    )

# Expyriment defaults
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.textbox_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.rectangle_colour = EXPERIMENT["element_color"]
io.defaults.eventfile_directory = (LOG_DIR / "events").as_posix()
io.defaults.datafile_directory = (LOG_DIR / "data").as_posix()
io.defaults.outputfile_time_stamp = True
io.defaults.mouse_show_cursor = False
control.defaults.initialize_delay = 3

# Load participant info and update stimulus config with calibration data
participant_info = read_last_participant(PARTICIPANTS_EXCEL_PATH)
STIMULUS.update(participant_info)
random.shuffle(STIMULUS["seeds"])

# Initialize iMotions
imotions_control = RemoteControliMotions(
    study=EXP_NAME, participant_info=participant_info, dummy=not args.imotions
)
imotions_control.connect()
imotions_event = EventRecievingiMotions(
    sample_rate=IMOTIONS["sample_rate"], dummy=not args.imotions
)
imotions_event.connect()
proceed_with_eyetracker_calibration = ask_for_eyetracker_calibration()
if not proceed_with_eyetracker_calibration:
    raise SystemExit("Eye-tracker calibration denied.")
imotions_control.start_study(mode=IMOTIONS["start_study_mode"])
ask_for_measurement_start()

# Experiment setup
exp = design.Experiment(name=EXP_NAME)
control.initialize(exp)
screen_size = exp.screen.size
prepare_script(
    SCRIPT,
    text_box_size=scale_2d_tuple(EXPERIMENT["text_box_size"], screen_size),
    text_size=scale_1d_value(EXPERIMENT["text_size"], screen_size),
)
vas_slider = VisualAnalogueScale(experiment=exp, vas_config=VAS)

# Initialize Thermoino
thermoino = ThermoinoComplexTimeCourses(
    port=THERMOINO["port"],
    mms_baseline=THERMOINO["mms_baseline"],
    mms_rate_of_rise=THERMOINO["mms_rate_of_rise"],
    dummy=not args.thermoino,
)
thermoino.connect()


def get_data_points(temp_course):
    """Get rating and temperature data points and send them to iMotions (run in callback)."""
    stopped_time = exp.clock.stopwatch_time
    vas_slider.rate()
    index = int((stopped_time / 1000) * STIMULUS["sample_rate"])
    imotions_event.send_data_rate_limited(
        timestamp=stopped_time,
        temperature=temp_course[index],
        rating=vas_slider.rating,
        debug=not args.imotions,
    )


def main():
    # Start experiment
    reward = 0
    control.start(skip_ready_screen=True, subject_id=participant_info["id"])
    logging.info(f"Started experiment with seed order {STIMULUS['seeds']}.")

    # Introduction
    for text in SCRIPT["welcome"].values():
        text.present()
        exp.keyboard.wait(K_SPACE)

    # Instruction
    for text in SCRIPT["instruction"].values():
        exp.keyboard.wait(
            K_SPACE,
            callback_function=lambda text=text: vas_slider.rate(
                instruction_textbox=text
            ),
        )

    # Ready
    SCRIPT["ready_set_go"].present()
    exp.keyboard.wait(K_SPACE)

    # Trial loop
    total_trials = len(STIMULUS["seeds"])
    for trial, seed in enumerate(STIMULUS["seeds"]):
        logging.info(f"Started trial ({trial + 1}/{total_trials}) with seed {seed}.")

        # Start with a waiting screen for the initalization of the complex time course
        SCRIPT["wait"].present()
        stimulus = StimulusGenerator(config=STIMULUS, seed=seed)
        thermoino.flush_ctc()
        thermoino.init_ctc(bin_size_ms=THERMOINO["bin_size_ms"])
        thermoino.create_ctc(
            temp_course=stimulus.y, sample_rate=STIMULUS["sample_rate"]
        )
        thermoino.load_ctc()
        thermoino.trigger()

        # Present the VAS slider and wait for the temperature to ramp up
        time_to_ramp_up = thermoino.prep_ctc()
        imotions_event.send_prep_markers()
        exp.clock.wait_seconds(
            time_to_ramp_up + 1.5,  # give participant time to prepare
            callback_function=lambda: vas_slider.rate(),
        )

        # Measure temperature and rating
        thermoino.exec_ctc()
        imotions_event.rate_limiter.reset()
        exp.clock.reset_stopwatch()  # used to get the temperature in the callback function
        imotions_event.send_stimulus_markers(seed)
        exp.clock.wait_seconds(
            stimulus.duration - 0.001,  # prevent index out of range error
            callback_function=lambda: get_data_points(temp_course=stimulus.y),
        )
        imotions_event.send_stimulus_markers(seed)
        logging.info("Complex temperature course (CTC) finished.")

        # Account for some delay at the end of the complex time course (see Thermoino documentation)
        exp.clock.wait_seconds(0.5, callback_function=lambda: vas_slider.rate())

        # Ramp down temperature
        time_to_ramp_down, _ = thermoino.set_temp(THERMOINO["mms_baseline"])
        exp.clock.wait_seconds(
            time_to_ramp_down, callback_function=lambda: vas_slider.rate()
        )
        imotions_event.send_prep_markers()
        logging.info(f"Finished trial ({trial + 1}/{total_trials}) with seed {seed}.")

        # Correlation check for reward
        data_points = pd.DataFrame(imotions_event.data_points)
        data_points.set_index("timestamp", inplace=True)
        corr = data_points.corr()["temperature"]["rating"]
        logging.info(f"Correlation between temperature and rating: {corr:.2f}.")
        if corr > 0.6:
            reward += 1
            logging.debug("Rewarding participant.")
            SCRIPT["reward"].present()
            exp.clock.wait_seconds(2.5)
        elif corr < 0.3 or np.isnan(corr):
            logging.error(
                "Correlation is too low. Is the participant paying attention?"
            )
        imotions_event.clear_data_points()

        # End of trial
        if trial == total_trials - 1:
            break
        SCRIPT["next_trial"].present()
        exp.keyboard.wait(K_SPACE)
        SCRIPT["approve"].present()
        exp.keyboard.wait(K_SPACE)

    # End of Experiment
    SCRIPT["bye"].present()
    exp.keyboard.wait(K_SPACE)

    control.end()
    imotions_control.end_study()
    for instance in [thermoino, imotions_event, imotions_control]:
        instance.close()
    logging.info("Experiment finished. Good job!")
    logging.info(f"Participant reward: {reward} €.")
    close_root_logging()


if __name__ == "__main__":
    main()
