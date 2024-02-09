# TODO
# short stimulus
# diff vs gradient for thermoino
# index of the wave (-1 or not)
# add randomization of stimulus order using expyriment
# adujst stimulus sample rate to the rest of the sample rates, should always be the same as for imotions because we send both at the same time


import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.expyriment.imotions import EventRecievingiMotions, RemoteControliMotions
from src.expyriment.participant_data import read_last_participant
from src.expyriment.stimulus_function import StimulusFunction
from src.expyriment.thermoino import ThermoinoComplexTimeCourses
from src.expyriment.tkinter_windows import ask_for_measurement_start
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
CONFIG_PATH = Path("src/expyriment/measurement_config.toml")
SCRIPT_PATH = Path("src/expyriment/measurement_script.yaml")
LOG_DIR = Path("runs/expyriment/measurement/")
PARTICIPANTS_EXCEL_PATH = LOG_DIR.parent / "participants.xlsx"

# Configure logging
log_file = LOG_DIR / datetime.now().strftime("%Y_%m_%d__%H_%M_%S.log")
configure_logging(stream_level=logging.DEBUG, file_path=log_file)

# Load configurations and script
config = load_configuration(CONFIG_PATH)
SCRIPT = load_script(SCRIPT_PATH)
THERMOINO = config["thermoino"]
EXPERIMENT = config["experiment"]
STIMULUS = config["stimulus"]
IMOTIONS = config["imotions"]
VAS = config["visual_analogue_scale"]

# Create an argument parser
parser = argparse.ArgumentParser(description="Run the pain-measurement experiment. Dry by default.")
parser.add_argument("-a", "--all", action="store_true", help="Enable all features")
parser.add_argument("-f", "--full_screen", action="store_true", help="Run in full screen mode")
# TODO
parser.add_argument("-s", "--full_stimuli", action="store_true", help="Use full stimuli duration")
parser.add_argument("-p", "--participant", action="store_true", help="Use real participant data")
parser.add_argument("-t", "--thermoino", action="store_true", help="Enable Thermoino device")
parser.add_argument("-i", "--imotions", action="store_true", help="Enable iMotions integration")
args = parser.parse_args()

# Adjust settings
if args.all:
    for flag in vars(args).keys():
        setattr(args, flag, True)
if not args.full_screen:
    control.defaults.window_size = (800, 600)
    control.set_develop_mode(True)
if not args.participant:
    read_last_participant = lambda x: config["dummy_participant"]
    logging.info("Using dummy participant data.")
if not args.imotions:
    ask_for_measurement_start = lambda: logging.info(
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

# Load participant info
participant_info = read_last_participant(PARTICIPANTS_EXCEL_PATH)

# Initialize iMotions
imotions_control = RemoteControliMotions(
    study=EXP_NAME, participant_info=participant_info, dummy=not args.imotions
)
imotions_control.connect()
imotions_event = EventRecievingiMotions(
    sample_rate=IMOTIONS["sample_rate"], dummy=not args.imotions
)
imotions_event.connect()
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


def get_vas_rating(stimulus_obj: StimulusFunction):
    # Runs rate-limited in the callback function
    stopped_time = exp.clock.stopwatch_time
    vas_slider.rate()
    index = max(
        0, int((stopped_time / 1000) * stimulus_obj.sample_rate) - 1
    )  # TODO check if -1 is necessary
    imotions_event.send_data_rate_limited(
        timestamp=stopped_time,
        temperature=stimulus_obj.wave[index],
        rating=vas_slider.rating,
        debug=not args.imotions,
    )


def main():
    # Start experiment
    control.start(skip_ready_screen=True)

    # Introduction
    for text in SCRIPT["welcome"].values():
        text.present()
        exp.keyboard.wait(K_SPACE)

    # Instruction
    for text in SCRIPT["instruction"].values():
        exp.keyboard.wait(
            K_SPACE,
            callback_function=lambda text=text: vas_slider.rate(instruction_textbox=text),
        )

    # Ready
    SCRIPT["ready_set_go"].present()
    exp.keyboard.wait(K_SPACE)

    # Trial loop
    for trial, seed in enumerate(STIMULUS["seeds"]):
        # Start with a waiting screen for the initalization of the complex time course
        SCRIPT["wait"].present()
        stimulus = (
            StimulusFunction(
                minimal_desired_duration=STIMULUS["minimal_desired_duration"],
                frequencies=1.0 / np.array(STIMULUS["periods"]),
                temp_range=participant_info["temp_range"],
                sample_rate=STIMULUS["sample_rate"],
                desired_big_decreases=STIMULUS["desired_big_decreases"],
                random_periods=STIMULUS["random_periods"],
                seed=seed,
            )
            .add_baseline_temp(baseline_temp=participant_info["baseline_temp"])
            .add_plateaus(
                plateau_duration=STIMULUS["plateau_duration"], n_plateaus=STIMULUS["n_plateaus"]
            )
            .generalize_big_decreases()
        )
        thermoino.flush_ctc()
        thermoino.init_ctc(bin_size_ms=THERMOINO["bin_size_ms"])
        thermoino.create_ctc(temp_course=stimulus.wave, sample_rate=stimulus.sample_rate)
        thermoino.load_ctc()
        thermoino.trigger()
        time_to_ramp_up = thermoino.prep_ctc()
        imotions_event.send_prep_markers()

        exp.clock.wait_seconds(
            time_to_ramp_up,
            callback_function=lambda: vas_slider.rate(),
        )

        # Measurement
        thermoino.exec_ctc()
        imotions_event.rate_limiter.reset()
        exp.clock.reset_stopwatch()  # used to get the temperature in the callback function
        imotions_event.send_stimulus_markers(seed)
        exp.clock.wait_seconds(
            stimulus.duration,
            callback_function=lambda stimulus=stimulus: get_vas_rating(stimulus),
        )
        imotions_event.send_stimulus_markers(seed)

        # NOTE maybe this could fixed by np.diff. TODO check
        # Account for the exec delay of the thermoino (see thermoino.exec_ctc()
        exp.clock.wait_seconds(0.5, callback_function=lambda: vas_slider.rate())

        # End of trial
        time_to_ramp_down, _ = thermoino.set_temp(THERMOINO["mms_baseline"])
        exp.clock.wait_seconds(time_to_ramp_down, callback_function=lambda: vas_slider.rate())
        imotions_event.send_prep_markers()
        thermoino.flush_ctc()
        if trial == len(STIMULUS["seeds"]) - 1:
            break
        SCRIPT["next_trial"].present()
        exp.keyboard.wait(K_SPACE)
        SCRIPT["approve"].present()
        exp.keyboard.wait(K_SPACE)

    #  End of Experiment
    SCRIPT["bye"].present()
    exp.keyboard.wait(K_SPACE)

    thermoino.close()
    imotions_event.close()
    imotions_control.end_study()
    imotions_control.close()
    close_root_logging()


if __name__ == "__main__":
    main()
