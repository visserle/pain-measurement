# TODO
# use callback_function to handle event sending
# + some mod of time to avoid sending too many events
# note that callback functions are called sub 1ms, not in accordance with frame rate
# add coluntdown to rating phase
# add randomization of stimulus order using expyriment

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.expyriment.imotions import EventRecievingiMotions, RemoteControliMotions
from src.expyriment.imotions_dummy import EventRecievingiMotionsDummy, RemoteControliMotionsDummy
from src.expyriment.participant_data import read_last_participant
from src.expyriment.stimulus_function import StimulusFunction
from src.expyriment.thermoino import ThermoinoComplexTimeCourses
from src.expyriment.thermoino_dummy import ThermoinoComplexTimeCoursesDummy
from src.expyriment.tkinter_windows import ask_for_measurement_start
from src.expyriment.utils import (
    load_configuration,
    load_script,
    prepare_script,
    scale_1d_value,
    scale_2d_tuple,
    warn_signal,
)
from src.expyriment.visual_analogue_scale import VisualAnalogueScale
from src.log_config import close_root_logging, configure_logging

# Check if the script is run from the command line
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser(description="Run the pain-measurement experiment.")
    parser.add_argument("--dummy", action="store_true", help="Run in development mode.")
    args = parser.parse_args()
    DEVELOP_MODE = args.dummy
else:
    DEVELOP_MODE = True

# ask_for_measurement_start()


# Constants
NAME = "measurement"
EXP_NAME = f"pain-{NAME}"
CONFIG_PATH = Path(f"src/expyriment/{NAME}_config.toml")
SCRIPT_PATH = Path(f"src/expyriment/{NAME}_script.yaml")
LOG_DIR = Path(f"runs/expyriment/{NAME}/")
PARTICIPANTS_EXCEL_PATH = LOG_DIR.parent / "participants.xlsx"

# Configure logging
log_file = LOG_DIR / datetime.now().strftime("%Y_%m_%d__%H_%M_%S.log")
configure_logging(stream_level=logging.DEBUG, file_path=log_file)


# Load configurations and script
config = load_configuration(CONFIG_PATH)
SCRIPT = load_script(SCRIPT_PATH)

# Experiment settings
THERMOINO = config["thermoino"]
EXPERIMENT = config["experiment"]
STIMULUS = config["stimulus"]
IMOTIONS = config["imotions"]
VAS = config["visual_analogue_scale"]


# Expyriment defaults
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.textbox_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.rectangle_colour = EXPERIMENT["element_color"]
io.defaults.eventfile_directory = (LOG_DIR / "events").as_posix()
io.defaults.datafile_directory = (LOG_DIR / "data").as_posix()
io.defaults.outputfile_time_stamp = True

# Development mode settings
if DEVELOP_MODE:
    ThermoinoComplexTimeCourses = ThermoinoComplexTimeCoursesDummy
    EventRecievingiMotions = EventRecievingiMotionsDummy
    RemoteControliMotions = RemoteControliMotionsDummy
    control.defaults.window_size = (800, 600)
    control.set_develop_mode(True)
    STIMULUS["iti_duration"] = 300
    STIMULUS["stimulus_duration"] = 200
    participant_info = config["dummy_participant"]
else:
    ThermoinoComplexTimeCourses = ThermoinoComplexTimeCoursesDummy  # NOTE REMOVE THIS
    EventRecievingiMotions = EventRecievingiMotionsDummy  # NOTE REMOVE THIS
    RemoteControliMotions = RemoteControliMotionsDummy  # NOTE REMOVE THIS
    participant_info = read_last_participant(PARTICIPANTS_EXCEL_PATH)

participant_info = read_last_participant(PARTICIPANTS_EXCEL_PATH)

# Experiment setup
exp = design.Experiment(name=EXP_NAME)
control.initialize(exp)
screen_size = exp.screen.size
print(screen_size)
prepare_script(
    SCRIPT,
    text_box_size=scale_2d_tuple(EXPERIMENT["text_box_size"], screen_size),
    text_size=scale_1d_value(EXPERIMENT["text_size"], screen_size),
)
vas_slider = VisualAnalogueScale(experiment=exp, vas_config=VAS)

# Initalize stimuli functions
STIMULUS["frequencies"] = 1.0 / np.array(STIMULUS["periods"])
stimuli_functions = {}
for seed in STIMULUS["seeds"]:
    stimulus = (
        StimulusFunction(
            minimal_desired_duration=STIMULUS["minimal_desired_duration"],
            frequencies=STIMULUS["frequencies"],
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
    stimuli_functions[seed] = stimulus

# Initialize iMotions
imotions_control = RemoteControliMotions(study=EXP_NAME, participant_info=participant_info)
imotions_control.connect()
imotions_event = EventRecievingiMotions()
imotions_event.connect()

# Initialize Thermoino
thermoino = ThermoinoComplexTimeCourses(
    port=THERMOINO["port"],
    mms_baseline=THERMOINO["mms_baseline"],
    mms_rate_of_rise=THERMOINO["mms_rate_of_rise"],
)
thermoino.connect()


def prepare_complex_time_course(stimulus: StimulusFunction, thermoino_config: dict) -> float:
    thermoino.flush_ctc()
    thermoino.init_ctc(bin_size_ms=thermoino_config["bin_size_ms"])
    thermoino.create_ctc(temp_course=stimulus.wave, sample_rate=stimulus.sample_rate)
    thermoino.load_ctc()
    thermoino.trigger()
    prep_duration = thermoino.prep_ctc()[1]
    imotions_event.send_prep_markers()
    return prep_duration

def run_measurement_trial():
    # move if statement out of rate function
    vas_slider.rate()
    imotions_event.send_ratings(rating=vas_slider.rating)
    # send temperature
    # using different sample rate
    

def main():
    imotions_control.start_study(mode=IMOTIONS["start_study_mode"])

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
            callback_function=lambda: vas_slider.rate(instruction_textbox=text),
        )

    # Ready
    SCRIPT["ready_set_go"].present()
    exp.keyboard.wait(K_SPACE)

    # Trial loop
    for idx, seed in enumerate(STIMULUS["seeds"]):
        # Preperation
        exp.clock.wait(
            waiting_time=prepare_complex_time_course(
                stimuli_functions[STIMULUS["seeds"][0]], THERMOINO
            ),
            callback_function=lambda: vas_slider.rate(),
        )
        # Measurement
        exp.clock.wait_seconds(
            time_sec=1,  # stimuli_functions[STIMULUS["seeds"][0]].duration,
            callback_function=run_measurement_trial,
        )
        # End of trial
        thermoino.temp = 38
        duration, _ = thermoino.set_temp(THERMOINO["mms_baseline"])
        exp.clock.wait(duration, callback_function=lambda: vas_slider.rate())
        imotions_event.send_prep_markers()

        if idx == len(STIMULUS["seeds"]) - 1:
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