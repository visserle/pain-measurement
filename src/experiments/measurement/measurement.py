import argparse
import copy
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.experiments.measurement.imotions import (
    EventRecievingiMotions,
    RemoteControliMotions,
)
from src.experiments.measurement.stimulus_generator import StimulusGenerator
from src.experiments.measurement.visual_analogue_scale import VisualAnalogueScale
from src.experiments.participant_data import (
    add_participant_info,
    read_last_participant,
)
from src.experiments.pop_ups import (
    ask_for_eyetracker_calibration,
    ask_for_measurement_start,
)
from src.experiments.thermoino import ThermoinoComplexTimeCourses
from src.experiments.utils import (
    load_configuration,
    load_script,
    prepare_audio,
    prepare_script,
    scale_1d_value,
    scale_2d_tuple,
)
from src.log_config import configure_logging

# Paths
EXP_NAME = "pain-measurement"
EXP_DIR = Path("src/experiments/measurement")
MEASUREMENT_RESULTS = Path("data/experiments/measurement_results.csv")
CALIBRATION_RESULTS = Path("data/experiments/calibration_results.csv")
LOG_FILE = Path("runs/experiments/measurement/logs") / datetime.now().strftime(
    "%Y_%m_%d__%H_%M_%S.log"
)


# Parse arguments
parser = argparse.ArgumentParser(description="Run the pain-measurement experiment.")
parser.add_argument(
    "-a",
    "--all",
    action="store_true",
    help="Use all flags",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode using dummy participants. Results will not be saved.",
)
parser.add_argument(
    "-w",
    "--windowed",
    action="store_true",
    help="Run in windowed mode",
)
parser.add_argument(
    "-m",
    "--muted",
    action="store_true",
    help="Mute the audio output.",
)
parser.add_argument(
    "-ds",
    "--dummy_stimulus",
    action="store_true",
    help="Use dummy stimulus",
)
parser.add_argument(
    "-dt",
    "--dummy_thermoino",
    action="store_true",
    help="Use dummy Thermoino device",
)
parser.add_argument(
    "-di",
    "--dummy_imotions",
    action="store_true",
    help="Use dummy iMotions",
)
args = parser.parse_args()

# Configure logging
configure_logging(
    stream_level=logging.INFO if not (args.debug or args.all) else logging.DEBUG,
    file_path=LOG_FILE,
)

# Load configurations and script
config = load_configuration(EXP_DIR / "measurement_config.toml")
script = load_script(EXP_DIR / "measurement_script.yaml")
thermoino_config = load_configuration(EXP_DIR.parent / "thermoino_config.toml")
# Experiment settings
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = config["experiment"]["element_color"]
stimuli.defaults.textbox_text_colour = config["experiment"]["element_color"]
stimuli.defaults.rectangle_colour = config["experiment"]["element_color"]
io.defaults.outputfile_time_stamp = True
io.defaults.mouse_show_cursor = False
control.defaults.initialize_delay = 3
# Adjust settings
if args.all:
    logging.debug("Using all flags for a dry run.")
    for flag in vars(args).keys():
        setattr(args, flag, True)
if args.debug or args.windowed:
    control.set_develop_mode(True)
if args.debug:
    read_last_participant = lambda *args, **kwargs: config["dummy_participant"]  # noqa: E731
    add_participant_info = lambda *args, **kwargs: logging.debug(  # noqa: E731
        f"Participant data: {args[0]}."
    )
    logging.debug(
        "Enabled debug mode with dummy participant data. "
        "Participant data will not be saved."
    )
if args.windowed:
    logging.debug("Run in windowed mode.")
    control.defaults.window_size = (860, 600)
if args.muted:
    logging.debug("Muting the audio output.")
if args.dummy_stimulus:
    logging.debug("Using dummy stimulus.")
    config["stimulus"] |= config["dummy_stimulus"]
if args.dummy_imotions:
    ask_for_eyetracker_calibration = (  # noqa: E731
        lambda: logging.debug(
            "Skip asking for eye-tracker calibration because of dummy iMotions."
        )
        or True  # hack to return True
    )
    ask_for_measurement_start = lambda: logging.debug(  # noqa: E731
        "Skip asking for measurement start because of dummy iMotions."
    )


# Load participant info and update stimulus config with calibration data
participant_info = read_last_participant(CALIBRATION_RESULTS)
participant_info["vas0"] = float(participant_info["vas0"])
participant_info["vas70"] = float(participant_info["vas70"])
participant_info["temperature_range"] = float(participant_info["temperature_range"])
participant_info["temperature_baseline"] = round(
    (participant_info["vas0"] + participant_info["vas70"]) / 2, 2
)
# check if VAS 70 is too high (risk of burn)
readjustment = False
if participant_info["vas70"] > 48.0:
    readjustment = True
    participant_info["vas70"] = 48.0
    logging.warning("VAS 70 is too high. Adjusting maximum temperature to 48.0 °C.")
# check if delta is too low (under 1.5 °C); round results to avoid float weirdness
temp_range = participant_info["vas70"] - participant_info["vas0"]
if temp_range < 1.5:
    readjustment = True
    missing_amount = 1.5 - temp_range
    participant_info["vas70"] = round(participant_info["vas70"] + missing_amount, 1)

    if participant_info["vas70"] > 48.0:
        overflow = participant_info["vas70"] - 48.0
        participant_info["vas0"] = round(participant_info["vas0"] - overflow, 1)
        participant_info["vas70"] = 48.0
    logging.warning("Temperature range is too low. Adjusting delta to 1.5 °C.")
# calculate new baseline and range
if readjustment:
    participant_info["temperature_baseline"] = round(
        (participant_info["vas0"] + participant_info["vas70"]) / 2, 1
    )
    participant_info["temperature_range"] = round(
        (participant_info["vas70"] - participant_info["vas0"]), 1
    )
    logging.info(
        f"New values: VAS 70 = {participant_info['vas70']} °C, "
        f"VAS 0 = {participant_info['vas0']} °C, "
        f"baseline = {participant_info['temperature_baseline']} °C, "
        f"range = {participant_info['temperature_range']} °C."
    )

# determine order of skin areas based on participant ID
id_is_odd = int(participant_info["id"]) % 2
skin_areas = range(1, 7) if id_is_odd else range(6, 0, -1)
logging.info(f"Start with skin area {skin_areas[0]}.")
# update config with calibration data
config["stimulus"] |= participant_info
# shuffle seeds for randomization
random.shuffle(config["stimulus"]["seeds"])

# Initialize iMotions
imotions_control = RemoteControliMotions(
    study=EXP_NAME, participant_info=participant_info, dummy=args.dummy_imotions
)
imotions_control.connect()
imotions_event = EventRecievingiMotions(
    sample_rate=config["imotions"]["sample_rate"], dummy=args.dummy_imotions
)
imotions_event.connect()
if not ask_for_eyetracker_calibration():
    raise SystemExit("Eye-tracker calibration denied.")
imotions_control.start_study(mode=config["imotions"]["start_study_mode"])
ask_for_measurement_start()
time.sleep(1)

# Experiment setup
exp = design.Experiment(name=EXP_NAME)
exp.set_log_level(0)
control.initialize(exp)
screen_size = exp.screen.size
audio = copy.deepcopy(script)  # audio needs the keys from the script
prepare_script(
    script,
    text_box_size=scale_2d_tuple(config["experiment"]["text_box_size"], screen_size),
    text_size=scale_1d_value(config["experiment"]["text_size"], screen_size),
)
prepare_audio(audio, EXP_DIR / "audio")
vas_slider = VisualAnalogueScale(experiment=exp, config=config["visual_analogue_scale"])


# Initialize Thermoino
thermoino = ThermoinoComplexTimeCourses(
    mms_baseline=thermoino_config["mms_baseline"],
    mms_rate_of_rise=thermoino_config["mms_rate_of_rise"],
    dummy=args.dummy_thermoino,
)
thermoino.connect()


def get_data_points(stimulus: StimulusGenerator) -> None:
    """
    Get rating and temperature data points and send them to iMotions (run in callback).
    """
    vas_slider.rate()  # slider has its own rate limiters (see VisualAnalogueScale)
    stopped_time = exp.clock.stopwatch_time
    index = int((stopped_time / 1000) * config["stimulus"]["sample_rate"])
    index = min(index, len(stimulus.y) - 1)  # prevent index out of bounds
    imotions_event.send_data_rate_limited(
        timestamp=stopped_time,
        temperature=stimulus.y[index],
        rating=vas_slider.rating,
        debug=args.dummy_imotions,
    )


def main():
    # Start experiment
    control.start(skip_ready_screen=True, subject_id=participant_info["id"])
    logging.info(f"Started measurement with seed order {config['stimulus']['seeds']}.")

    # Introduction
    for text, sound in zip(script[s := "welcome"].values(), audio[s].values()):
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    # Instruction with VAS slider
    for text, sound in zip(script[s := "instruction"].values(), audio[s].values()):
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(
            K_SPACE,
            callback_function=lambda text=text: vas_slider.rate(text),
        )

    # Ready
    for text, sound in zip(script[s := "ready"].values(), audio[s].values()):
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    # Trial loop
    total_trials = len(config["stimulus"]["seeds"])
    correlations = []  # between temperature and rating
    reward = 0.0
    for trial, seed in enumerate(config["stimulus"]["seeds"]):
        logging.info(f"Started trial ({trial + 1}/{total_trials}) with seed {seed}.")

        # Start with a waiting screen for the initalization of the complex time course
        script["wait"].present()
        stimulus = StimulusGenerator(config=config["stimulus"], seed=seed)
        thermoino.flush_ctc()
        thermoino.init_ctc(bin_size_ms=thermoino_config["bin_size_ms"])
        thermoino.create_ctc(
            temp_course=stimulus.y, sample_rate=config["stimulus"]["sample_rate"]
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
        exp.clock.reset_stopwatch()  # needed for the callback
        imotions_event.send_stimulus_markers(seed)
        exp.clock.wait_seconds(
            stimulus.duration,
            callback_function=lambda: get_data_points(stimulus),
        )
        imotions_event.send_stimulus_markers(seed)
        logging.info("Complex temperature course (CTC) finished.")

        # Add delay at the end of the complex time course (see thermoino.py)
        exp.clock.wait_seconds(1, callback_function=lambda: vas_slider.rate())

        # Ramp down temperature
        time_to_ramp_down, _ = thermoino.set_temp(thermoino_config["mms_baseline"])
        exp.clock.wait_seconds(
            time_to_ramp_down, callback_function=lambda: vas_slider.rate()
        )
        imotions_event.send_prep_markers()
        logging.info(f"Finished trial ({trial + 1}/{total_trials}) with seed {seed}.")

        # Log and reward participant
        data_points = pd.DataFrame(imotions_event.data_points)
        data_points.set_index("timestamp", inplace=True)
        correlation = np.round(data_points.corr()["temperature"]["rating"], 2).item()
        correlations.append(correlation)
        logging.info(
            f"VAS ratings: "
            f"min = {int(data_points['rating'].min())}, "
            f"max = {int(data_points['rating'].max())}, "
            f"mean = {int(data_points['rating'].mean())}, "
            f"std = {int(data_points['rating'].std())}."
        )
        # warning if pain rating is not covering the full spectrum
        if not (
            (data_points["rating"]).min() == 0 and data_points["rating"].max() == 100
        ):
            logging.warning("Pain rating is not covering the full spectrum. ")
        logging.info(f"Correlation between temperature and rating: {correlation}")
        if correlation > 0.7:
            reward += 0.5
            logging.info("Rewarding participant.")
            script["reward"].present()
            exp.clock.wait_seconds(2.5)
        elif correlation < 0.3 or np.isnan(correlation):
            logging.error(
                "Correlation is too low. Is the participant paying attention?"
            )
        imotions_event.clear_data_points()

        # Next trial
        if trial == total_trials - 1:
            break
        logging.info(
            f"Next, use skin area {skin_areas[(trial + 1) % len(skin_areas)]}."
        )
        script["next_trial"].present()
        audio["next_trial"].play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)
        script["approve"].present()
        exp.keyboard.wait(K_SPACE)

    # Save participant data
    participant_info_ = read_last_participant()  # reload to remove calibration data
    participant_info_["readjustment"] = readjustment
    participant_info_["seed_order"] = config["stimulus"]["seeds"]
    participant_info_["correlations"] = correlations
    participant_info_["reward"] = reward
    add_participant_info(participant_info_, MEASUREMENT_RESULTS)

    # End of Experiment
    script["bye"].present()
    audio["bye"].play(maxtime=args.muted)
    exp.clock.wait_seconds(7)

    control.end()
    imotions_control.end_study()
    for instance in [thermoino, imotions_event, imotions_control]:
        instance.close()
    logging.info("Measurement successfully finished.")
    logging.info(f"Participant reward: {reward} €.")
    sys.exit(0)


if __name__ == "__main__":
    main()
