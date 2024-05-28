import argparse
import copy
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE, K_n, K_y

from src.experiments.calibration.estimator import BayesianEstimatorVAS
from src.experiments.participant_data import (
    add_participant_info,
    read_last_participant,
)
from src.experiments.pop_ups import ask_for_calibration_start
from src.experiments.thermoino import Thermoino
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
EXP_NAME = "pain-calibration"
EXP_DIR = Path("src/experiments/calibration")
AUDIO_DIR = EXP_DIR / "audio"
RESULTS_DIR = Path("data/experiments")
RUN_DIR = Path("runs/experiments/calibration")
RUN_DIR.mkdir(parents=True, exist_ok=True)  # needed for expyriment
LOG_DIR = RUN_DIR / "logs"

SCRIPT_FILE = EXP_DIR / "calibration_script.yaml"
CONFIG_FILE = EXP_DIR / "calibration_config.toml"
THERMOINO_CONFIG_FILE = EXP_DIR.parent / "thermoino_config.toml"
CALIBRATION_RESULTS = RESULTS_DIR / "calibration_results.csv"
log_file = LOG_DIR / datetime.now().strftime("%Y_%m_%d__%H_%M_%S.log")

# Load configurations and script
config = load_configuration(CONFIG_FILE)
SCRIPT = load_script(SCRIPT_FILE)
EXPERIMENT = config["experiment"]
ESTIMATOR = config["estimator"]
STIMULUS = config["stimulus"]
JITTER = random.randint(0, STIMULUS["iti_max_jitter"])

# Create an argument parser
parser = argparse.ArgumentParser(description="Run the pain-calibration experiment.")
parser.add_argument("-a", "--all", action="store_true", help="Use all flags")
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode with dummy participant data. Results will not be saved.",
)
parser.add_argument(
    "-w", "--windowed", action="store_true", help="Run in windowed mode."
)
parser.add_argument(
    "-ds", "--dummy_stimulus", action="store_true", help="Use dummy stimulus."
)
parser.add_argument(
    "-dt", "--dummy_thermoino", action="store_true", help="Use dummy Thermoino device."
)
args = parser.parse_args()

# Configure logging
configure_logging(
    stream_level=logging.INFO if not (args.debug or args.all) else logging.DEBUG,
    file_path=log_file if not (args.debug or args.all) else None,
)

# Adjust settings
if args.all:
    logging.debug("Using all flags for a dry run.")
    for flag in vars(args).keys():
        setattr(args, flag, True)
if args.debug or args.windowed:
    control.set_develop_mode(True)
    ask_for_calibration_start = lambda: logging.debug(  # noqa: E731
        "Skip asking for calibration start in debug mode."
    )
if args.debug:
    logging.debug(
        "Enabled debug mode with dummy participant data. Results will not be saved."
    )
if args.windowed:
    logging.debug("Running in windowed mode.")
    control.defaults.window_size = (860, 600)
if args.dummy_stimulus:
    logging.debug("Using dummy stimulus.")
    STIMULUS["iti_duration"] = 0.2
    STIMULUS["stimulus_duration"] = 0.2
    JITTER = 0
    ESTIMATOR["trials_vas70"] = 2
    ESTIMATOR["trials_vas0"] = 2

# Expyriment defaults
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.textbox_text_colour = EXPERIMENT["element_color"]
io.defaults.eventfile_directory = (LOG_DIR.parent / "events").as_posix()
io.defaults.datafile_directory = (LOG_DIR.parent / "data").as_posix()
io.defaults.outputfile_time_stamp = True
io.defaults.mouse_show_cursor = False
control.defaults.initialize_delay = 3

# Experiment setup
participant_info = (
    read_last_participant() if not args.debug else config["dummy_participant"]
)
# determine order of skin areas based on participant ID
id_is_odd = int(participant_info["id"]) % 2
SKIN_AREAS = range(1, 7) if id_is_odd else range(6, 0, -1)
logging.info(f"Use skin area {SKIN_AREAS[-2]} for calibration.")

# Pop-up window before expyriment starts
ask_for_calibration_start()
exp = design.Experiment(name=EXP_NAME)
control.initialize(exp)
screen_size = exp.screen.size

# Prepare stimuli objects
AUDIO = copy.deepcopy(SCRIPT)  # audio needs the keys from the script
prepare_script(
    SCRIPT,
    text_size=scale_1d_value(EXPERIMENT["text_size"], screen_size),
    text_box_size=scale_2d_tuple(EXPERIMENT["text_box_size"], screen_size),
)
prepare_audio(AUDIO, AUDIO_DIR)

cross = {}
for name, color in zip(
    ["idle", "pain"], [EXPERIMENT["element_color"], EXPERIMENT["cross_pain_color"]]
):
    cross[name] = stimuli.FixCross(
        size=scale_2d_tuple(EXPERIMENT["cross_size"], screen_size),
        line_width=scale_1d_value(EXPERIMENT["cross_line_width"], screen_size),
        colour=color,
    )
    cross[name].preload()

# Load VAS pictures, move it a bit up and scale it for a nice fit
vas_pictures = {}
for pic in ["unmarked", "marked"]:
    vas_pictures[pic] = stimuli.Picture(
        Path(EXP_DIR / f"vas_{pic}.png").as_posix(),
        position=(0, scale_1d_value(100, screen_size)),
    )
    vas_pictures[pic].scale(scale_1d_value(1.5, screen_size))
    vas_pictures[pic].preload()

# Initialize Thermoino
# (we load the config here and not at the top because of ask_for_calibration_start
# where we may need to change the port in the config file)
THERMOINO = load_configuration(THERMOINO_CONFIG_FILE)
thermoino = Thermoino(
    port=THERMOINO["port"],
    mms_baseline=THERMOINO["mms_baseline"],
    mms_rate_of_rise=THERMOINO["mms_rate_of_rise"],
    dummy=args.dummy_thermoino,
)
thermoino.connect()


def run_estimation_trials(estimator: BayesianEstimatorVAS) -> None:
    """Run estimation trials and return the final estimate."""
    for trial in range(estimator.trials):
        cross["idle"].present()
        exp.clock.wait_seconds(STIMULUS["iti_duration"] + JITTER)
        thermoino.trigger()
        time_to_ramp_up, _ = thermoino.set_temp(estimator.get_estimate())
        cross["pain"].present()
        exp.clock.wait_seconds(STIMULUS["stimulus_duration"] + time_to_ramp_up)
        time_to_ramp_down, _ = thermoino.set_temp(THERMOINO["mms_baseline"])
        cross["idle"].present()
        exp.clock.wait_seconds(time_to_ramp_down)

        SCRIPT[f"question_vas{estimator.vas_value}"].present()
        found, _ = exp.keyboard.wait(keys=[K_y, K_n])
        if found == K_y:
            estimator.conduct_trial(response="y", trial=trial)
            SCRIPT["answer_yes"].present()
        elif found == K_n:
            estimator.conduct_trial(response="n", trial=trial)
            SCRIPT["answer_no"].present()
        exp.clock.wait_seconds(1)
    success = estimator.validate_steps()
    if not success:
        logging.error("Please repeat the calibration if applicable.")
        SCRIPT["fail"].present()
        AUDIO["fail"].play()
        exp.clock.wait_seconds(3)
        control.end()
        sys.exit(1)


def main():
    # Start experiment
    control.start(skip_ready_screen=True, subject_id=participant_info["id"])
    logging.info("Started calibration.")

    # Introduction
    for text, audio in zip(SCRIPT[s := "welcome"].values(), AUDIO[s].values()):
        text.present()
        audio.play()
        exp.keyboard.wait(K_SPACE)

    # Pre-exposure Trials
    logging.info("Started pre-exposure trials.")
    for idx, temp in enumerate(STIMULUS["preexposure_temperatures"]):
        cross["idle"].present()
        iti_duration = (
            STIMULUS["iti_duration"] if idx != 0 else STIMULUS["iti_duration_short"]
        )
        exp.clock.wait_seconds(iti_duration + JITTER)
        thermoino.trigger()
        time_to_ramp_up, _ = thermoino.set_temp(temp)
        cross["pain"].present()
        exp.clock.wait_seconds(STIMULUS["stimulus_duration"] + time_to_ramp_up)
        time_to_ramp_down, _ = thermoino.set_temp(THERMOINO["mms_baseline"])
        cross["idle"].present()
        exp.clock.wait_seconds(time_to_ramp_down)

    # Pre-exposure Feedback
    SCRIPT["question_preexposure"].present()
    AUDIO["question_preexposure"].play()
    found, _ = exp.keyboard.wait(keys=[K_y, K_n])
    if found == K_y:
        participant_info["preexposure_painful"] = True
        ESTIMATOR["temp_start_vas70"] -= STIMULUS["preexposure_correction"]
        SCRIPT["answer_yes"].present()
        logging.info("Pre-exposure was painful.")
    elif found == K_n:
        participant_info["preexposure_painful"] = False
        SCRIPT["answer_no"].present()
        logging.info("Pre-exposure was not painful.")
    exp.clock.wait_seconds(1)

    # VAS 70 estimation
    for (key, text), audio in zip(SCRIPT[s := "info_vas70"].items(), AUDIO[s].values()):
        # Show VAS pictures, first the unmarked, then the marked one
        if "picture" in str(key):
            if "wait" in str(key):
                vas_pictures["unmarked"].present()
                exp.clock.wait_seconds(3)
                text.present(clear=True, update=False)
                vas_pictures["unmarked"].present(clear=False, update=True)
                exp.keyboard.wait(K_SPACE)
            else:
                vas_pictures["marked"].present(clear=True, update=False)
                text.present(clear=False, update=True)
                audio.play()
                exp.keyboard.wait(K_SPACE)
            continue
        text.present()
        audio.play()
        exp.keyboard.wait(K_SPACE)

    estimator_vas70 = BayesianEstimatorVAS(
        vas_value=70,
        trials=ESTIMATOR["trials_vas70"],
        temp_start=ESTIMATOR["temp_start_vas70"],
        temp_std=ESTIMATOR["temp_std_vas70"],
        likelihood_std=ESTIMATOR["likelihood_std_vas70"],
    )
    logging.info("Started VAS 70 estimation.")
    run_estimation_trials(estimator=estimator_vas70)
    participant_info["vas70"] = estimator_vas70.get_estimate()
    SCRIPT["excellent"].present()  # say something nice to the participant
    AUDIO["excellent"].play()
    exp.clock.wait_seconds(1.5)

    # Pain threshold (VAS 0) estimation
    SCRIPT["info_vas0"].present()
    AUDIO["info_vas0"].play()
    exp.keyboard.wait(K_SPACE)
    estimator_vas0 = BayesianEstimatorVAS(
        vas_value=0,
        trials=ESTIMATOR["trials_vas0"],
        temp_start=estimator_vas70.get_estimate() - ESTIMATOR["temp_start_vas0_offset"],
        temp_std=ESTIMATOR["temp_std_vas0"],
        likelihood_std=ESTIMATOR["likelihood_std_vas0"],
    )
    logging.info("Started VAS 0 (pain threshold) estimation.")
    run_estimation_trials(estimator=estimator_vas0)
    participant_info["vas0"] = estimator_vas0.get_estimate()

    # Check if the temperature range is reasonable
    temperature_range = participant_info["vas70"] - participant_info["vas0"]
    if not 1 <= temperature_range <= 5:
        range_error = "close together" if temperature_range < 1 else "far apart"
        logging.error(
            f"VAS 70 and VAS 0 are too {range_error}. Please repeat the calibration."
        )
        SCRIPT["fail"].present()
        AUDIO["fail"].play()
        exp.clock.wait_seconds(3)
        control.end()
        sys.exit(1)

    # Save participant data
    participant_info["temperature_baseline"] = round(
        (participant_info["vas0"] + participant_info["vas70"]) / 2, 1
    )
    participant_info["temperature_range"] = round(
        participant_info["vas70"] - participant_info["vas0"], 1
    )
    participant_info["vas70_temps"] = estimator_vas70.temps
    participant_info["vas0_temps"] = estimator_vas0.temps
    add_participant_info(
        participant_info, CALIBRATION_RESULTS
    ) if not args.debug else None
    if args.debug:
        logging.debug(f"Participant data: {participant_info}")

    # End of Experiment
    SCRIPT["bye"].present()
    AUDIO["bye"].play()
    exp.clock.wait_seconds(7)

    control.end()
    thermoino.close()
    logging.info("Calibration successfully finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
