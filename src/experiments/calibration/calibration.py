import argparse
import copy
import logging
import random
import sys
import time
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

EXP_NAME = "calibration"
EXP_DIR = Path("src/experiments/calibration")
RESULTS_FILE = Path("data/experiments/calibration_results.csv")
LOG_FILE = Path("runs/experiments/calibration") / datetime.now().strftime(
    "%Y_%m_%d__%H_%M_%S.log"
)


# Parse arguments
parser = argparse.ArgumentParser(description="Run the pain-calibration experiment.")
parser.add_argument(
    "-a",
    "--all",
    action="store_true",
    help="Use all flags for a dry run.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode with dummy participant data. Results will not be saved.",
)
parser.add_argument(
    "-w",
    "--windowed",
    action="store_true",
    help="Run in windowed mode.",
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
    help="Use dummy stimulus.",
)
parser.add_argument(
    "-dt",
    "--dummy_thermoino",
    action="store_true",
    help="Use dummy Thermoino device.",
)
args = parser.parse_args()

# Configure logging
configure_logging(
    stream_level=logging.INFO if not (args.debug or args.all) else logging.DEBUG,
    file_path=LOG_FILE,
)

# Load configurations and script
config = load_configuration(EXP_DIR / "calibration_config.toml")
script = load_script(EXP_DIR / "calibration_script.yaml")
thermoino_config = load_configuration(EXP_DIR.parent / "thermoino_config.toml")
# Expyriment defaults
design.defaults.experiment_background_colour = C_DARKGREY
stimuli.defaults.textline_text_colour = config["experiment"]["element_color"]
stimuli.defaults.textbox_text_colour = config["experiment"]["element_color"]
io.defaults.mouse_show_cursor = False
control.defaults.initialize_delay = 3
# Adjust settings
if args.all:
    logging.debug("Using all flags for a dry run.")
    for flag in vars(args).keys():
        setattr(args, flag, True)
if args.debug or args.windowed:
    control.set_develop_mode(True)
    control.defaults.window_mode = False
    ask_for_calibration_start = lambda: logging.debug(  # noqa: E731
        "Skip asking for calibration start in debug mode."
    )
if args.debug:
    logging.debug(
        "Enabled debug mode with dummy participant data. Results will not be saved."
    )
if args.windowed:
    logging.debug("Running in windowed mode.")
    control.defaults.window_mode = True
    control.defaults.window_size = (860, 600)
if args.muted:
    logging.debug("Muting the audio output.")
if args.dummy_stimulus:
    logging.debug("Using dummy stimulus.")
    config["stimulus"]["iti_duration"] = 0.2
    config["stimulus"]["stimulus_duration"] = 0.2
    config["estimator"]["trials_vas70"] = 2
    config["estimator"]["trials_vas0"] = 2


# Setup experiment
participant_info = (
    read_last_participant() if not args.debug else config["dummy_participant"]
)
id_is_odd = int(participant_info["id"]) % 2  # determine skin area for calibration
skind_areas = range(1, 7) if id_is_odd else range(6, 0, -1)
logging.info(f"Use skin area {skind_areas[-2]} for calibration.")
ask_for_calibration_start()  # pop-up window
time.sleep(1)  # wait for the pop-up to close
exp = design.Experiment(name=EXP_NAME)
exp.set_log_level(0)
control.initialize(exp)
screen_size = exp.screen.size

# Prepare stimuli
audio = copy.deepcopy(script)  # audio shares the same keys
prepare_audio(audio, EXP_DIR / "audio")
prepare_script(
    script,
    text_size=scale_1d_value(config["experiment"]["text_size"], screen_size),
    text_box_size=scale_2d_tuple(config["experiment"]["text_box_size"], screen_size),
)
cross = {}
for name, color in zip(
    ["idle", "pain"],
    [config["experiment"]["element_color"], config["experiment"]["cross_pain_color"]],
):
    cross[name] = stimuli.FixCross(
        size=scale_2d_tuple(config["experiment"]["cross_size"], screen_size),
        line_width=scale_1d_value(
            config["experiment"]["cross_line_width"], screen_size
        ),
        colour=color,
    )
    cross[name].preload()
vas_pictures = {}  # load VAS pictures, scale and move them up for a better fit
for pic in ["unmarked", "marked"]:
    vas_pictures[pic] = stimuli.Picture(
        Path(EXP_DIR / f"vas_{pic}.png").as_posix(),
        position=(0, scale_1d_value(100, screen_size)),
    )
    vas_pictures[pic].scale(scale_1d_value(1.5, screen_size))
    vas_pictures[pic].preload()

# Initialize Thermoino
thermoino = Thermoino(
    mms_baseline=thermoino_config["mms_baseline"],
    mms_rate_of_rise=thermoino_config["mms_rate_of_rise"],
    dummy=args.dummy_thermoino,
)
thermoino.connect()


def apply_hardcoded_temperatures(temperatures: list[float]) -> None:
    """Apply hardcoded temperatures from a list to the Thermoino device."""
    for idx, temp in enumerate(temperatures):
        cross["idle"].present()
        exp.clock.wait_seconds(
            (config["stimulus"]["iti_duration"] + random.randint(0, 1))
            if idx != 0
            else 3
        )
        thermoino.trigger()
        time_to_ramp_up, _ = thermoino.set_temp(temp)
        cross["pain"].present()
        exp.clock.wait_seconds(
            config["stimulus"]["stimulus_duration"] + time_to_ramp_up
        )
        time_to_ramp_down, _ = thermoino.set_temp(thermoino_config["mms_baseline"])
        cross["idle"].present()
        exp.clock.wait_seconds(time_to_ramp_down)


def run_estimation_trials(estimator: BayesianEstimatorVAS) -> None:
    """Run estimation trials and return the final estimate."""
    for trial in range(estimator.trials):
        cross["idle"].present()
        exp.clock.wait_seconds(
            config["stimulus"]["iti_duration"] + random.randint(0, 1)
        )
        thermoino.trigger()
        time_to_ramp_up, _ = thermoino.set_temp(estimator.get_estimate())
        cross["pain"].present()
        exp.clock.wait_seconds(
            config["stimulus"]["stimulus_duration"] + time_to_ramp_up
        )
        time_to_ramp_down, _ = thermoino.set_temp(thermoino_config["mms_baseline"])
        cross["idle"].present()
        exp.clock.wait_seconds(time_to_ramp_down)

        script[f"question_vas{estimator.vas_value}"].present()
        found, _ = exp.keyboard.wait(keys=[K_y, K_n])
        if found == K_y:
            estimator.conduct_trial(response="y", trial=trial)  # chr(K_y) = 'y'
            script["answer_yes"].present()
        elif found == K_n:
            estimator.conduct_trial(response="n", trial=trial)
            script["answer_no"].present()
        exp.clock.wait_seconds(1)


def main():
    # Start experiment
    control.start(skip_ready_screen=True, subject_id=participant_info["id"])
    logging.info("Started calibration.")

    # Introduction
    for text, sound in zip(script[s := "welcome"].values(), audio[s].values()):
        text.present()
        # only plays for 1 ms if args.muted is True, otherwise plays the whole audio
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    # Warm-up trials
    logging.info("Started warm-up trials.")
    apply_hardcoded_temperatures(config["stimulus"]["warmup_temperatures"])
    script["post_warmup"].present()
    audio["post_warmup"].play(maxtime=args.muted)
    exp.keyboard.wait(K_SPACE)

    # Pre-exposure trials
    logging.info("Started pre-exposure trials.")
    apply_hardcoded_temperatures(config["stimulus"]["preexposure_temperatures"])

    # Pre-exposure feedback
    script["question_preexposure"].present()
    audio["question_preexposure"].play(maxtime=args.muted)
    found, _ = exp.keyboard.wait(keys=[K_y, K_n])
    if found == K_y:
        participant_info["preexposure_painful"] = True
        script["answer_yes"].present()
        logging.info("Pre-exposure was painful.")
    elif found == K_n:
        participant_info["preexposure_painful"] = False
        config["estimator"]["temp_start_vas0"] += config["stimulus"][
            "preexposure_correction"
        ]
        script["answer_no"].present()
        logging.info("Pre-exposure was not painful.")
    exp.clock.wait_seconds(1)

    # Pain threshold (VAS 0) estimation
    for (key, text), sound in zip(script[s := "info_vas0"].items(), audio[s].values()):
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)
    estimator_vas0 = BayesianEstimatorVAS(
        vas_value=0,
        trials=config["estimator"]["trials_vas0"],
        temp_start=config["estimator"]["temp_start_vas0"],
        temp_std=config["estimator"]["temp_std_vas0"],
        likelihood_std=config["estimator"]["likelihood_std_vas0"],
    )
    logging.info("Started VAS 0 (pain threshold) estimation.")
    run_estimation_trials(estimator=estimator_vas0)
    script["excellent"].present()  # say something nice to the participant
    audio["excellent"].play(maxtime=args.muted)
    exp.clock.wait_seconds(1.5)

    # VAS 70 estimation
    for (key, text), sound in zip(script[s := "info_vas70"].items(), audio[s].values()):
        # Show VAS pictures, first the unmarked, then the marked one
        if "picture" in str(key):
            if "wait" in str(key):
                vas_pictures["unmarked"].present()
                exp.clock.wait_seconds(3)
                text.present(clear=True, update=False)
                vas_pictures["unmarked"].present(clear=False, update=True)
                exp.keyboard.wait(K_SPACE)
            else:
                if "marked" in str(key):
                    vas_pictures["marked"].present(clear=True, update=False)
                else:
                    vas_pictures["unmarked"].present(clear=True, update=False)
                text.present(clear=False, update=True)
                sound.play(maxtime=args.muted)
                exp.keyboard.wait(K_SPACE)
            continue
        text.present()
        sound.play(maxtime=args.muted)
        exp.keyboard.wait(K_SPACE)

    estimator_vas70 = BayesianEstimatorVAS(
        vas_value=70,
        trials=config["estimator"]["trials_vas70"],
        temp_start=estimator_vas0.get_estimate()
        + config["estimator"]["temp_start_vas70_offset"],
        temp_std=config["estimator"]["temp_std_vas70"],
        likelihood_std=config["estimator"]["likelihood_std_vas70"],
    )
    logging.info("Started VAS 70 estimation.")
    run_estimation_trials(estimator=estimator_vas70)

    # Save participant data
    participant_info["vas0"] = estimator_vas0.get_estimate()
    participant_info["vas70"] = estimator_vas70.get_estimate()
    participant_info["temperature_range"] = round(
        participant_info["vas70"] - participant_info["vas0"], 1
    )
    participant_info["vas0_temps"] = estimator_vas0.temps
    participant_info["vas70_temps"] = estimator_vas70.temps
    logging.info(
        f"Calibrated values: VAS 0 = {participant_info['vas0']}, "
        f"VAS 70 = {participant_info['vas70']}, "
        f"range = {participant_info['temperature_range']}."
    )
    add_participant_info(participant_info, RESULTS_FILE) if not args.debug else None
    if args.debug:
        logging.debug(f"Participant data: {participant_info}")

    # End of Experiment
    script["bye"].present()
    audio["bye"].play(maxtime=args.muted)
    exp.clock.wait_seconds(5)

    control.end()
    thermoino.close()
    logging.info("Calibration successfully finished.")
    sys.exit(0)


if __name__ == "__main__":
    main()
