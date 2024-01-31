import json
import logging
from pathlib import Path

import toml
import yaml
from expyriment import control, design, misc, stimuli

from src.expyriment.estimator import BayesianEstimatorVAS
from src.log_config import configure_logging

configure_logging(stream_level=logging.DEBUG)

# Load config from TOML file
with open('src/expyriment/calibration_config.toml', 'r', encoding='utf8') as file:
    config = toml.load(file)

# Load SCRIPT from YAML file
with open('src/expyriment/calibration_SCRIPT.yaml', 'r', encoding='utf8') as file:
    SCRIPT = yaml.safe_load(file)

EXP_NAME = "pain-calibration"
THERMOINO = config['thermoino']
EXPERIMENT = config['experiment']
STIMULUS = config['stimulus']
ESTIMATOR = config['estimator']

DEVELOP_MODE = True
if DEVELOP_MODE:
    control.set_develop_mode(True)
    STIMULUS["iti_duration"] = 500
    STIMULUS["stimulus_duration"] = 200

control.defaults.window_size = (800, 600)
design.defaults.experiment_background_colour = misc.constants.C_DARKGREY
stimuli.defaults.textline_text_colour = EXPERIMENT["element_color"]
stimuli.defaults.textbox_text_colour = EXPERIMENT["element_color"]

# Note: wait() has the following signature:
# wait(keys=None, duration=None, wait_for_keyup=False, callback_function=None, process_control_events=True)

def press_space():
    """Press space to continue."""
    exp.keyboard.wait(keys=misc.constants.K_SPACE)



exp = design.Experiment(name=EXP_NAME)

control.initialize(exp)

# Convert scripts to stimuli
for key in SCRIPT.keys():
    SCRIPT[key] = stimuli.TextBox(
        text=SCRIPT[key],
        size=[600, 500],
        position=[0, -100],
        text_size=20,
    )

# Preload stimuli
for text in SCRIPT.values():
    text.preload()

cross_idle = stimuli.FixCross(colour=EXPERIMENT["element_color"])
cross_pain = stimuli.FixCross(colour=EXPERIMENT["cross_pain_color"])
cross_idle.preload
cross_pain.preload

control.start(skip_ready_screen=True)


SCRIPT["welcome_1"].present()
press_space()
SCRIPT["welcome_2"].present()
press_space()
SCRIPT["welcome_3"].present()
press_space()
SCRIPT["info_preexposure"].present()
press_space()

for idx, _ in enumerate(STIMULUS["preexposure_temperatures"]):
    cross_idle.present()
    # For the first trial, we use a shorter ITI
    misc.Clock().wait(STIMULUS["iti_duration"] if not idx == 0 else STIMULUS["iti_duration_short"])
    cross_pain.present()
    misc.Clock().wait(STIMULUS["stimulus_duration"])

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
SCRIPT["info_vas70_1"].present()
press_space()
SCRIPT["info_vas70_2"].present()
press_space()
SCRIPT["info_vas70_3"].present()

estimator_vas70 = BayesianEstimatorVAS(
    vas_value=70,
    temp_start=ESTIMATOR["temp_start_vas70"],
    temp_std=ESTIMATOR["temp_std_vas70"],
    trials=ESTIMATOR["trials_vas70"]
    )

press_space()

for trial in range(estimator_vas70.trials):
    SCRIPT["question_vas70"].present()
    found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
    if found == misc.constants.K_y:
        success = estimator_vas70.conduct_trial(response="y",trial=trial)
        SCRIPT["answer_yes"].present()
    elif found == misc.constants.K_n:
        success = estimator_vas70.conduct_trial(response="n",trial=trial)
        SCRIPT["answer_no"].present()
    misc.Clock().wait(1000)
# Play a warn signal if all steps of the calibration were in the same direction 
if not success:
    stimuli.Tone(duration=1000, frequency=440).play()

SCRIPT["info_vas0"].present()
estimator_vas0 = BayesianEstimatorVAS(
    vas_value=0,
    temp_start=estimator_vas70.get_estimate() - ESTIMATOR["temp_start_vas0_offset"],
    temp_std=ESTIMATOR["temp_std_vas0"],
    trials=ESTIMATOR["trials_vas0"]
    )

press_space()

for trial in range(estimator_vas0.trials):
    SCRIPT["question_vas0"].present()
    found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
    if found == misc.constants.K_y:
        success = estimator_vas0.conduct_trial(response="y",trial=trial)
        SCRIPT["answer_yes"].present()
    elif found == misc.constants.K_n:
        sucess = estimator_vas0.conduct_trial(response="n",trial=trial)
        SCRIPT["answer_no"].present()
    misc.Clock().wait(1000)
# Play a warn signal if all steps of the calibration were in the same direction 
if not success:
    stimuli.Tone(duration=1000, frequency=440).play()

SCRIPT["bye"].present()

control.end()
