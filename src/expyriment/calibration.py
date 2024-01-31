import yaml
import logging
from expyriment import design, control, stimuli, misc

from src.expyriment.estimator import BayesianEstimatorVAS
from src.log_config import configure_logging

configure_logging(stream_level=logging.DEBUG)

# Note: wait() has the following signature:
# wait(keys=None, duration=None, wait_for_keyup=False, callback_function=None, process_control_events=True)

def press_space():
    """Press space to continue."""
    exp.keyboard.wait(keys=misc.constants.K_SPACE)


control.set_develop_mode(True)
control.defaults.window_size = (800, 600)
design.defaults.experiment_background_colour = misc.constants.C_DARKGREY

exp = design.Experiment(name="Calibration")

# Load script from YAML file
with open('src/expyriment/calibration_script.yaml', 'r', encoding='utf8') as file:
    script = yaml.safe_load(file)

control.initialize(exp)

# Convert script to stimuli
for key in script.keys():
    script[key] = stimuli.TextBox(
        text=script[key],
        size=[600, 500],
        position=[0, -100],  
        text_size=20,
        text_colour=misc.constants.C_WHITE,
    )

# Preload stimuli
for text in script.values():
    text.preload()
    

correction_after_preexposure = 2.0
# Initialize estimator for VAS 70
temp_start_vas70 = 40.0
trials_vas70 = 7
trials_vas0 = 5


control.start(skip_ready_screen=True)

script["welcome_1"].present()
stimuli.Tone(duration=1000, frequency=4400).play()
press_space()
script["welcome_2"].present()
press_space()
script["welcome_3"].present()
press_space()
script["info_preexposure"].present()
press_space()
stimuli.FixCross(colour=misc.constants.C_WHITE).present()
press_space()
script["question_preexposure"].present()
found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
if found == misc.constants.K_y:
    logging.info("Preexposure was already painful.")
    temp_start_vas70 -= correction_after_preexposure
    script["answer_yes"].present()
elif found == misc.constants.K_n:
    logging.info("Preexposure was not painful.")
    script["answer_no"].present()
misc.Clock().wait(1000)
script["info_vas70_1"].present()
press_space()
script["info_vas70_2"].present()
press_space()
script["info_vas70_3"].present()
estimator_vas70 = BayesianEstimatorVAS(
    vas_value=70, 
    temp_start=temp_start_vas70, 
    temp_std=3.5, 
    trials=trials_vas70
    )
press_space()


for trial in range(estimator_vas70.trials):
    script["question_vas70"].present()
    found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
    if found == misc.constants.K_y:
        estimator_vas70.conduct_trial(response="y",trial=trial)
        script["answer_yes"].present()
    elif found == misc.constants.K_n:
        estimator_vas70.conduct_trial(response="n",trial=trial)
        script["answer_no"].present()
    misc.Clock().wait(1000)

script["info_vas0"].present()
estimator_vas0 = BayesianEstimatorVAS(
    vas_value=0, 
    temp_start=estimator_vas70.get_estimaten() - 3, # TODO
    temp_std=3.5, 
    trials=trials_vas0
    )
press_space()


for trial in range(estimator_vas0.trials):
    script["question_vas0"].present()
    found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
    if found == misc.constants.K_y:
        estimator_vas0.conduct_trial(response="y",trial=trial)
        script["answer_yes"].present()
    elif found == misc.constants.K_n:
        estimator_vas0.conduct_trial(response="n",trial=trial)
        script["answer_no"].present()
    misc.Clock().wait(1000)
script["bye"].present()

control.end()
