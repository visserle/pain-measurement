import yaml
from expyriment import design, control, stimuli, misc

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

control.start(skip_ready_screen=True)

# script["welcome_1"].present(clear=True, update=True)
# stimuli.Tone(duration=1000, frequency=4400).play()
# press_space()
# script["welcome_2"].present(clear=True, update=True)
# press_space()
# script["welcome_3"].present(clear=True, update=True)
# press_space()
# script["info_preexposure"].present(clear=True, update=True)
# press_space()
# stimuli.FixCross(colour=misc.constants.C_WHITE).present(clear=True, update=True)
# press_space()
# script["question_preexposure"].present(clear=True, update=True)
# found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
# if found == misc.constants.K_y:
#     script["answer_yes"].present(clear=True, update=True)
# elif found == misc.constants.K_n:
#     script["answer_no"].present(clear=True, update=True)
# misc.Clock().wait(1000)
script["info_vas70_1"].present(clear=True, update=True)
press_space()
script["info_vas70_2"].present(clear=True, update=True)
press_space()
script["info_vas70_3"].present(clear=True, update=True)
press_space()
script["question_vas70"].present(clear=True, update=True)
found, _ = exp.keyboard.wait(keys=[misc.constants.K_y, misc.constants.K_n])
if found == misc.constants.K_y:
    script["answer_yes"].present(clear=True, update=True)
elif found == misc.constants.K_n:
    script["answer_no"].present(clear=True, update=True)
misc.Clock().wait(1000)



control.end()
