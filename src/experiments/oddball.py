import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import pygame
from expyriment import control, design, io, stimuli
from expyriment.misc.constants import C_DARKGREY, K_SPACE

from src.expyriment.imotions import EventRecievingiMotions, RemoteControliMotions
from src.expyriment.pop_ups import (
    ask_for_measurement_start,
)

pygame.mixer.init()

DEVELOP_MODE = False
EXP_NAME = "oddball"

if DEVELOP_MODE:
    control.defaults.window_size = (800, 600)
    control.set_develop_mode(True)
    ask_for_measurement_start = lambda: logging.info(  # noqa: E731
        "Skip asking for measurement start because of dummy iMotions."
    )

participant_info = {"id": "dummy", "age": 20, "gender": "Female"}

# Stimuli
normal = stimuli.Tone(duration=100, frequency=440)
odd = stimuli.Tone(duration=100, frequency=880)

# Preload stimuli
normal.preload()
odd.preload()

# Experiment logic
random.seed(42)
blocks = range(10)
trials = range(10)
odd_trials = random.sample(trials, 3)


# Initialize iMotions
imotions_control = RemoteControliMotions(
    study=EXP_NAME, participant_info=participant_info, dummy=DEVELOP_MODE
)
imotions_control.connect()
imotions_control.start_study()
imotions_event = EventRecievingiMotions(100, dummy=DEVELOP_MODE)
imotions_event.connect()
ask_for_measurement_start()

# Experiment setup
exp = design.Experiment(name=EXP_NAME)
control.initialize(exp)

# Start experiment
control.start(skip_ready_screen=False)
exp.keyboard.wait()
for block in blocks:
    exp.keyboard.wait()
    for trial in trials:
        if trial in odd_trials:
            imotions_event.send_marker(marker_name="odd_trial", value=trial)
            odd.present()
        else:
            normal.present()
        exp.clock.wait(1000)
        imotions_event
        exp.clock.wait(random.randint(0, 500))

control.end()
imotions_event.close()
imotions_control.end_study()
imotions_control.close()
