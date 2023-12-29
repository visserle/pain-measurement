# work in progress

import sys
from pathlib import Path
from datetime import datetime
import time
import logging
import platform

import tkinter as tk
from tkinter import messagebox

from src.experiments.psychopy_import import psychopy_import
screeninfo = psychopy_import("screeninfo")

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def log_file_path(exp_dir):
    """Returns a Path object for a log file in the log directory with a timestamped filename for the psychopy experiment."""
    exp_dir = Path(exp_dir)
    file_dir = Path('log')
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
    file_path = exp_dir / file_dir / file_name
    return file_path


def ask_for_confirmation(second_monitor=False):
    """Confirmation dialog window to check if everything is ready to start the experiment."""
    root = tk.Tk()
    root.withdraw()
    # Get screen dimensions and set the root window to specified monitor
    monitors = screeninfo.get_monitors()
    if len(monitors) > 1:
        if second_monitor:
            monitor = monitors[1]
        else:
            monitor = monitors[0]
        root.geometry(f'+{monitor.x}+{monitor.y}')
    else:
        logger.warning("Only one monitor detected.")
    # Bring the window to the front
    root.lift()
    if platform.system() == "Windows":
        root.attributes('-topmost', True)
    response = messagebox.askyesno(
        title = "Alles startklar?",
        message = """
        - MMS Programm umgestellt?\n
        - Sensor Preview geöffnet?\n
        - ... ?""")
    if response:
        logger.debug("Confirmation for experiment start received.")
        response = True
    else:
        logger.error("Confirmation for experiment start denied.")
        response = False
    root.destroy()

    #time.sleep(0.5) # TODO: Maybe this helps to open Psychopy with the mouse focused, not sure though
    
    return response


def rgb255_to_rgb_psychopy(rgb255):
    """Convert a RGB color from 0-255 scale to the [-1,1] coloar scale exclusively used in psychopy.""" 
    return [(x / 127.5) - 1 for x in rgb255]

if __name__ == "__main__":
    ask_for_confirmation()