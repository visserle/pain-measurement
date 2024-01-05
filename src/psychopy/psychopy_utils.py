# work in progress

import sys
from pathlib import Path
from datetime import datetime
import time
import logging
import platform

import tkinter as tk
from tkinter import messagebox

from src.psychopy.psychopy_import import psychopy_import

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def runs_psychopy_path(exp_dir, sub_dir):
    """
    Returns a Path object for `root/runs/psychopy` for recording a psychopy experiment.
    For a log file it returns the file path in the log directory with a timestamped filename;
    for data it returns the data directory.
    
    This is hacky code. It assumes that the experiment is located in the root/src.psychopy directory while the log directory
    is located in root/runs/psychopy. It also assumes that the experiment directory is named the same as the experiment.
    """
    exp_dir = Path(exp_dir)
    exp_name = exp_dir.name
    work_dir = exp_dir.parent.parent.parent
    # Make sure we are in root directory
    assert work_dir.name == "pain-measurement" or work_dir.name == "pain-placebo"
    exp_dir = work_dir / "runs" / "psychopy" / exp_name
    if sub_dir == "logs":
        logs_dir = exp_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_name = datetime.now().strftime("%Y_%m_%d__%H_%M_%S") + ".log"
        log_path = logs_dir / log_name
        return log_path
    elif sub_dir == "data":
        return exp_dir # psychopy takes care of the rest
    else:
        raise ValueError(f"Invalid sub_dir {sub_dir} provided. Must be either 'logs' (custom loggings) or 'data' (psychopy loggings).")


def ask_for_confirmation(second_monitor=False):
    """Confirmation dialog window to check if everything is ready to start the experiment."""
    screeninfo = psychopy_import("screeninfo")
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
