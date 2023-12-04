# work in progress

import logging
import tkinter as tk
from tkinter import messagebox

from src.experiments.psychopy_import import psychopy_import
screeninfo = psychopy_import("screeninfo")

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


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
        logging.warning("Only one monitor detected.")
    # Bring the window to the front
    root.attributes('-topmost', True)
    root.update()
    response = messagebox.askyesno(
        title = "Alles startklar?",
        message = """
        - MMS Programm umgestellt?\n
        - Sensor Preview geöffnet?\n
        - ... ?""",
        parent=root)
    if response:
        logger.info("Confirmation for experiment start received.")
        response = True
    else:
        logger.error("Confirmation for experiment start denied.")
        response = False

    root.destroy()
    return response


def rgb255_to_rgb_psychopy(rgb255):
    """
    Convert an RGB color from 0-255 scale to 0-1 scale.

    Args:
    color_255 (tuple): A tuple of three integers representing the RGB color in 0-255 scale.

    Returns:
    tuple: A tuple of three floats representing the RGB color in 0-1 scale.
    """
    return [(x / 127.5) - 1 for x in rgb255]
