# work in progress

import logging
import importlib
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

def psychopy_import(package_name: str):
    """
    Helper function to import/install a package in a PsychoPy environment.

    If the specified package is not found in the psychopy env, the script attempts to install it using pip.

    Note: If you are running this function for the first time in a psychopy script, it will most likely throw an error. Simply run the script again and it should work.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        try:
            # Try to install the package using pip in a subprocess
            process = subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            if process.returncode != 0:
                raise Exception("pip installation failed")
            return importlib.import_module(package_name)
        except Exception as exc:
            print(f"Failed to install and import '{package_name}': {exc}")
            raise

screeninfo = psychopy_import("screeninfo")


def ask_for_confirmation(second_monitor=False):
    """Confirmation dialog to check if everything is ready to start the experiment."""
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
        - Kalibrierung durchgeführt?\n
        - Thermoino angeschlossen?\n
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
