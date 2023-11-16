import tkinter as tk
from tkinter import messagebox
import logging
import importlib
import subprocess
import sys

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

def psychopy_import(package_name):
    """
    Helper function to import/install a package in a PsychoPy environment.
    """
    try:
        return importlib.import_module(package_name)
    except ImportError:
        try:
            # Try to install the package using pip in a subprocess
            process = subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
            if process.returncode != 0:
                raise Exception("pip installation failed")
            # Try to import the package again after installation
            return importlib.import_module(package_name)
        except Exception as exc:
            print(f"Failed to install and import '{package_name}': {exc}")
            raise

screeninfo = psychopy_import("screeninfo")


def ask_for_confirmation(second_monitor=False):
    root = tk.Tk()
    root.withdraw()
    # Get screen dimensions and set the root window to the second monitor
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
