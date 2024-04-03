"""
Note that all pop_ups have to be called before expyriment window creation or bad things
will happen.
"""

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from src.helpers import center_tk_window

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def ask_for_eyetracker_calibration() -> bool:
    user_choice = {"proceed": False}  # Dictionary to store user's choice

    def on_proceed():
        user_choice["proceed"] = True
        root.destroy()

    def on_abort():
        user_choice["proceed"] = False
        root.destroy()

    root = tk.Tk()
    root.withdraw()
    root.title("iMotions")

    label = tk.Label(root, text="Kalibrierung für Eye-Tracking starten?")
    label.pack(pady=10, padx=10)

    abort_button = tk.Button(root, text="Nein", command=on_abort)
    abort_button.pack(side=tk.LEFT, padx=20, pady=20)

    proceed_button = tk.Button(root, text="Ja", command=on_proceed)
    proceed_button.pack(side=tk.RIGHT, padx=20, pady=20)

    center_tk_window(root)
    root.deiconify()
    root.mainloop()

    return user_choice["proceed"]


def ask_for_measurement_start() -> bool:
    """Custom confirmation dialog window with checkboxes for each item."""
    items = [
        "MMS Programm umgestellt?",
        "MMS Trigger-bereit?",
        "Jalousien unten?",
        "Hautareal gewechselt?",
        "iMotions' Kalibrierung bestätigt?",
        "Sensor Preview geöffnet?",
        "Signale überprüft (PPG, EDA, Eyetracking)?",
    ]
    root = tk.Tk()
    root.withdraw()
    dialog = ChecklistDialog(root, items)
    center_tk_window(root)
    response = dialog.show()
    logger.debug(
        f"Confirmation for experiment start {'recieved' if response else 'denied'}."
    )
    return response


class ChecklistDialog:
    def __init__(self, root, items):
        self.root = root
        self.root.title("Ready to Start?")
        self.items = items
        self.response = False
        self.setup_ui()

    def setup_ui(self):
        """Sets up the UI components of the dialog."""
        self._create_checkboxes()
        self._create_proceed_button()
        self.root.deiconify()

    def _create_checkboxes(self):
        """Creates a checkbox for each item."""
        self.check_vars = []
        for item in self.items:
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.root, text=item, variable=var)
            chk.pack(anchor="w", padx=20, pady=5)
            self.check_vars.append(var)

    def _create_proceed_button(self):
        """Creates the proceed button."""
        self.proceed_button = ttk.Button(
            self.root, text="Proceed", command=self.on_proceed
        )
        self.proceed_button.pack(pady=20)

    def on_proceed(self):
        """Handles the proceed button click event."""
        if all(var.get() for var in self.check_vars):
            self.response = True
            self.root.destroy()
        else:
            messagebox.showwarning(
                "Warning", "Please confirm all items before proceeding."
            )

    def show(self):
        """Displays the dialog and returns the user's response."""
        self.root.mainloop()
        return self.response


if __name__ == "__main__":
    ask_for_eyetracker_calibration()
    ask_for_measurement_start()
