# Note that the apps have to be called before an expyriment experiment is initialized or bad things will happen.
import logging
import tkinter as tk
from tkinter import messagebox, ttk

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def center_tk_window(window: tk.Tk):
    """Center a window on the primary screen."""
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    center_x = int(screen_width / 2 - width / 2)
    center_y = int(screen_height / 2 - height / 2)
    window.geometry(f"+{center_x}+{center_y}")
    return (center_x, center_y)


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
        "iMotions' Kalibrierung bestätigt?",
        "Sensor Preview geöffnet?",
    ]
    root = tk.Tk()
    root.withdraw()
    dialog = ChecklistDialog(root, items)
    center_tk_window(root)
    response = dialog.show()
    if response:
        logger.debug("Confirmation for experiment start received.")
    else:
        logger.debug("Confirmation for experiment start denied.")
    return response


class ParticipantDataApp:
    """
    A simple GUI for entering participant data. It allows easy access to the participant_info dictionary
    for further processing or validation of participant information.
    """

    def __init__(self, root):
        self.root = root
        self.participant_info = {}
        self._setup_ui()

    def _setup_ui(self):
        """Configures the main UI components for the participant data input form."""
        self._configure_window()
        self._create_data_fields()
        self._create_submit_button()

    def _configure_window(self):
        """Sets window title and configures the grid layout."""
        self.root.title("Participant Data Input")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)

    def _create_data_fields(self):
        """Creates input fields for participant data."""
        fields = ["ID", "Age", "Gender"]
        self.entries = {}
        for i, field in enumerate(fields):
            self._create_field(field, i)

    def _create_field(self, field, row):
        """Creates a single field in the UI."""
        label = ttk.Label(self.root, text=f"{field}:")
        label.grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        if field == "Gender":
            entry = ttk.Combobox(self.root, values=["Male", "Female"], state="readonly")
        else:
            entry = ttk.Entry(self.root)
        entry.grid(column=1, row=row, sticky=tk.EW, padx=5, pady=5)
        self.entries[field] = entry

    def _create_submit_button(self):
        """Creates the submit button."""
        submit_button = ttk.Button(self.root, text="Submit", command=self._submit_data)
        submit_button.grid(
            column=0, row=len(self.entries), columnspan=2, sticky=tk.EW, padx=5, pady=5
        )

    def _submit_data(self):
        """Handles data submission and validation."""
        if self._validate_entries():
            self._extract_data()
            self.root.destroy()

    def _validate_entries(self):
        """Validates the entries to ensure they are not empty and meet specific criteria."""
        for field, entry in self.entries.items():
            value = entry.get().strip()
            if not value:
                messagebox.showwarning("Missing Information", f"{field} is required.")
                return False
            if field == "Age":
                if not value.isdigit():
                    messagebox.showwarning("Invalid Input", "Age must be a number.")
                    return False
        return True

    def _extract_data(self):
        """Extracts and processes data from the input fields."""
        self.participant_info = {
            "id": self.entries["ID"].get().strip(),
            "age": int(self.entries["Age"].get()),
            "gender": self.entries["Gender"].get(),
        }


if __name__ == "__main__":
    ask_for_eyetracker_calibration()
    # ask_for_measurement_start()
