import logging
import platform
import tkinter as tk
import warnings
from datetime import datetime
from tkinter import ttk

from src.expyriment.utils import center_window

warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)
import pandas as pd

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


# def ask_for_partcipant_info():
#     """Ask for participant information."""

#     # Function to execute when submit button is clicked
#     def submit_data():
#         participant_info = {}
#         participant_info["id"] = id_entry.get()
#         participant_info["age"] = age_entry.get()
#         participant_info["gender"] = gender_combobox.get()
#         logger.info(f"Participant ID: {participant_info['id']}")
#         logger.info(f"Participant Age: {participant_info['age']}")
#         logger.info(f"Participant Gender: {participant_info['gender']}")
#         # Close the window
#         root.destroy()
#         return participant_info

#     # Create a window
#     root = tk.Tk()
#     root.title("Participant Data Input")
#     root.withdraw()

#     # Configure grid layout
#     root.columnconfigure(0, weight=1)
#     root.columnconfigure(1, weight=3)

#     # Data for creating labels and entries
#     fields = ["ID", "Age", "Gender"]
#     entries = []

#     for i, field in enumerate(fields[:-1]):  # Exclude Gender from this loop
#         label = ttk.Label(root, text=f"{field}:")
#         label.grid(column=0, row=i, sticky=tk.W, padx=5, pady=5)
#         entry = ttk.Entry(root)
#         entry.grid(column=1, row=i, sticky=tk.EW, padx=5, pady=5)
#         entries.append(entry)

#     # Unpack entries to individual variables for easy access
#     id_entry, age_entry = entries

#     # Adding a combobox for gender selection, set to readonly to prevent typing
#     gender_label = ttk.Label(root, text="Gender:")
#     gender_label.grid(column=0, row=len(fields) - 1, sticky=tk.W, padx=5, pady=5)
#     gender_combobox = ttk.Combobox(root, values=["Male", "Female"], state="readonly")
#     gender_combobox.grid(column=1, row=len(fields) - 1, sticky=tk.EW, padx=5, pady=5)

#     submit_button = ttk.Button(root, text="Submit", command=submit_data)
#     submit_button.grid(column=0, row=len(fields), columnspan=2, sticky=tk.EW, padx=5, pady=5)

#     center_window(root)
#     root.deiconify()

#     root.mainloop()



class ParticipantDataApp:
    def __init__(self, master):
        self.master = master
        self.participant_info = {}  # Store participant info here

        self.setup_ui()

    def setup_ui(self):
        self.master.title("Participant Data Input")
        # Configure grid layout
        self.master.columnconfigure(0, weight=1)
        self.master.columnconfigure(1, weight=3)

        fields = ["ID", "Age", "Gender"]
        entries = []

        # Create entries for ID and Age
        for i, field in enumerate(fields[:-1]):  # Exclude Gender from this loop
            label = ttk.Label(self.master, text=f"{field}:")
            label.grid(column=0, row=i, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(self.master)
            entry.grid(column=1, row=i, sticky=tk.EW, padx=5, pady=5)
            entries.append(entry)

        # Unpack entries to individual variables for easy access
        self.id_entry, self.age_entry = entries

        # Create gender combobox
        gender_label = ttk.Label(self.master, text="Gender:")
        gender_label.grid(column=0, row=2, sticky=tk.W, padx=5, pady=5)
        self.gender_combobox = ttk.Combobox(self.master, values=["Male", "Female"], state="readonly")
        self.gender_combobox.grid(column=1, row=2, sticky=tk.EW, padx=5, pady=5)

        submit_button = ttk.Button(self.master, text="Submit", command=self.submit_data)
        submit_button.grid(column=0, row=3, columnspan=2, sticky=tk.EW, padx=5, pady=5)

    def submit_data(self):
        self.participant_info["id"] = self.id_entry.get()
        self.participant_info["age"] = self.age_entry.get()
        self.participant_info["gender"] = self.gender_combobox.get()
        logger.info(f"Participant ID: {self.participant_info['id']}")
        logger.info(f"Participant Age: {self.participant_info['age']}")
        logger.info(f"Participant Gender: {self.participant_info['gender']}")
        self.master.destroy()

def ask_for_participant_info():
    root = tk.Tk()
    root.withdraw()  # Hide window initially
    app = ParticipantDataApp(root)
    center_window(root)
    root.deiconify()  # Show window when ready
    root.mainloop()
    return app.participant_info



def init_excel_file(file_path):
    headers = [
        "time_stamp",
        "id",
        "age",
        "gender",
        "vas0",
        "vas70",
        "baseline_temp",
        "temp_range",
    ]
    if not file_path.exists():
        df = pd.DataFrame(columns=headers)
        df.to_excel(file_path, index=False)


def complete_participant_info(participant_info: dict):
    """
    Add additional information to the participant_info dict:
    - time_stamp
    - baseline_temp
    - temp_range

    Note that the participant_info dict must contain the following keys:
    - id
    - age
    - gender
    - vas0
    - vas70.
    """
    participant_info["time_stamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    participant_info["baseline_temp"] = round(
        (participant_info["vas0"] + participant_info["vas70"]) / 2, 1
    )
    participant_info["temp_range"] = round(participant_info["vas70"] - participant_info["vas0"], 1)
    return participant_info


def add_participant_info(file_path, participant_info: dict):
    """
    Adds a participant to the participants.xlsx file.

    Example usage:
    -------
    ```python
    from participants import add_participant

    add_participant(file_path, participant_info)
    ```
    """
    # Add additional information to the participant_info dict
    if "time_stamp" not in participant_info:
        participant_info = complete_participant_info(participant_info)

    # Create a dataframe with the same order as the participants.xlsx file
    new_order = [
        "time_stamp",
        "id",
        "age",
        "gender",
        "vas0",
        "vas70",
        "baseline_temp",
        "temp_range",
    ]
    new_participant = pd.DataFrame([participant_info], columns=new_order)

    # Append participant info to the participants.xlsx file
    participants_df = pd.read_excel(file_path)
    if participants_df.empty or participants_df.isna().all().all():
        participants_df = new_participant
    else:
        # Check if the last participant is the same as the one you want to add
        last_participant = participants_df.iloc[-1]["id"]
        if last_participant == participant_info["id"]:
            logger.critical(
                f"Participant {participant_info['id']} already exists as the last entry."
            )
        participants_df = pd.concat([participants_df, new_participant], ignore_index=True)
    # Save the updated participants.xlsx file
    participant_info.to_excel(file_path, index=False)
    logger.info(f"Added participant {participant_info['id']} to {file_path}")
    return participant_info


def read_last_participant(file_path): # TODO FIXME
    """
    Returns information about the last participant from the participants.xlsx file.

    Example for usage in psychopy:
    -------
    ```python
    from participants import read_last_participant

    participant_info = read_last_participant()
    ```
    """
    df = pd.read_excel(file_path)
    last_row = df.iloc[-1]

    participant_info = {
        "time_stamp": last_row["time_stamp"],
        "participant": last_row["participant"],
        "age": last_row["age"],
        "gender": last_row["gender"],
        "baseline_temp": last_row["baseline_temp"],
        "temp_range": last_row["temp_range"],
    }

    # Check if the participant data is from today
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in participant_info["time_stamp"]:
        logger.warning(
            f"Participant data from {participant_info['participant']} ({participant_info['time_stamp']}) is not from today."
        )

    logger.info(
        f"Participant data from {participant_info['participant']} ({participant_info['time_stamp']}) loaded."
    )

    return participant_info


if __name__ == "__main__":
    # ask_for_partcipant_info()
    ask_for_participant_info()
