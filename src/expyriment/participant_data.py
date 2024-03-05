import logging
import tkinter as tk
from datetime import datetime

import pandas as pd

from src.expyriment.pop_ups import ParticipantDataApp
from src.helpers import center_tk_window

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


COLUMN_HEADERS = [
    "time_stamp",
    "id",
    "age",
    "gender",
    "vas0",
    "vas70",
    "temperature_baseline",
    "temperature_range",
]


def ask_for_participant_info(file_path):
    root = tk.Tk()
    root.withdraw()
    app = ParticipantDataApp(root)
    center_tk_window(root)
    root.deiconify()
    root.mainloop()

    if app.participant_info:
        participant_info = app.participant_info
        # Check if the participant ID already exists in the file
        if file_path.exists():
            existing_data = pd.read_excel(file_path)
            if participant_info["id"] in existing_data["id"].values:
                logger.critical(f"Participant ID already exists in {file_path}.")
        logger.info(f"Participant ID: {participant_info['id']}")
        logger.info(f"Participant Age: {participant_info['age']}")
        logger.info(f"Participant Gender: {participant_info['gender']}")
        return participant_info
    logger.warning("No participant information entered.")
    return None


def init_excel_file(file_path):
    if not file_path.exists():
        df = pd.DataFrame(columns=COLUMN_HEADERS)
        df.to_excel(file_path, index=False)


def add_participant_info(file_path, participant_info: dict) -> dict:
    """
    Add a participant to the participants.xlsx file.

    Note that the participant_info dict must contain the following keys:
        - id
        - age
        - gender
        - vas0
        - vas70

    The following key will be added:
        - time_stamp
        - temperature_baseline
        - temperature_range
    """
    # Add additional information to the participant_info dict
    if "time_stamp" not in participant_info:
        participant_info = _complete_participant_info(participant_info)

    # Create a dataframe with the same order as the participants.xlsx file
    participant_info_df = pd.DataFrame([participant_info], columns=COLUMN_HEADERS)

    # Append participant info to the participants.xlsx file
    participants_xlsx = pd.read_excel(file_path)
    if participants_xlsx.empty or participants_xlsx.isna().all().all():
        participants_xlsx = participant_info_df
    else:
        # Check if the last participant is the same as the one you want to add
        last_participant = participants_xlsx.iloc[-1]["id"]
        if last_participant == participant_info["id"]:
            logger.warning(
                f"Participant {participant_info['id']} already exists as the last entry."
            )
        participants_xlsx = pd.concat(
            [participants_xlsx, participant_info_df], ignore_index=True
        )
    # Save the updated participants.xlsx file
    participants_xlsx.to_excel(file_path, index=False)
    logger.info(f"Added participant {participant_info['id']} to {file_path}.")

    return participant_info


def _complete_participant_info(participant_info: dict) -> dict:
    """
    Add additional information to the participant_info dict:
    - time_stamp
    - temperature_baseline
    - temperature_range
    """
    participant_info["time_stamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    participant_info["temperature_baseline"] = round(
        (participant_info["vas0"] + participant_info["vas70"]) / 2, 1
    )
    participant_info["temperature_range"] = round(
        participant_info["vas70"] - participant_info["vas0"], 1
    )
    return participant_info


def read_last_participant(file_path) -> dict:
    """
    Returns information about the last participant from the participants.xlsx file.
    """
    last_row = pd.read_excel(file_path).iloc[-1]
    participant_info = last_row.to_dict()
    participant_info["id"] = int(participant_info["id"])
    logger.info(
        f"Participant data from {participant_info['id']} ({participant_info['time_stamp']}) loaded."
    )
    # Check if the participant data is from today
    today = datetime.now().strftime("%Y-%m-%d")
    if today not in participant_info["time_stamp"]:
        logger.warning("Participant data is not from today.")
    return participant_info


if __name__ == "__main__":
    ask_for_participant_info()
    print("Done.")
