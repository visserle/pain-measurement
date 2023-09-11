# work in progress

# TODO
# add a function to read the last participant to use calibration values in the next experiment
# add logging 
# documentation:
# this will be run from psychopy, so we need to make sure that the path is correct

import os
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd

try:
    import openpyxl
except ImportError:
    try:  
        import subprocess
        import sys
        process = subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"], check=False)
        if process.returncode != 0:
            raise Exception("pip installation failed")
    except Exception as exc:
        print(f"Failed to install and import 'openpyxl': {exc}")
        raise exc

from .logger import setup_logger
logger = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)

# Set the path to the Excel file
exp_dirs = "calibration", "mpad1", "mpad2"
X_DIR = Path.cwd()
if X_DIR.stem in exp_dirs: # run from psychopy runner
    FILE_DIR = X_DIR.parent / 'participants.xlsx'
else: # run from project root
    EXP_DIR = X_DIR / 'experiments'
    FILE_DIR = EXP_DIR / 'participants.xlsx'


def init_excel_file():
    headers = ['time_stamp', 'participant', 'age', 'gender', 'vas0', 'vas70', 'baseline_temp', 'temp_range']
    if not FILE_DIR.exists():
        df = pd.DataFrame(columns=headers)
        df.to_excel(FILE_DIR, index=False)

def add_participant(participant, age, gender, vas0, vas70):
    """
    Example for usage in psychopy:
    -------
    ```python
    from participants import add_participant
    add_participant(
        expInfo['participant'],
        expInfo['age'],
        expInfo['gender'],
        estimator_vas0.get_estimate(),
        estimator_vas70.get_estimate())
    ```
    """
    time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_data = pd.DataFrame([{
        'time_stamp': time_stamp,
        'participant': participant,
        'age': age,
        'gender': gender,
        'vas0': vas0,
        'vas70': vas70,
        'baseline_temp': round((vas0 + vas70) / 2, 1),
        'temp_range': vas70 - vas0
    }])
    df = pd.read_excel(FILE_DIR)
    if df.empty or df.isna().all().all():
        df = new_data
    else:
        # Check if the last participant is the same as the one you want to add
        last_participant = df.iloc[-1]['participant']
        if last_participant == participant:
            logger.critical(f"Participant {participant} already exists as the last entry.")
        df = pd.concat([df, new_data], ignore_index=True)
    
    df.to_excel(FILE_DIR, index=False)
    logger.info(f"Added participant {participant} to {FILE_DIR}")

def read_last_participant():
    """
    Returns important information about the last participant.

    Example for usage in psychopy:
    -------
    ```python
    from participants import read_last_participant
    participant_info = read_last_participant()
    ```
    """
    # Read the Excel file into a DataFrame
    df = pd.read_excel(FILE_DIR)

    # Get the last row of the DataFrame
    last_row = df.iloc[-1]

    # Extract relevant information
    participant_info = {
        'time_stamp': last_row['time_stamp'],
        'participant': last_row['participant'],
        'age': last_row['age'],
        'gender': last_row['gender'],
        'baseline_temp': last_row['baseline_temp'],
        'temp_range': last_row['temp_range']
    }

    logger.info(f"Participant data from {participant_info['participant']}, {participant_info['time_stamp']}, loaded.")

    return participant_info

# does not work with logging
# def main():
#     init_excel_file()
#     add_participant('John', 22, 'm', 3.5, 4.0)
#     add_participant('Jane', 25, 'f', 2.5, 3.8)
#     read_last_participant()

# if __name__ == '__main__':
#     main()