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

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

# Set the path to the Excel file
exp_dirs = "calibration", "mpad1", "mpad2"
X_DIR = Path.cwd()
if X_DIR.stem in exp_dirs: # when running from psychopy runner
    FILE_DIR = X_DIR.parent / 'participants.xlsx'
else: # when running from project root
    EXP_DIR = X_DIR / 'experiments'
    FILE_DIR = EXP_DIR / 'participants.xlsx'


def init_excel_file():
    headers = ['time_stamp', 'participant', 'age', 'gender', 'vas0', 'vas70', 'baseline_temp', 'temp_range']
    if not FILE_DIR.exists():
        df = pd.DataFrame(columns=headers)
        df.to_excel(FILE_DIR, index=False)

def add_participant(participant, age, gender, vas0, vas70):
    """
    Adds a participant to the participants.xlsx file.

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
        'age': int(age),
        'gender': str(gender),
        'vas0': int(vas0),
        'vas70': int(vas70),
        'baseline_temp': round((vas0 + vas70) / 2, 1),
        'temp_range': round(vas70 - vas0, 1)
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
    Returns information about the last participant from the participants.xlsx file.

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

    # Check if the participant data is from today
    today = datetime.now().strftime('%Y-%m-%d')
    if today not in participant_info['time_stamp']:
        logger.warning(f"Participant data from {participant_info['participant']} ({participant_info['time_stamp']}) is not from today.")

    logger.info(f"Participant data from {participant_info['participant']} ({participant_info['time_stamp']}) loaded.")

    return participant_info