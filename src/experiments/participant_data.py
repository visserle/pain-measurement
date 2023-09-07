# work in progress

# TODO
# add a function to read the last participant to use calibration values in the next experiment
# add logging 

import os
from datetime import datetime
from pathlib import Path
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

PROJECT_DIR = Path.cwd()
EXP_DIR = PROJECT_DIR / 'experiments'
FILE_DIR = EXP_DIR / 'participants.xlsx'

# Ensure the directory exists
EXP_DIR.mkdir(parents=True, exist_ok=True)

def init_excel_file():
    headers = ['time_stamp', 'participant', 'age', 'gender', 'vas_0', 'vas_70']
    if not FILE_DIR.exists():
        df = pd.DataFrame(columns=headers)
        df.to_excel(FILE_DIR, index=False)

def add_participant(participant, age, gender, vas_0, vas_70):
    """
    Example for usage in psychopy:
    -------
    ```python
    from participants import add_participant
    add_participant(
        expInfo['participant'],
        expInfo['age'],
        expInfo['gender']
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
        'vas_0': vas_0,
        'vas_70': vas_70
    }])
    df = pd.read_excel(FILE_DIR)
    if df.empty or df.isna().all().all():
        df = new_data
    else:
        df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(FILE_DIR, index=False)


def read_last_participant():
    df = pd.read_excel(FILE_DIR)
    last_row = df.tail(1)
    for index, row in last_row.iterrows():
        print(row['participant'], row['age'], row['time_stamp'], row['vas_0'], row['vas_70'])

def main():
    init_excel_file()
    add_participant('John', 22, 'm', 3.5, 4.0)
    add_participant('Jane', 25, 'f', 2.5, 3.8)
    read_last_participant()

if __name__ == '__main__':
    main()