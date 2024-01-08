import os
import pandas as pd
import numpy as np
from functools import reduce

from dataclasses import dataclass
from typing import List, Optional, Dict
from src.data.info_data import trial, rating, temperature, eda, ecg, eeg, pupillometry, affectiva, system
#from src.data.info_participants import TODO

# Define DataSignal class
@dataclass
class DataSignal:
    name: str
    data: pd.DataFrame

# Define Participant class
@dataclass
class Participant:
    id: str
    signals: dict[str, DataSignal]

    def to_dataframe(self):
        data_frames = [signal.data for signal in self.signals.values()]
        # Use reduce to merge all DataFrames on 'Timestamp'
        merged_df = reduce(
            lambda left, right:
                # Do not use pd.concat() because it will add duplicate time stamps
                pd.merge(left, right, on='Timestamp', how='outer'),
                data_frames
                )
        return merged_df.sort_values(by=['Timestamp'])


# Function to load signal data
def load_signal_data(signal_path, data_info):
    # Find start index for data
    with open(signal_path, 'r') as file:
        lines = file.readlines(2**16) # only read a few lines
    data_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line)

    # Load and process data
    data = pd.read_csv(
        signal_path, 
        skiprows=data_start_index + 1,
        usecols=lambda column: column in data_info.keep_columns,
    )
    return DataSignal(name=data_info.name, data=data)

# Function to load participant data
def load_participant_data(participant_id, base_path, data_infos):
    participant_path = os.path.join(base_path, participant_id)
    signals = {}
    for data_info in data_infos:
        file_path = os.path.join(participant_path, data_info.path)
        if os.path.exists(file_path):
            signals[data_info.name] = load_signal_data(file_path, data_info)
    return Participant(id=participant_id, signals=signals)
