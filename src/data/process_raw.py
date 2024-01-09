import os
import pandas as pd
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List

from src.data.info_data import DataInfo
#from src.data.info_participants import ParticipantInfo TODO


@dataclass
class Data:
    name: str
    data: pd.DataFrame

@dataclass
class Participant:
    id: str
    signals: Dict[str, Data]

    def to_dataframe(self) -> pd.DataFrame:
        data_frames = [signal.data for signal in self.signals.values()]
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'),
            data_frames
        )
        return merged_df.sort_values(by=['Timestamp'])


def load_data(signal_path: str, data_info: DataInfo) -> Data:
    with open(signal_path, 'r') as file:
        lines = file.readlines(2**16)
    data_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line)

    data = pd.read_csv(
        signal_path, 
        skiprows=data_start_index + 1,
        usecols=lambda column: column in data_info.keep_columns,
    )
    return Data(name=data_info.name, data=data)

def load_participant_data(participant_id: str, base_path: str, data_infos: List[DataInfo]) -> Participant:
    participant_path = os.path.join(base_path, participant_id)
    signals: Dict[str, Data] = {}
    for data_info in data_infos:
        file_path = os.path.join(participant_path, data_info.path)
        if os.path.exists(file_path):
            signals[data_info.name] = load_data(file_path, data_info)
    return Participant(id=participant_id, signals=signals)

def main():
    pass
