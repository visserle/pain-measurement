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
    dataset: pd.DataFrame # single imotions csv file

@dataclass
class Participant:
    id: str
    datasets: Dict[str, Data]

    def to_dataframe(self) -> pd.DataFrame:
        data_frames = [data.dataset for data in self.datasets.values()]
        # Use reduce to merge all DataFrames on 'Timestamp'
        merged_df = reduce(
            # Use reduce to merge all DataFrames on 'Timestamp'
            lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'),
            data_frames
        )
        return merged_df.sort_values(by=['Timestamp'])


def load_data(data_path: str, data_info: DataInfo) -> Data:
    # Find start index for data
    with open(data_path, 'r') as file:
        lines = file.readlines(2**16) # only read a few lines
    data_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line)
    
    # Load and process data
    dataset = pd.read_csv(
        data_path, 
        skiprows=data_start_index + 1,
        usecols=lambda column: column in data_info.keep_columns,
    )
    return Data(name=data_info.name, dataset=dataset)

def load_participant_data(participant_id: str, base_path: str, data_infos: List[DataInfo]) -> Participant:
    participant_path = os.path.join(base_path, participant_id)
    datasets: Dict[str, Data] = {}
    for data_info in data_infos:
        file_path = os.path.join(participant_path, data_info.path)
        if os.path.exists(file_path):
            datasets[data_info.name] = load_data(file_path, data_info)
    return Participant(id=participant_id, datasets=datasets)

def main():
    pass
