import os
import sys
import pandas as pd
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List
import logging

from src.data.info_external import ExternalInfo, EXTERNAL_LIST
from src.data.info_participants import PARTICIPANT_LIST
from src.log_config import configure_logging

configure_logging()

@dataclass
class Data:
    name: str
    dataset: pd.DataFrame # single imotions csv files

@dataclass
class Participant:
    id: str
    datasets: Dict[str, Data]

    def to_dataframe(self) -> pd.DataFrame:
        data_frames = [data.dataset for data in self.datasets.values()]
        # Use reduce to merge all DataFrames on 'Timestamp'
        merged_df = reduce(
            # pd.concat would lead to duplicate timestamps
            lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'),
            data_frames
        )
        merged_df.sort_values(by=['Timestamp'], inplace=True)
        logging.info(f"Dataframe shape: {merged_df.shape}")
        return merged_df
    
    def save_dataframe(self, path: str) -> None:
        logging.info(f"Saved participant {self.id} to {path}")
        self.to_dataframe().to_csv(path, index=False, header=True)


def load_data(data_path: str, data_info: ExternalInfo) -> Data:
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
    dataset.rename(columns=data_info.rename_columns, inplace=True) if data_info.rename_columns else None
    return Data(name=data_info.name, dataset=dataset)

def load_participant_data(participant_id: str, base_path: str, data_infos: List[ExternalInfo]) -> Participant:
    participant_path = os.path.join(base_path, participant_id)
    datasets: Dict[str, Data] = {}
    for data_info in data_infos:
        file_path = os.path.join(participant_path, data_info.path)
        if os.path.exists(file_path):
            datasets[data_info.name] = load_data(file_path, data_info)
    return Participant(id=participant_id, datasets=datasets)

def main():
    DATA_DIR = Path("./data")
    EXTERNAL_DIR = DATA_DIR / "external"
    RAW_DIR = DATA_DIR / "raw"
  
    for participant in PARTICIPANT_LIST:
        load_participant_data(
            participant.id, EXTERNAL_DIR, EXTERNAL_LIST
            ).save_dataframe(RAW_DIR / f"{participant.id}.csv")

if __name__ == "__main__":
    main()
    