import os
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
    """Dataclass for a single imotions csv files"""
    name: str
    dataset: pd.DataFrame

@dataclass
class Participant:
    id: str
    datasets: Dict[str, Data]
    
    def save_individual_datasets(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        for data_name, data in self.datasets.items():
            file_path = os.path.join(output_dir, f"{self.id}_{data_name}.csv")
            data.dataset.to_csv(file_path, index=False)
            logging.info(f"Dataset '{data_name}' for participant {self.id} saved to {file_path}")

    def save_whole_dataset(self, output_dir: str, filename: str = None) -> None:
        if not filename:
            filename = f"{self.id}.csv"
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        merged_df = self.merge_datasets()
        merged_df.to_csv(output_path, index=False)
        logging.info(f"Dataframe for participant {self.id} saved to {output_path}")
        
    def merge_datasets(self) -> pd.DataFrame:
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


def load_dataset(data_path: str, data_info: ExternalInfo) -> Data:
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

def load_participant_datasets(participant_id: str, base_path: str, data_infos: List[ExternalInfo]) -> Participant:
    participant_path = os.path.join(base_path, participant_id)
    datasets: Dict[str, Data] = {}
    for data_info in data_infos:
        file_path = os.path.join(participant_path, data_info.path)
        if os.path.exists(file_path):
            datasets[data_info.name] = load_dataset(file_path, data_info)
    return Participant(id=participant_id, datasets=datasets)


def main():
    DATA_DIR = Path("./data")
    EXTERNAL_DIR = DATA_DIR / "external"
    RAW_DIR = DATA_DIR / "raw"
    
    
    for particpant in PARTICIPANT_LIST:
        participant_data = load_participant_datasets(
            particpant.id, EXTERNAL_DIR, EXTERNAL_LIST
            )
        participant_data.save_individual_datasets(RAW_DIR / particpant.id)
        #participant_data.save_whole_dataset(RAW_DIR / f"{particpant.id}.csv")

if __name__ == "__main__":
    main()
    