import os
import pandas as pd
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List
import logging

from src.data.config_data import DataConfigBase
from src.data.config_imotions import iMotionsConfig, IMOTIONS_LIST
from src.data.config_participants import PARTICIPANT_LIST

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
    
    def __getattr__(self, name):
        if name in self.datasets.keys():
           return self.datasets[name].dataset
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
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


def load_dataset(
        data_path: str, 
        data_config: DataConfigBase
        ) -> Data:
    
    if isinstance(data_config, iMotionsConfig):
        # Find start index in external data files
        with open(data_path, 'r') as file:
            lines = file.readlines(2**16) # only read a few lines
        data_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line)
    # Load and process data
    dataset = pd.read_csv(
        data_path, 
        skiprows=data_start_index + 1 if isinstance(data_config, iMotionsConfig) else 0,
        usecols=lambda column: column in data_config.keep_columns,
    )
    dataset.rename(columns=data_config.rename_columns, inplace=True) if data_config.rename_columns else None
    return Data(name=data_config.name, dataset=dataset)

def load_participant_datasets(
        participant_id: str, 
        base_path: str, 
        data_configs: List[iMotionsConfig]
        ) -> Participant:
    participant_path = os.path.join(base_path, participant_id)
    datasets: Dict[str, Data] = {}
    for data_config in data_configs:
        file_path = os.path.join(participant_path, data_config.path)
        if os.path.exists(file_path):
            datasets[data_config.name] = load_dataset(file_path, data_config)
    return Participant(id=participant_id, datasets=datasets)


def main():
    DATA_DIR = Path("./data")
    IMOTIONS_DIR = DATA_DIR / "imotions"
    RAW_DIR = DATA_DIR / "raw"
    
    for particpant in PARTICIPANT_LIST:
        participant_data = load_participant_datasets(
            particpant.id, IMOTIONS_DIR, IMOTIONS_LIST
            )
        participant_data.save_individual_datasets(RAW_DIR / particpant.id)

if __name__ == "__main__":
    main()
