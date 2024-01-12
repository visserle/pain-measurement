# work in progress
# TODO:
# - maybe make transformation faster by using map or list comprehension?

import os
import pandas as pd
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List
import logging

from src.data.config_data import DataConfigBase
from src.data.config_data_imotions import iMotionsConfig, IMOTIONS_LIST
from src.data.config_data_raw import RawConfig, RAW_LIST
from src.data.config_data_trial import TrialConfig, TRIAL_LIST
from src.data.config_participant import ParticipantConfig, PARTICIPANT_LIST

from src.log_config import configure_logging

configure_logging()


@dataclass
class Data:
    """Dataclass for a single csv files"""
    name: str
    dataset: pd.DataFrame

@dataclass
class Participant:
    """Dataclass for a single participant"""
    id: str
    datasets: Dict[str, Data]
    
    def __getattr__(self, name):
        if name in self.datasets:
           return self.datasets[name].dataset
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

def load_dataset(
        participant_config: ParticipantConfig,
        data_config: DataConfigBase
        ) -> Data:

    # iMotions data needs special treatment because it is external
    if not isinstance(data_config, iMotionsConfig):
        data_path = data_config.load_dir / participant_config.id / f"{participant_config.id}_{data_config.name}.csv"
        data_start_index = None
    elif isinstance(data_config, iMotionsConfig):
        data_path = data_config.load_dir / participant_config.id / f"{data_config.name_imotions}.csv"
        # iMotions data files have metadata we need to skip
        with open(data_path, 'r') as file:
            lines = file.readlines(2**16) # only read a few lines
        data_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
    
    # Load and process data
    dataset = pd.read_csv(
        data_path, 
        skiprows=data_start_index,
        usecols=lambda column: column in data_config.load_columns,
    )
    if isinstance(data_config, iMotionsConfig):
        dataset.rename(columns=data_config.rename_columns, inplace=True) if data_config.rename_columns else None
    return Data(name=data_config.name, dataset=dataset)

def load_participant_datasets(
        participant_config: ParticipantConfig, 
        data_configs: List[DataConfigBase]
        ) -> Participant:

    datasets: Dict[str, Data] = {}
    for data_config in data_configs:
        datasets[data_config.name] = load_dataset(participant_config, data_config)
    return Participant(id=participant_config.id, datasets=datasets)

def transform_dataset(
        data: Data,
        data_config: DataConfigBase
        ) -> Data:
    """Transform a single dataset."""
    if data_config.transformations:
        for transformation in data_config.transformations:
            data.dataset = transformation(data.dataset)
    return data

def transform_participant_datasets(
        participant: Participant,
        data_configs: List[DataConfigBase]
        ) -> Participant:
    """Transform all datasets for a single participant."""
    for data_config in data_configs:
        transform_dataset(participant.datasets[data_config.name], data_config)
    return participant


def save_dataset(
        data: Data,
        participant_config: ParticipantConfig,
        data_config: DataConfigBase
        ) -> None:
    """Save a single dataset to a csv file."""
    output_dir = data_config.save_dir / participant_config.id
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{participant_config.id}_{data_config.name}.csv"
    data.dataset.to_csv(file_path, index=True)
    logging.info(f"Dataset '{data_config.name}' for participant {participant_config.id} saved to {file_path}")

def save_participant_datasets(
        participant: Participant,
        data_configs: List[DataConfigBase]
        ) -> None:
    """Save all datasets for a single participant to csv files."""
    for data_config in data_configs:
        save_dataset(participant.datasets[data_config.name], participant, data_config)


def main():
    list_of_data_configs = [
        IMOTIONS_LIST,
        RAW_LIST,
        TRIAL_LIST,
    ]
    
    for data_configs in list_of_data_configs:
        for particpant in PARTICIPANT_LIST:
            participant_data = load_participant_datasets(
                particpant, data_configs)
            participant_data = transform_participant_datasets(
                participant_data, data_configs)
            save_participant_datasets(participant_data, data_configs)

    data = load_participant_datasets(PARTICIPANT_LIST[0], TRIAL_LIST)
    print(data.trial.head())
    
if __name__ == "__main__":
    main()


# def merge_participant_datasets(self) -> pd.DataFrame:
#     data_frames = [data.dataset for data in self.datasets.values()]
#     # Use reduce to merge all DataFrames on 'Timestamp'
#     merged_df = reduce(
#         # pd.concat would lead to duplicate timestamps
#         lambda left, right: pd.merge(left, right, on='Timestamp', how='outer'),
#         data_frames
#     )
#     merged_df.sort_values(by=['Timestamp'], inplace=True)
#     logging.info(f"Dataframe shape: {merged_df.shape}")
#     return merged_df

