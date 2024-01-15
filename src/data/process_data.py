# work in progress
# TODO:
# - maybe make transformation faster by using map or list comprehension?
# - switch to polars?

"""
This is the main script for processing data obtained from the iMotions software.

The script has three main steps:
1. Load data from csv files
2. Transform data
3. Save data to csv files


"""

import os
from pathlib import Path
from functools import reduce
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

import pandas as pd
import polars as pl

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
    dataset: pl.DataFrame

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

    file_path = data_config.load_dir / participant_config.id / f"{participant_config.id}_{data_config.name}.csv"
    
    # iMotions data are stored in a different format and have metadata we need to skip
    if isinstance(data_config, iMotionsConfig):
        file_path = data_config.load_dir / participant_config.id / f"{data_config.name_imotions}.csv"
        with open(file_path, 'r') as file:
            lines = file.readlines(2**16) # only read a few lines
        file_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1

    # Load and process data using Polars
    dataset = pl.read_csv(
        file_path, 
        columns=data_config.load_columns,
        skip_rows=0 if not file_start_index else file_start_index,
        infer_schema_length=1000,
    )
    
    # For iMotions data we also want to rename some columns
    if isinstance(data_config, iMotionsConfig):
        if data_config.rename_columns:
            dataset = dataset.rename(data_config.rename_columns)

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
    """
    Transform a single dataset.
    Note that we just map a list of functions to the dataset. Could be made faster probably.
    
    From the old, basic code:
    
    def apply_func_participant(participant, func):
    #TODO: use map instead, e.g.:
    # dict(zip(a, map(f, a.values())))
    # dict(map(lambda item: (item[0], f(item[1])), my_dictionary.items()
    for data in participant.datasets:
        participant.datasets[data].dataset = func(participant.datasets[data].dataset)
    return participant
    
    """
    if data_config.transformations:
        for transformation in data_config.transformations:
            data.dataset = transformation(data.dataset)
    return data

def transform_participant_datasets(
        participant_data: Participant,
        data_configs: List[DataConfigBase]
        ) -> Participant:
    """Transform all datasets for a single participant."""
    # Special case for imotions data: we need to add the stimuli seed column
    if isinstance(data_configs[0], iMotionsConfig):
        # datasets are missing the trial information (via Stimuli_Seed) and need to be merged with the trial data first
        for data_config in IMOTIONS_LIST:
            # add the stimuli seed column to all datasets of the participant except for the trial data which already has it
            if "Stimuli_Seed" not in participant_data.datasets[data_config.name].dataset.columns:
                participant_data.datasets[data_config.name].dataset = participant_data.datasets[data_config.name].dataset.join(
                    participant_data.trial,
                    on='Timestamp',
                    how='outer_coalesce',
                ).sort('Timestamp')
            assert participant_data.datasets[data_config.name].dataset['Timestamp'].is_sorted(descending=False)

    # Do the regular transformation(s) as defined in the config
    for data_config in data_configs:
        transform_dataset(participant_data.datasets[data_config.name], data_config)
    return participant_data


def save_dataset(
        data: Data,
        participant_config: ParticipantConfig,
        data_config: DataConfigBase
        ) -> None:
    """Save a single dataset to a csv file."""
    output_dir = data_config.save_dir / participant_config.id
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{participant_config.id}_{data_config.name}.csv"

    # Write the DataFrame to CSV
    # TODO: problems with saving timedelta format, but we can do without it for now
    data.dataset.write_csv(file_path)
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
        #RAW_LIST,
        #TRIAL_LIST,
    ]

    for data_configs in list_of_data_configs:
        for participant_config in PARTICIPANT_LIST:
            participant_data = load_participant_datasets(
                participant_config, data_configs)
            participant_data = transform_participant_datasets(
                participant_data, data_configs)
            save_participant_datasets(participant_data, data_configs)

    # data = load_participant_datasets(PARTICIPANT_LIST[0], IMOTIONS_LIST)
    # print(data.eeg.head())
    
    # data_2 = transform_participant_datasets(data, IMOTIONS_LIST)
    # print(data_2.eeg.head())

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
