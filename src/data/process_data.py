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

def load_dataset_old(
        participant_config: ParticipantConfig,
        data_config: DataConfigBase,
        ) -> Data:

    file_path = data_config.load_dir / participant_config.id / f"{participant_config.id}_{data_config.name}.csv"
    # iMotions files are stored in a different format and have metadata we need to skip
    if isinstance(data_config, iMotionsConfig):
        file_path = data_config.load_dir / participant_config.id / f"{data_config.name_imotions}.csv"
        with open(file_path, 'r') as file:
            lines = file.readlines(2**16) # only read a few lines
        file_start_index = next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
     
    # Load and process data
    dataset = pd.read_csv(
        file_path,
        skiprows=None if not file_start_index else file_start_index,
        usecols=lambda column: column in data_config.load_columns,
    )
    # For iMotions data we also want to rename some columns
    if isinstance(data_config, iMotionsConfig):
        dataset.rename(columns=data_config.rename_columns, inplace=True) if data_config.rename_columns else None
    return Data(name=data_config.name, dataset=dataset)

def load_participant_datasets_old(
        participant_config: ParticipantConfig, 
        data_configs: List[DataConfigBase]
        ) -> Participant:

    datasets: Dict[str, Data] = {}
    for data_config in data_configs:
        datasets[data_config.name] = load_dataset_old(participant_config, data_config)
    return Participant(id=participant_config.id, datasets=datasets)




def load_dataset(
        participant_config: ParticipantConfig,
        data_config: DataConfigBase
        ) -> Data:

    file_path = data_config.load_dir / participant_config.id / f"{participant_config.id}_{data_config.name}.csv"
    file_start_index = 0
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
        skip_rows=file_start_index,
        infer_schema_length=1000,
    )
    
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
        participant: Participant,
        data_configs: List[DataConfigBase]
        ) -> Participant:
    """Transform all datasets for a single participant."""
    
    # RawConfig datasets are missing the trial information (via Stimuli_Seed) and need to be merged with the trial data first
    if isinstance(data_configs[0], RawConfig):
        pd.options.mode.chained_assignment = None  # default='warn'
        # add the stimuli seed column to all raw datasets of the participant
        for data in participant.datasets.keys(): # FIXME TODO this is not in sync with the other loops, we have to base everything on data_configs !!!!
            if "Stimuli_Seed" not in participant.datasets[data].dataset.columns:
                participant.datasets[data].dataset = pd.merge(
                    participant.datasets[data].dataset, 
                    participant.trial, 
                    on='Timestamp', how='outer')
                participant.datasets[data].dataset.sort_values(by=['Timestamp'], inplace=True)
                participant.datasets[data].dataset.reset_index(drop=True, inplace=True)
        pd.options.mode.chained_assignment = 'warn'

    # Do the actual transformation(s)
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
    data.dataset.to_csv(
        file_path, 
        index=True)
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
        # RAW_LIST,
        # TRIAL_LIST,
    ]

    # for data_configs in list_of_data_configs:
    #     for particpant in PARTICIPANT_LIST:
    #         participant_data = load_participant_datasets(
    #             particpant, data_configs)
    #         participant_data = transform_participant_datasets(
    #             participant_data, data_configs)
    #         save_participant_datasets(participant_data, data_configs)

    data = load_participant_datasets(PARTICIPANT_LIST[0], IMOTIONS_LIST)
    print(data.eeg.head())

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

########################################
########################################
########### POLARS VERSION #############
#IMPOPRTANT: DO NOT DELETE PANDAS 
# VERSIONS ABOVE, WE CAN USE CUDF 
# INSTEAD OF POLARS
########################################
########################################

# to create a time delta column, this is the most efficient way:

# # Create a range of integers (0 to 999999) representing each millisecond
# milliseconds_range = pl.Series(range(1000000))

# # Cast the series to timedelta with milliseconds as the unit
# timedelta_range = milliseconds_range.cast(pl.Duration(time_unit='ms'))






# make a polars version of the functions above
# currently, we are running into problems with the index and timedelta format

# def save_dataset_pl(
#         data: Data,
#         participant_config: ParticipantConfig,
#         data_config: DataConfigBase
#         ) -> None:
#     """Save a single dataset to a csv file."""
#     output_dir = data_config.save_dir / participant_config.id
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     file_path = output_dir / f"{participant_config.id}_{data_config.name}.csv"

#     # Convert Pandas DataFrame to Polars DataFrame before saving
#     pl_df = pl.from_pandas(data.dataset)
#     # Up to now, we set a index in the pandas transfomations but polars
#     # does not treat the index as a "real data". Therefore, we need to add it
#     # manually as a new column.
#     # TODO: find a better solution
#     #pl_df = pl_df.insert_at_idx(0, pl.Series("Time", range(1, len(pl_df) + 1)))

#     # Write the DataFrame to CSV
#     pl_df.write_csv(file_path)
#     logging.info(f"Dataset '{data_config.name}' for participant {participant_config.id} saved to {file_path}")

# def save_participant_datasets_pl(
#         participant: Participant,
#         data_configs: List[DataConfigBase]
#         ) -> None:
#     """Save all datasets for a single participant to csv files."""
#     for data_config in data_configs:
#         save_dataset_pl(participant.datasets[data_config.name], participant, data_config)



