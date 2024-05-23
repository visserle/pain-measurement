# work in progress

#  TODO
# - add argument parsing for the main function
# - use parquet for internal processing instead of csv? https://kaveland.no/friends-dont-let-friends-export-to-csv.html
# - support csv as an addtional output format at the end


"""
This is the main script for processing data obtained from the iMotions software.

The script has three main steps:
1. Load data from csv files
2. Transform data
3. Save data to csv files
"""

import logging
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import polars as pl

from src.data.config_data import DataConfigBase
from src.data.config_data_imotions import IMOTIONS_LIST, iMotionsConfig
from src.data.config_data_interim import INTERIM_LIST
from src.data.config_data_raw import RAW_LIST, RawConfig
from src.data.config_participant import PARTICIPANT_LIST, ParticipantConfig
from src.log_config import configure_logging

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


@dataclass
class Dataset:
    """Dataclass for a single csv file"""

    name: str
    df: pl.DataFrame


@dataclass
class ParticipantData:
    """Dataclass for a single participant"""

    id: str
    datasets: dict[str, Dataset]

    def __call__(self, attr_name):
        return getattr(self, attr_name)

    def __getattr__(self, name):
        if name in self.datasets:
            return self.datasets[name].df
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __repr__(self):
        return f"Participant(id={self.id}, datasets={self.datasets.keys()})"


def load_dataset(
    participant_config: ParticipantConfig,
    data_config: DataConfigBase,
) -> Dataset:
    # Load and process data using Polars
    file_path, file_start_index = _get_file_path_and_start_index(
        data_config, participant_config
    )
    df = pl.read_csv(
        file_path,
        columns=data_config.load_columns,
        skip_rows=file_start_index,
        dtypes={
            load_column: pl.Float64 for load_column in data_config.load_columns
        },  # FIXME TODO dirty hack, add data schema instead
        # infer_schema_length=1000,
    )
    # For iMotions data we also want to rename some columns
    df = _rename_imotions_columns(df, data_config)

    logger.debug(
        "Dataset '%s' for participant %s loaded from %s",
        data_config.name,
        participant_config.id,
        file_path,
    )
    return Dataset(name=data_config.name, df=df)


def load_participant_datasets(
    participant_config: ParticipantConfig,
    data_configs: list[DataConfigBase],
) -> ParticipantData:
    datasets: dict[str, Dataset] = {}
    for data_config in data_configs:
        datasets[data_config.name] = load_dataset(participant_config, data_config)

    logger.info(
        f"Participant {participant_config.id} loaded with datasets: {datasets.keys()}"
    )
    return ParticipantData(id=participant_config.id, datasets=datasets)


def transform_dataset(
    dataset: Dataset,
    data_config: DataConfigBase,
) -> Dataset:
    """
    Transform a single dataset.
    Note that we just map a list of functions to the dataset.
    Could be made faster probably.

    From the old, basic code:

    def apply_func_participant(func, participant):
    #TODO: use map instead, e.g.:
    # dict(zip(a, map(f, a.values())))
    # dict(map(lambda item: (item[0], f(item[1])), my_dictionary.items()
    for data in participant.datasets:
        participant.datasets[data].dataset = func(participant.datasets[data].dataset)
    return participant

    """
    if data_config.transformations:
        for transformation in data_config.transformations:
            dataset.df = transformation(dataset.df)
            logger.debug(
                "Dataset '%s' transformed with %s",
                data_config.name,
                transformation.__repr__(),
            )
            # TODO: add **kwargs to transformations and pass them here using lambda functions? or better in the config?
    return dataset


def transform_participant_datasets(
    participant_config: ParticipantConfig,  # not used yet TODO
    participant_data: ParticipantData,
    data_configs: list[DataConfigBase],
) -> ParticipantData:
    """Transform all datasets for a single participant."""

    # Special transformation for iMotions data (e.g. create trials) first
    participant_data = _imotions_transformation(participant_data, data_configs)
    # Do the regular transformation(s) as defined in the config
    for data_config in data_configs:
        transform_dataset(participant_data.datasets[data_config.name], data_config)
    logger.info(f"Participant {participant_data.id} datasets successfully transformed")

    if isinstance(data_configs[0], DataConfigBase):
        pass  # TODO: merge datasets into one big dataset at the end

    # also make_labels at the end?
    return participant_data


def save_dataset(
    dataset: Dataset,
    participant_data: ParticipantData,
    data_config: DataConfigBase,
) -> None:
    """Save a single dataset to a csv file."""
    output_dir = data_config.save_dir / participant_data.id
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{participant_data.id}_{data_config.name}.csv"

    # Write the DataFrame to CSV
    # TODO: problems with saving timedelta format, but we can do without it for now
    dataset.df.write_csv(file_path)
    logger.debug(
        "Dataset '%s' for participant %s saved to %s",
        data_config.name,
        participant_data.id,
        file_path,
    )


def save_participant_datasets(
    participant_config: ParticipantConfig,
    participant_data: ParticipantData,
    data_configs: list[DataConfigBase],
) -> None:
    """Save all datasets for a single participant to csv files."""
    for data_config in data_configs:
        save_dataset(
            participant_data.datasets[data_config.name], participant_data, data_config
        )

    logger.info(
        f"Participant {participant_data.id} saved with datasets: "
        f"{participant_data.datasets.keys()}"
    )


def _get_file_path_and_start_index(
    data_config: DataConfigBase,
    participant_config: ParticipantConfig,
) -> tuple[Path, int]:
    """
    Get the file path and start index for loading data.

    Note that iMotions data is stored in a different format and has metadata we need to skip.
    """
    if not isinstance(data_config, iMotionsConfig):
        file_path = (
            data_config.load_dir
            / participant_config.id
            / f"{participant_config.id}_{data_config.name}.csv"
        )
        return file_path, 0
    else:
        file_path = (
            data_config.load_dir
            / participant_config.id
            / f"{data_config.name_imotions}.csv"
        )
        with open(file_path, "r") as file:
            lines = file.readlines(2**16)  # only read a few lines
        file_start_index = (
            next(i for i, line in enumerate(lines) if "#DATA" in line) + 1
        )
        return file_path, file_start_index


def _rename_imotions_columns(
    df: pl.DataFrame,
    data_config: iMotionsConfig,
) -> pl.DataFrame:
    if isinstance(data_config, iMotionsConfig):
        if data_config.rename_columns:
            df = df.rename(data_config.rename_columns)
    return df


def _imotions_transformation(
    participant_data: ParticipantData,
    data_configs: list[DataConfigBase],
) -> ParticipantData:
    """
    Special transformation for imotions data: we first want to add the participant id
    and merge trial information into each dataset (via Stimulus_Seed).

    Note that this cannot be done via a transformation from the config.
    """
    if isinstance(data_configs[0], iMotionsConfig):
        for data_config in IMOTIONS_LIST:
            # add the stimuli seed column to all datasets of the participant except for
            # the trial data which already has it
            if (
                "Stimulus_Seed"
                not in participant_data.datasets[data_config.name].df.columns
            ):
                participant_data.datasets[data_config.name].df = (
                    participant_data.datasets[data_config.name]
                    .df.join(
                        participant_data.trial, on="Timestamp", how="outer_coalesce"
                    )
                    .sort("Timestamp")
                )
            # add participant id to all datasets of the participant
            participant_data.datasets[data_config.name].df = participant_data.datasets[
                data_config.name
            ].df.with_columns(
                pl.lit(participant_data.id).alias("Participant").cast(pl.Int8)
            )
        logger.debug(
            "Participant %s datasets are now merged with trial information",
            participant_data.id,
        )
    return participant_data


def main():
    configure_logging(stream_level=logging.DEBUG)

    list_of_data_configs = [IMOTIONS_LIST, RAW_LIST, INTERIM_LIST]

    for data_configs in list_of_data_configs:
        for participant_config in PARTICIPANT_LIST[:2]:
            participant_data = load_participant_datasets(
                participant_config, data_configs
            )
            participant_data = transform_participant_datasets(
                participant_config, participant_data, data_configs
            )
            save_participant_datasets(
                participant_config, participant_data, data_configs
            )
    logger.info("All participants processed successfully")

    print(participant_data.eda)


if __name__ == "__main__":
    main()
