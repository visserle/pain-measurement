# NOTE, TODO: for unchanged data representations simply import the config from a previous stage
"""
For processing interim (already cleaned) data.
"""

from dataclasses import dataclass, field
from pathlib import Path

from src.data.config_data import DataConfigBase
from src.features.eda import process_eda
from src.features.pupillometry import process_pupillometry
from src.features.transformations import interpolate  # resample

LOAD_FROM = Path("data/interim")
SAVE_TO = Path("data/preprocessed")


@dataclass
class InterimConfig(DataConfigBase):
    name: str
    load_columns: list[str]
    transformations: list[callable] = field(default_factory=list)
    sampling_rate: int | None = None
    notes: str | None = None

    def __post_init__(self):
        self.load_dir = LOAD_FROM
        self.save_dir = SAVE_TO
        self.transformations = (
            [interpolate]  # TODO FIXME Why is this here? Do we really need this?
            if not self.transformations
            else [interpolate] + self.transformations
        )
        self.load_columns = ["Participant", "Trial", "Timestamp"] + self.load_columns


STIMULUS = InterimConfig(
    name="stimulus",
    load_columns=["Temperature", "Rating", "Stimulus_Seed", "Skin_Area"],
)

EEG = InterimConfig(
    name="eeg",
    load_columns=[
        "EEG_RAW_Ch1",
        "EEG_RAW_Ch2",
        "EEG_RAW_Ch3",
        "EEG_RAW_Ch4",
        "EEG_RAW_Ch5",
        "EEG_RAW_Ch6",
        "EEG_RAW_Ch7",
        "EEG_RAW_Ch8",
    ],
)

EDA = InterimConfig(
    name="eda",
    load_columns=["EDA_RAW", "EDA_Phasic", "EDA_Tonic"],
)

PPG = InterimConfig(
    name="ppg",
    load_columns=[
        "PPG_RAW",
        "PPG_HeartRate",
        "PPG_IBI",
    ],
)

PUPILLOMETRY = InterimConfig(
    name="pupillometry",
    load_columns=[
        "Pupillometry_L",
        "Pupillometry_R",
    ],
    sampling_rate=60,
)

AFFECTIVA = InterimConfig(
    name="affectiva",
    load_columns=[
        "Anger",
        "Contempt",
        "Disgust",
        "Fear",
        "Joy",
        "Sadness",
        "Surprise",
        "Engagement",
        "Valence",
        "Sentimentality",
        "Confusion",
        "Neutral",
        "Attention",
        "Brow Furrow",
        "Brow Raise",
        "Cheek Raise",
        "Chin Raise",
        "Dimpler",
        "Eye Closure",
        "Eye Widen",
        "Inner Brow Raise",
        "Jaw Drop",
        "Lip Corner Depressor",
        "Lip Press",
        "Lip Pucker",
        "Lip Stretch",
        "Lip Suck",
        "Lid Tighten",
        "Mouth Open",
        "Nose Wrinkle",
        "Smile",
        "Smirk",
        "Upper Lip Raise",
        "Blink",
        "BlinkRate",
        "Pitch",
        "Yaw",
        "Roll",
        "Interocular Distance",
    ],
    notes="Affectiva data is not sampled at a constant rate and the only orignal data that can contain NaNs.",
)


INTERIM_LIST = [STIMULUS, EEG, EDA, PPG, PUPILLOMETRY, AFFECTIVA]
INTERIM_DICT = {config.name: config for config in INTERIM_LIST}
