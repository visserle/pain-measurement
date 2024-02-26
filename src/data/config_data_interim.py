"""
For processing interim (already cleaned) data.
"""
from dataclasses import dataclass, field
from pathlib import Path

from src.data.config_data import DataConfigBase
from src.features.pupillometry import process_pupillometry
from src.features.transformations import interpolate  # resample


@dataclass
class InterimConfig(DataConfigBase):
    name: str
    load_columns: list[str]
    transformations: list[callable] = field(default_factory=list)
    sampling_rate: int | None = None
    notes: str | None = None

    def __post_init__(self):
        self.load_dir = Path("data/raw")
        self.save_dir = Path("data/interim")
        self.transformations = (
            [interpolate] if not self.transformations else [] + self.transformations
        )


HEAT = InterimConfig(
    name="heat",
    load_columns=["Trial", "Timestamp", "Temperature", "Rating"],  #
)

EEG = InterimConfig(
    name="eeg",
    load_columns=[
        "Trial",
        "Timestamp",
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
    load_columns=["Trial", "Timestamp", "EDA_RAW"],
)

PPG = InterimConfig(
    name="ppg",
    load_columns=[
        "Trial",
        "Timestamp",
        "PPG_RAW",
        "PPG_HeartRate",
        "PPG_IBI",
    ],
)

PUPILLOMETRY = InterimConfig(
    name="pupillometry",
    load_columns=[
        "Trial",
        "Timestamp",
        "Pupillometry_L",
        "Pupillometry_R",
    ],
    transformations=[process_pupillometry],
    sampling_rate=60,
)

AFFECTIVA = InterimConfig(
    name="affectiva",
    load_columns=[
        "Trial",
        "Timestamp",
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


INTERIM_LIST = [HEAT, EEG, EDA, PPG, PUPILLOMETRY, AFFECTIVA]
INTERIM_DICT = {config.name: config for config in INTERIM_LIST}
