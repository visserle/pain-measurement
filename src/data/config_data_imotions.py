"""Contains information about the iMotions data files (= external data)."""

from dataclasses import dataclass, field
from pathlib import Path

from src.data.config_data import DataConfigBase
from src.features.transformations import (
    create_trials,
    interpolate_to_marker_timestamps,
)

# - add schemas for csv (for now dtype keyword only)
# -> no need for infer_schema_length then (quite costly)

LOAD_FROM = Path("data/imotions")
SAVE_TO = Path("data/raw")


@dataclass
class iMotionsConfig(DataConfigBase):
    name: str
    name_imotions: str
    load_columns: list[str]
    dtypes: dict[str, str] = field(default_factory=dict)  # TODO maybe full schema?
    rename_columns: dict[str, str] = field(default_factory=dict)
    transformations: list[callable] = field(default_factory=list)
    sample_rate: int | None = None
    notes: str | None = None

    def __post_init__(self):
        self.load_dir = LOAD_FROM
        self.save_dir = SAVE_TO
        self.transformations = [
            create_trials,
            interpolate_to_marker_timestamps,
        ]


TRIAL = iMotionsConfig(
    name="trial",
    name_imotions="ExternalMarker_ET_EventAPI_ExternDevice",
    load_columns=["Timestamp", "MarkerDescription"],
    rename_columns={
        "MarkerDescription": "Stimulus_Seed",
    },
    notes="MarkerDescription contains the stimulus seed and is originally send once at the start and end of each trial.",
)

STIMULUS = iMotionsConfig(
    name="stimulus",
    name_imotions="CustomCurves_CustomCurves@1_ET_EventAPI_ExternDevice",
    load_columns=["Timestamp", "Temperature", "Rating"],
)

EDA = iMotionsConfig(
    name="eda",
    name_imotions="Shimmer3_EDA_&_PPG_Shimmer3_EDA_&_PPG_ET_Shimmer_ShimmerDevice",
    # name_imotions="Shimmer3_GSR+_&_EDA_(D200)_Shimmer3_GSR+_&_EDA_(D200)_ET_Shimmer_ShimmerDevice",
    load_columns=[
        "RowNumber",
        "SampleNumber",
        "Timestamp",
        "GSR Conductance CAL",
        "VSenseBatt CAL",
        "Packet reception rate RAW",
    ],
    rename_columns={
        "GSR Conductance CAL": "EDA_RAW",
        "VSenseBatt CAL": "EDA_d_Battery",
        "Packet reception rate RAW": "EDA_d_PacketReceptionRate",
    },
)

PPG = iMotionsConfig(
    name="ppg",
    # name_imotions="Shimmer3_GSR+_&_EDA_(D200)_Shimmer3_GSR+_&_EDA_(D200)_ET_Shimmer_ShimmerDevice",
    name_imotions="Shimmer3_EDA_&_PPG_Shimmer3_EDA_&_PPG_ET_Shimmer_ShimmerDevice",
    load_columns=[
        "RowNumber",
        "SampleNumber",
        "Timestamp",
        "Internal ADC A13 PPG CAL",
        "Heart Rate PPG ALG",
        "IBI PPG ALG",
        "VSenseBatt CAL",
        "Packet reception rate RAW",
    ],
    rename_columns={
        "Internal ADC A13 PPG CAL": "PPG_RAW",
        "Heart Rate PPG ALG": "PPG_HeartRate",
        "IBI PPG ALG": "PPG_IBI",
        "VSenseBatt CAL": "PPG_d_Battery",
        "Packet reception rate RAW": "PPG_d_PacketReceptionRate",
    },
)

EEG = iMotionsConfig(
    name="eeg",
    name_imotions="EEG_Enobio-USB_ENOBIO-8-NE-Device_(COM3)_ET_Lsl_LslSensor",
    load_columns=["Timestamp", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"],
    rename_columns={
        "Ch1": "EEG_RAW_Ch1",
        "Ch2": "EEG_RAW_Ch2",
        "Ch3": "EEG_RAW_Ch3",
        "Ch4": "EEG_RAW_Ch4",
        "Ch5": "EEG_RAW_Ch5",
        "Ch6": "EEG_RAW_Ch6",
        "Ch7": "EEG_RAW_Ch7",
        "Ch8": "EEG_RAW_Ch8",
    },
    notes="has no SampleNumber column",
)

# # Wifi-EEG
# EEG = iMotionsConfig(
#     name="eeg",
#     name_imotions="EEG_Enobio-Wifi_ENOBIO-8-NE-Wifi(00_07_80_0D_82_F7)_ET_Lsl_LslSensor",
#     load_columns=["Timestamp", "Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"],
#     rename_columns={
#         "Ch1": "EEG_RAW_Ch1",
#         "Ch2": "EEG_RAW_Ch2",
#         "Ch3": "EEG_RAW_Ch3",
#         "Ch4": "EEG_RAW_Ch4",
#         "Ch5": "EEG_RAW_Ch5",
#         "Ch6": "EEG_RAW_Ch6",
#         "Ch7": "EEG_RAW_Ch7",
#         "Ch8": "EEG_RAW_Ch8",
#     },
#     notes="has no SampleNumber column",
# )

PUPILLOMETRY = iMotionsConfig(
    name="pupillometry",
    name_imotions="ET_Eyetracker",
    load_columns=[
        "Timestamp",
        "ET_PupilLeft",
        "ET_PupilRight",
        "ET_DistanceLeft",
        "ET_DistanceRight",
    ],
    rename_columns={
        "ET_PupilLeft": "Pupillometry_L",
        "ET_PupilRight": "Pupillometry_R",
        "ET_DistanceLeft": "Pupillometry_L_Distance",
        "ET_DistanceRight": "Pupillometry_R_Distance",
    },
)

AFFECTIVA = iMotionsConfig(
    name="affectiva",
    name_imotions="Affectiva_AFFDEX_ET_Affectiva_AffectivaCameraDevice",
    load_columns=[
        "SampleNumber",
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
    notes="Affectiva data is not sampled at a constant rate and can contain NaNs.",
)

SYSTEM = iMotionsConfig(
    name="system",
    name_imotions="System_Load_Monitor_iMotions.SysMonitor@1_ET_EventAPI_ExternDevice",
    load_columns=["Timestamp", "CPU Sys", "Memory Sys", "CPU Proc", "Memory Proc"],
    notes="has no SampleNumber column",
)


IMOTIONS_LIST = [TRIAL, STIMULUS, EEG, EDA, PPG, PUPILLOMETRY, AFFECTIVA, SYSTEM]
IMOTIONS_DICT = {config.name: config for config in IMOTIONS_LIST}
