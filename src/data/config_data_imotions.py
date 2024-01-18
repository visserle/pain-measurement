"""Contains information about the iMotions data files (= external data)."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from src.data.config_data import DataConfigBase
from src.data.transform_data import create_trials, interpolate_to_marker_timestamps


@dataclass
class iMotionsConfig(DataConfigBase):
    name: str
    name_imotions: str
    load_columns: List[str] # columns to load
    rename_columns: Optional[Dict[str, str]] = None
    transformations: Optional[List] = None
    sample_rate: Optional[int] = None # TODO?!
    notes: Optional[str] = None
    
    def __post_init__(self):
        self.load_dir = Path('data/imotions')
        self.save_dir = Path('data/raw')
        self.transformations = [create_trials, interpolate_to_marker_timestamps] if not self.transformations else [create_trials, interpolate_to_marker_timestamps] + self.transformations
        

TRIAL = iMotionsConfig(
    name = 'trial',
    name_imotions = 'ExternalMarker_ET_EventAPI_ExternDevice',
    load_columns = ["Timestamp","MarkerDescription"],
    rename_columns = {
        "MarkerDescription": "Stimuli_Seed",
        },
    notes = "MarkerDescription contains the stimulus seed and is originally send once at the start and end of each trial.",
)

TEMPERATURE = iMotionsConfig(
    name = 'temperature',
    name_imotions = 'TemperatureCurve_TemperatureCurve@1_ET_EventAPI_ExternDevice',
    load_columns = ["Timestamp","Temperature"],
)

RATING = iMotionsConfig(
    name = 'rating',
    name_imotions = 'RatingCurve_RatingCurve@1_ET_EventAPI_ExternDevice',
    load_columns = ["Timestamp","Rating"],
)

EDA = iMotionsConfig(
    name = 'eda',
    name_imotions = 'Shimmer3_GSR+_&_EDA_(D200)_Shimmer3_GSR+_&_EDA_(D200)_ET_Shimmer_ShimmerDevice',
    load_columns = ["SampleNumber","Timestamp","GSR Conductance CAL","VSenseBatt CAL","Packet reception rate RAW"],
    rename_columns = {
        "GSR Conductance CAL": "EDA_RAW",
        "VSenseBatt CAL": "EDA_d_Battery",
        "Packet reception rate RAW": "EDA_d_PacketReceptionRate",
        },
)

ECG = iMotionsConfig(
    name = 'ecg',
    name_imotions = 'Shimmer3_ECG_(68BF)_Shimmer3_ECG_(68BF)_ET_Shimmer_ShimmerDevice',
    load_columns = ["SampleNumber","Timestamp","ECG LL-RA CAL","ECG LA-RA CAL","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG","VSenseBatt CAL","Packet reception rate RAW"],
    rename_columns = {
        "ECG LL-RA CAL": "ECG_LL-RA",
        "ECG LA-RA CAL": "ECG_LA-RA",
        "ECG Vx-RL CAL": "ECG_Vx-RL",
        "Heart Rate ECG LL-RA ALG": "ECG_LL-RA_HeartRate",
        "IBI ECG LL-RA ALG": "ECG_LL-RA_IBI",
        "VSenseBatt CAL": "ECG_d_Battery",
        "Packet reception rate RAW": "ECG_d_PacketReceptionRate",
        },
)

EEG = iMotionsConfig(
    name = 'eeg',
    name_imotions = 'EEG_Enobio-Wifi_ENOBIO-8-NE-Wifi(00_07_80_0D_82_F7)_ET_Lsl_LslSensor',
    load_columns = ["Timestamp","Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
    rename_columns = {
        "Ch1": "EEG_RAW_Ch1",
        "Ch2": "EEG_RAW_Ch2",
        "Ch3": "EEG_RAW_Ch3",
        "Ch4": "EEG_RAW_Ch4",
        "Ch5": "EEG_RAW_Ch5",
        "Ch6": "EEG_RAW_Ch6",
        "Ch7": "EEG_RAW_Ch7",
        "Ch8": "EEG_RAW_Ch8",
        },
    notes = "has no SampleNumber column",
)

PUPILLOMETRY = iMotionsConfig(
    name = 'pupillometry',
    name_imotions = 'ET_Eyetracker',
    load_columns = ["Timestamp","ET_PupilLeft","ET_PupilRight","ET_DistanceLeft","ET_DistanceRight"],
    rename_columns = {
        "ET_PupilLeft": "Pupillometry_L",
        "ET_PupilRight": "Pupillometry_R",
        "ET_DistanceLeft": "Pupillometry_L_Distance",
        "ET_DistanceRight": "Pupillometry_R_Distance",
        },
)

AFFECTIVA = iMotionsConfig(
    name = 'affectiva',
    name_imotions = 'Affectiva_AFFDEX_ET_Affectiva_AffectivaCameraDevice',
    load_columns = ["SampleNumber","Timestamp","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    notes = "Affectiva data is not sampled at a constant rate and can contain NaNs.",
)

SYSTEM = iMotionsConfig(
    name = 'system',
    name_imotions = 'System_Load_Monitor_iMotions.SysMonitor@1_ET_EventAPI_ExternDevice',
    load_columns = ["Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
    notes = "has no SampleNumber column",
)


IMOTIONS_LIST = [TRIAL, TEMPERATURE, RATING, EDA, ECG, EEG, PUPILLOMETRY, AFFECTIVA, SYSTEM]
