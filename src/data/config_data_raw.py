from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from src.data.config_data import DataConfigBase
from src.data.transform_data import create_timedelta_index


@dataclass
class RawConfig(DataConfigBase):
    name: str
    load_columns: List[str]
    plot_columns: Optional[List[str]] = None
    transformations: Optional[List] = None
    notes: Optional[str] = None
    
    def __post_init__(self):
        self.load_dir = Path('data/raw')
        self.save_dir = Path('data/trial')
        self.transformations = [create_timedelta_index] if not self.transformations else [create_timedelta_index] + self.transformations


TRIAL = RawConfig(
    name = 'trial',
    load_columns = ["Timestamp","Stimuli_Seed"],
    notes = ""
)

TEMPERATURE = RawConfig(
    name = 'temperature',
    load_columns = ["Timestamp","Temperature"],
    plot_columns = ["Temperature"],
)

RATING = RawConfig(
    name = 'rating',
    load_columns = ["Timestamp","Rating"],
    plot_columns = ["Rating"],
)

EDA = RawConfig(
    name = 'eda',
    load_columns = ["Timestamp","EDA_RAW","EDA_d_Battery","EDA_d_PacketReceptionRate"],
    plot_columns = ["EDA_RAW"],
)

ECG = RawConfig(
    name = 'ecg',
    load_columns = ["Timestamp","ECG_LL-RA","ECG_LA-RA","ECG_Vx-RL","ECG_LL-RA_HeartRate","ECG_LL-RA_IBI","ECG_d_Battery","ECG_d_PacketReceptionRate"],
    plot_columns = ["ECG_LL-RA","ECG_LA-RA","ECG_Vx-RL","ECG_LL-RA_HeartRate","ECG_LL-RA_IBI"],
)

EEG = RawConfig(
    name = 'eeg',
    load_columns = ["Timestamp","EEG_RAW_Ch1","EEG_RAW_Ch2","EEG_RAW_Ch3","EEG_RAW_Ch4","EEG_RAW_Ch5","EEG_RAW_Ch6","EEG_RAW_Ch7","EEG_RAW_Ch8"],
    plot_columns = ["EEG_RAW_Ch1","EEG_RAW_Ch2","EEG_RAW_Ch3","EEG_RAW_Ch4","EEG_RAW_Ch5","EEG_RAW_Ch6","EEG_RAW_Ch7","EEG_RAW_Ch8"],
)

PUPILLOMETRY = RawConfig(
    name = 'pupillometry',
    load_columns = ["Timestamp","Pupillometry_L","Pupillometry_R","Pupillometry_L_Distance","Pupillometry_R_Distance"],
    plot_columns=["Pupillometry_L","Pupillometry_R"],
)

AFFECTIVA = RawConfig(
    name = 'affectiva',
    load_columns = ["Timestamp","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    plot_columns = ["Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    notes = "Affectiva data is not sampled at a constant rate and the only orignal data that can contain NaNs.",
)

SYSTEM = RawConfig(
    name = 'system',
    load_columns = ["Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
)

DATA_DICT = {
    TRIAL.name: TRIAL,
    TEMPERATURE.name: TEMPERATURE,
    RATING.name: RATING,
    EDA.name: EDA,
    ECG.name: ECG,
    EEG.name: EEG,
    PUPILLOMETRY.name: PUPILLOMETRY,
    AFFECTIVA.name: AFFECTIVA,
    SYSTEM.name: SYSTEM,
}

RAW_LIST = [TRIAL, TEMPERATURE, RATING, EDA, ECG, EEG, PUPILLOMETRY, AFFECTIVA, SYSTEM]
