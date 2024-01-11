"""Contains information about the iMotions data files (= external data)."""

from dataclasses import dataclass
from typing import List, Optional, Dict
from src.data.config_data import DataConfigBase

@dataclass
class iMotionsConfig(DataConfigBase):
    name: str
    path: str
    imotions_columns: List[str]  # columns copied from the imotions csv file
    keep_columns: List[str]
    rename_columns: Optional[Dict[str, str]] = None
    device_sampling_rate: Optional[float] = None
    notes: Optional[str] = None
    frozen = True


TRIAL = iMotionsConfig(
    name = 'trial',
    path = 'ExternalMarker_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","MarkerName","MarkerDescription","MarkerType","SceneType"],
    keep_columns = ["Timestamp","MarkerDescription"],
    rename_columns = {
        "MarkerDescription": "Stimuli_Seed",
        },
    notes = "MarkerDescription contains the stimulus seed and is originally send once at the start and end of each trial.",
)

TEMPERATURE = iMotionsConfig(
    name = 'temperature',
    path = 'TemperatureCurve_TemperatureCurve@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","Temperature"],
    keep_columns = ["Timestamp","Temperature"],
)

RATING = iMotionsConfig(
    name = 'rating',
    path = 'RatingCurve_RatingCurve@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","Rating"],
    keep_columns = ["Timestamp","Rating"],
)

EDA = iMotionsConfig(
    name = 'eda',
    path = 'Shimmer3_GSR+_&_EDA_(D200)_Shimmer3_GSR+_&_EDA_(D200)_ET_Shimmer_ShimmerDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Timestamp RAW","Timestamp CAL","System Timestamp CAL","VSenseBatt RAW","VSenseBatt CAL","GSR RAW","GSR Resistance CAL","GSR Conductance CAL","Packet reception rate RAW"],
    keep_columns = ["Timestamp","GSR Conductance CAL","VSenseBatt CAL","Packet reception rate RAW"],
    rename_columns = {
        "GSR Conductance CAL": "EDA_RAW",
        "VSenseBatt CAL": "EDA_d_Battery",
        "Packet reception rate RAW": "EDA_d_PacketReceptionRate",
        },
    device_sampling_rate = 128,
)

ECG = iMotionsConfig(
    name = 'ecg',
    path = 'Shimmer3_ECG_(68BF)_Shimmer3_ECG_(68BF)_ET_Shimmer_ShimmerDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Timestamp RAW","Timestamp CAL","System Timestamp CAL","VSenseBatt RAW","VSenseBatt CAL","EXG1 Status RAW","ECG LL-RA RAW","ECG LL-RA CAL","ECG LA-RA RAW","ECG LA-RA CAL","EXG2 Status RAW","ECG Vx-RL RAW","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG","Packet reception rate RAW"],
    keep_columns = ["Timestamp","ECG LL-RA CAL","ECG LA-RA CAL","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG","VSenseBatt CAL","Packet reception rate RAW"],
    rename_columns = {
        "ECG LL-RA CAL": "ECG_LL-RA",
        "ECG LA-RA CAL": "ECG_LA-RA",
        "ECG Vx-RL CAL": "ECG_Vx-RL",
        "Heart Rate ECG LL-RA ALG": "ECG_LL-RA_HeartRate",
        "IBI ECG LL-RA ALG": "ECG_LL-RA_IBI",
        "VSenseBatt CAL": "ECG_d_Battery",
        "Packet reception rate RAW": "ECG_d_PacketReceptionRate",
        },
    device_sampling_rate = 512,
)

EEG = iMotionsConfig(
    name = 'eeg',
    path = 'EEG_Enobio-Wifi_ENOBIO-8-NE-Wifi(00_07_80_0D_82_F7)_ET_Lsl_LslSensor.csv',
    imotions_columns = ["RowNumber","Timestamp","LSL Timestamp","Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
    keep_columns = ["Timestamp","Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
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
    device_sampling_rate = 500,
)

PUPILLOMETRY = iMotionsConfig(
    name = 'pupillometry',
    path = 'ET_Eyetracker.csv',
    imotions_columns = ["RowNumber","Timestamp","ET_GazeLeftx","ET_GazeLefty","ET_GazeRightx","ET_GazeRighty","ET_PupilLeft","ET_PupilRight","ET_TimeSignal","ET_DistanceLeft","ET_DistanceRight","ET_CameraLeftX","ET_CameraLeftY","ET_CameraRightX","ET_CameraRightY"],
    keep_columns = ["Timestamp","ET_PupilLeft","ET_PupilRight","ET_DistanceLeft","ET_DistanceRight"],
    rename_columns = {
        "ET_PupilLeft": "Pupillometry_L",
        "ET_PupilRight": "Pupillometry_R",
        "ET_DistanceLeft": "Pupillometry_L_Distance",
        "ET_DistanceRight": "Pupillometry_R_Distance",
        },
    device_sampling_rate = 60,
)

AFFECTIVA = iMotionsConfig(
    name = 'affectiva',
    path = 'Affectiva_AFFDEX_ET_Affectiva_AffectivaCameraDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    keep_columns = ["Timestamp","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    notes = "Affectiva data is not sampled at a constant rate and can contain NaNs.",
)

SYSTEM = iMotionsConfig(
    name = 'system',
    path = 'System_Load_Monitor_iMotions.SysMonitor@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
    keep_columns = ["Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
)


IMOTIONS_LIST = [TRIAL, TEMPERATURE, RATING, EDA, ECG, EEG, PUPILLOMETRY, AFFECTIVA, SYSTEM]
