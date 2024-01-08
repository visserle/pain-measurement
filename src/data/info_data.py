from dataclasses import dataclass
from typing import List, Optional, Dict

__all__ = ['DATA_DICT', 'DataInfo', 'trial', 'temperature', 'rating', 'eda', 'ecg', 'eeg', 'pupillometry', 'affectiva', 'system']


@dataclass
class DataInfo:
    name: str
    path: str
    imotions_columns: List[str]  # columns copied from the imotions csv file
    keep_columns: List[str]
    plot_columns: Optional[List[str]] = None
    rename_columns: Optional[Dict[str, str]] = None
    native_sampling_rate: Optional[float] = None


trial = DataInfo(
    name = 'trial',
    path = 'ExternalMarker_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","MarkerName","MarkerDescription","MarkerType","SceneType"],
    keep_columns = ["Timestamp","MarkerDescription"],
)

temperature = DataInfo(
    name = 'temperature',
    path = 'TemperatureCurve_TemperatureCurve@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","Temperature"],
    keep_columns = ["Timestamp","Temperature"],
    plot_columns = ["Temperature"],
)

rating = DataInfo(
    name = 'rating',
    path = 'RatingCurve_RatingCurve@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","Rating"],
    keep_columns = ["Timestamp","Rating"],
    plot_columns = ["Rating"],
)

eda = DataInfo(
    name = 'eda',
    path = 'Shimmer3_GSR+_&_EDA_(D200)_Shimmer3_GSR+_&_EDA_(D200)_ET_Shimmer_ShimmerDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Timestamp RAW","Timestamp CAL","System Timestamp CAL","VSenseBatt RAW","VSenseBatt CAL","GSR RAW","GSR Resistance CAL","GSR Conductance CAL","Packet reception rate RAW"],
    keep_columns = ["Timestamp","GSR RAW","GSR Conductance CAL"],
    plot_columns = ["GSR Conductance CAL"],
    native_sampling_rate = 128,
)

ecg = DataInfo(
    name = 'ecg',
    path = 'Shimmer3_ECG_(68BF)_Shimmer3_ECG_(68BF)_ET_Shimmer_ShimmerDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Timestamp RAW","Timestamp CAL","System Timestamp CAL","VSenseBatt RAW","VSenseBatt CAL","EXG1 Status RAW","ECG LL-RA RAW","ECG LL-RA CAL","ECG LA-RA RAW","ECG LA-RA CAL","EXG2 Status RAW","ECG Vx-RL RAW","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG","Packet reception rate RAW"],
    keep_columns = ["Timestamp","ECG LL-RA RAW","ECG LL-RA CAL","ECG LA-RA RAW","ECG LA-RA CAL","ECG Vx-RL RAW","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG"],
    plot_columns = ["ECG LL-RA CAL","ECG LA-RA CAL","ECG Vx-RL CAL","Heart Rate ECG LL-RA ALG","IBI ECG LL-RA ALG"],
    native_sampling_rate = 512,
)

eeg = DataInfo(
    name = 'eeg',
    path = 'EEG_Enobio-Wifi_ENOBIO-8-NE-Wifi(00_07_80_0D_82_F7)_ET_Lsl_LslSensor.csv',
    imotions_columns = ["RowNumber","Timestamp","LSL Timestamp","Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
    keep_columns = ["Timestamp","Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
    plot_columns = ["Ch1","Ch2","Ch3","Ch4","Ch5","Ch6","Ch7","Ch8"],
    native_sampling_rate = 500,
)

pupillometry = DataInfo(
    name = 'pupillometry',
    path = 'ET_Eyetracker.csv',
    imotions_columns = ["RowNumber","Timestamp","ET_GazeLeftx","ET_GazeLefty","ET_GazeRightx","ET_GazeRighty","ET_PupilLeft","ET_PupilRight","ET_TimeSignal","ET_DistanceLeft","ET_DistanceRight","ET_CameraLeftX","ET_CameraLeftY","ET_CameraRightX","ET_CameraRightY"],
    keep_columns = ["Timestamp","ET_GazeLeftx","ET_GazeLefty","ET_GazeRightx","ET_GazeRighty","ET_PupilLeft","ET_PupilRight","ET_TimeSignal","ET_DistanceLeft","ET_DistanceRight","ET_CameraLeftX","ET_CameraLeftY","ET_CameraRightX","ET_CameraRightY"],
    plot_columns = ["ET_GazeLeftx","ET_GazeLefty","ET_GazeRightx","ET_GazeRighty","ET_PupilLeft","ET_PupilRight","ET_TimeSignal","ET_DistanceLeft","ET_DistanceRight","ET_CameraLeftX","ET_CameraLeftY","ET_CameraRightX","ET_CameraRightY"],
    native_sampling_rate = 60,
)

affectiva = DataInfo(
    name = 'affectiva',
    path = 'Affectiva_AFFDEX_ET_Affectiva_AffectivaCameraDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","SampleNumber","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    keep_columns = ["Timestamp","Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
    plot_columns = ["Anger","Contempt","Disgust","Fear","Joy","Sadness","Surprise","Engagement","Valence","Sentimentality","Confusion","Neutral","Attention","Brow Furrow","Brow Raise","Cheek Raise","Chin Raise","Dimpler","Eye Closure","Eye Widen","Inner Brow Raise","Jaw Drop","Lip Corner Depressor","Lip Press","Lip Pucker","Lip Stretch","Lip Suck","Lid Tighten","Mouth Open","Nose Wrinkle","Smile","Smirk","Upper Lip Raise","Blink","BlinkRate","Pitch","Yaw","Roll","Interocular Distance"],
)

system = DataInfo(
    name = 'system',
    path = 'System_Load_Monitor_iMotions.SysMonitor@1_ET_EventAPI_ExternDevice.csv',
    imotions_columns = ["RowNumber","Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
    keep_columns = ["Timestamp","CPU Sys","Memory Sys","CPU Proc","Memory Proc"],
)


DATA_DICT = {
    trial.name: trial,
    temperature.name: temperature,
    rating.name: rating,
    eda.name: eda,
    ecg.name: ecg,
    eeg.name: eeg,
    pupillometry.name: pupillometry,
    affectiva.name: affectiva,
    system.name: system,
}