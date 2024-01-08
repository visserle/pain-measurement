from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from src.data.info_data import DataInfo, trial, temperature, rating, eda, ecg, pupillometry, affectiva, system

assert Path.cwd().name == 'pain-measurement', 'Working directory must be the root of the project'
ROOT_DIR = Path.cwd()

@dataclass
class Participant:
    id: str
    signals: List[str] = field(init=False)
    non_available_signals: Optional[List[str]] = field(default_factory=list)

    SIGNAL_LIST = ['trial', 'temperature', 'rating', 'eda', 'ecg', 'eeg', 'pupillometry', 'affectiva', 'system']

    def __post_init__(self):
        if not all(signal in self.SIGNAL_LIST for signal in self.non_available_signals):
            raise ValueError("non_available_signals contains invalid signal names")
        self.signals = [signal for signal in self.SIGNAL_LIST if signal not in self.non_available_signals]

    def get_path(self, data_info: DataInfo) -> Path:
        """Constructs the full path for a participant's data."""
        return ROOT_DIR / 'data' / 'raw' / self.id / data_info.path
        

p_001 = Participant(
    id = '001_pilot_bjoern',
)

p_002 = Participant(
    id = '002_pilot_melis',
    non_available_signals = ['eeg'],
)

print(p_001.get_path(trial))


# @dataclass
# class Participant:
#     id: str
#     signals: list = field(init=False)
#     non_available_signals: list = None
    
#     def __post_init__(self):
#         SIGNAL_LIST = ['trial','temperature','rating','eda','ecg','eeg','pupillometry','affectiva','system']
#         self.signals = [signal for signal in SIGNAL_LIST if signal not in self.non_available_signals] if self.non_available_signals else SIGNAL_LIST
