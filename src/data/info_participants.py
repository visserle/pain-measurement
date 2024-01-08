from dataclasses import dataclass, field
from typing import List, Optional


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


participant_001 = Participant(
    id = '001_pilot_bjoern',
)

participant_002 = Participant(
    id = '002_pilot_melis',
    non_available_signals = ['eeg'],
)




# @dataclass
# class Participant:
#     id: str
#     signals: list = field(init=False)
#     non_available_signals: list = None
    
#     def __post_init__(self):
#         SIGNAL_LIST = ['trial','temperature','rating','eda','ecg','eeg','pupillometry','affectiva','system']
#         self.signals = [signal for signal in SIGNAL_LIST if signal not in self.non_available_signals] if self.non_available_signals else SIGNAL_LIST
