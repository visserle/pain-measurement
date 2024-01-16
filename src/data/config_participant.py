# TODO
# change to json or yaml file to make it writable
# create quality assessment functions that check if each trial is valid
# also note down which ecg channel has the best quality

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParticipantConfig:
    id: str
    non_available_data: Optional[List[str]] = field(default_factory=list)
    exclude_trials: Optional[List[int]] = field(default_factory=list)
    best_ecg_channel: Optional[str] = None


p_001 = ParticipantConfig(
    id = '001_pilot_bjoern',
)

p_002 = ParticipantConfig(
    id = '002_pilot_melis',
    non_available_data = ['eeg'],
)

PARTICIPANT_LIST = [
    p_001,
    #p_002,
]