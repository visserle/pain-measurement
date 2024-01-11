from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParticipantInfo:
    id: str
    non_available_data: Optional[List[str]] = field(default_factory=list)
    exclude_trials: Optional[List[int]] = field(default_factory=list)


p_001 = ParticipantInfo(
    id = '001_pilot_bjoern',
)

p_002 = ParticipantInfo(
    id = '002_pilot_melis',
    non_available_data = ['eeg'],
)


PARTICIPANT_LIST = [
    p_001,
    #p_002,
]