from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from src.data.info_data import DATA_DICT

if Path.cwd().name != 'pain-measurement':
    raise EnvironmentError('Working directory must be the root of the project')
ROOT_DIR = Path.cwd()


@dataclass
class Participant:
    id: str
    data: List[str] = field(init=False)
    non_available_data: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        # Handle missing keys in DATA_DICT
        self.data = [data for data in DATA_DICT if data not in self.non_available_data and data in DATA_DICT]
        
    def get_path(self, data: str) -> Path:
        if data not in DATA_DICT:
            raise KeyError(f"Data key '{data}' not found in DATA_DICT")
        return ROOT_DIR / 'data' / 'raw' / self.id / DATA_DICT[data].path
    
    def get_paths(self, ignore: Optional[List[str]] = None) -> List[Path]:
        if ignore is None:
            ignore = []
        return [self.get_path(data) for data in self.data if data not in ignore]


p_001 = Participant(
    id = '001_pilot_bjoern',
)

p_002 = Participant(
    id = '002_pilot_melis',
    non_available_data = ['eeg'],
)
