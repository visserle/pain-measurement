# TODO
# change to json or yaml file to make it easier to read and write
# create quality assessment functions that check if each trial is valid

from dataclasses import dataclass, field


@dataclass
class ParticipantConfig:
    id: str
    exclude_trials: list[int] = field(default_factory=list)


p_1 = ParticipantConfig(id="1")


PARTICIPANT_LIST = [p_1]

PARTICIPANT_DICT = {participant.id: participant for participant in PARTICIPANT_LIST}
