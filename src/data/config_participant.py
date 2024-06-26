# TODO
# change to json or yaml file to make it easier to read and write?
# create quality assessment functions that check if each trial is valid
# add support for excluding trials

import json
from dataclasses import dataclass, field
from pathlib import Path

PARTICIPANTS_JSON = Path("src/data/participants.json")
with open(PARTICIPANTS_JSON, "r") as f:
    participants = json.load(f)
    PARTICIPANT_IDS = participants["PARTICIPANT_IDS"]
    EXCLUDED = participants["EXCLUDED"]


@dataclass
class ParticipantConfig:
    id: str
    excluded: bool = False
    exclude_trials: list[int] = field(default_factory=list)


class ParticipantManager:
    def __init__(self):
        self._participants_list: list[ParticipantConfig] = []
        self._participants_dict: dict[str, ParticipantConfig] = {}

    def add_participant(self, participant: ParticipantConfig):
        if participant.id in self._participants_dict:
            raise ValueError(f"Participant with id {participant.id} already exists")
        self._participants_list.append(participant)
        self._participants_dict[participant.id] = participant

    def remove_participant(self, participant_id: str):
        participant = self._participants_dict.pop(participant_id, None)
        if participant:
            self._participants_list.remove(participant)

    def get_participant_dict(self):  # TODO maybe use a property instead with a setter
        return dict(
            filter(lambda p: not p[1].excluded, self._participants_dict.items())
        )

    def get_participant_list(self):
        return list(filter(lambda p: not p.excluded, self._participants_list))


# Create participants
manager = ParticipantManager()
for participant_id in PARTICIPANT_IDS:
    if participant_id in EXCLUDED:
        manager.add_participant(ParticipantConfig(id=participant_id, excluded=True))
    else:
        manager.add_participant(ParticipantConfig(id=participant_id))

PARTICIPANT_LIST = manager.get_participant_list()
PARTICIPANT_DICT = manager.get_participant_dict()


if __name__ == "__main__":
    print(PARTICIPANT_LIST)
