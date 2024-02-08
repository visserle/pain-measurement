# TODO
# change to json or yaml file to make it writable
# create quality assessment functions that check if each trial is valid
# also note down which ecg channel has the best quality

from dataclasses import dataclass, field


@dataclass
class ParticipantConfig:
    id: str
    not_available_data: list[str] = field(default_factory=list)
    exclude_trials: list[int] = field(default_factory=list)
    best_ecg_channel: str | None = None


# p_001 = ParticipantConfig(
#     id="001_pilot_bjoern",
# )

# p_002 = ParticipantConfig(
#     id="002_pilot_melis",
#     not_available_data=["eeg"],
# )

# p_003 = ParticipantConfig(
#     id="003_pilot_noah",
# )

p_004_1 = ParticipantConfig(
    id="004_zoey1"
)

p_004_2 = ParticipantConfig(
    id="004_zoey2"
)

p_004_3 = ParticipantConfig(
    id="004_zoey3"
)

p_004_4 = ParticipantConfig(
    id="004_zoey4"
)

PARTICIPANT_LIST = [
    p_004_1,
    p_004_2,
    p_004_3,
    p_004_4,
]

PARTICIPANT_DICT = {participant.id: participant for participant in PARTICIPANT_LIST}
