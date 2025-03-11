from src.models.architectures.LongShortTermMemory import LongShortTermMemory
from src.models.architectures.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.architectures.PatchTST import PatchTST
from src.models.architectures.TimesNet import TimesNet

MODELS = {
    "MLP": {
        "class": MultiLayerPerceptron,
        "format": "2D",
        "hyperparameters": {
            "hidden_size": {"type": "exp", "low": 8, "high": 13},
            "depth": {"type": "int", "low": 1, "high": 5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.8},
        },
    },
    # "LSTM": {
    #     "class": LongShortTermMemory,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "hidden_size": {"type": "exp", "low": 6, "high": 11},
    #         "num_layers": {"type": "int", "low": 1, "high": 5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    #     },
    # },
    # "TimesNet": {
    #     "class": TimesNet,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "int", "low": 8, "high": 32},  # default 16
    #         "e_layers": {"type": "int", "low": 2, "high": 3},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
    # "PatchTST": {
    #     "class": PatchTST,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "exp", "low": 6, "high": 8},  # default 128
    #         "e_layers": {"type": "int", "low": 2, "high": 3},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
}
