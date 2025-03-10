from src.models.architectures.LongShortTermMemory import LongShortTermMemory
from src.models.architectures.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.architectures.PatchTST import PatchTST

MODELS = {
    # "MLP": {
    #     "class": MultiLayerPerceptron,
    #     "format": "2D",
    #     "hyperparameters": {
    #         "hidden_size": {
    #             "type": "categorical",
    #             "choices": [256, 512, 1024, 2048, 4096, 8192],
    #         },
    #         "depth": {"type": "int", "low": 1, "high": 5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #         "dropout_rate": {"type": "float", "low": 0.0, "high": 0.8},
    #     },
    # },
    # Define other models with their respective hyperparameters and input sizes
    # "LSTM": {
    #     "class": LongShortTermMemory,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "hidden_size": {
    #             "type": "categorical",
    #             "choices": [64, 128, 256, 512, 1024],
    #         },
    #         "num_layers": {"type": "int", "low": 1, "high": 5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    #     },
    # },
    # "TimesNet": {
    #     "class": TimesNet,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "categorical", "choices": [16]},
    #         "e_layers": {"type": "int", "low": 2, "high": 3},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
    "PatchTST": {
        "class": PatchTST,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "categorical", "choices": [128]},
            "e_layers": {"type": "int", "low": 2, "high": 3},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
}
