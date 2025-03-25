from src.models.architectures.Crossformer import Crossformer
from src.models.architectures.LongShortTermMemory import LongShortTermMemory
from src.models.architectures.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.architectures.PatchTST import PatchTST
from src.models.architectures.TimesNet import TimesNet

MODELS = {
    # "MLP": {
    #     "class": MultiLayerPerceptron,
    #     "format": "2D",
    #     "hyperparameters": {
    #         "hidden_size": {"type": "exp", "low": 8, "high": 13},
    #         "depth": {"type": "int", "low": 1, "high": 5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #         "dropout_rate": {"type": "float", "low": 0.0, "high": 0.8},
    #     },
    # },
    # "LSTM": {
    #     "class": LongShortTermMemory,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "hidden_size": {"type": "exp", "low": 6, "high": 11},
    #         "num_layers": {"type": "int", "low": 1, "high": 5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    #     },
    # },
    "TimesNet": {
        "class": TimesNet,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 3, "high": 5},  # 8 to 32
            "e_layers": {"type": "int", "low": 1, "high": 3},
            "d_ff": {"type": "exp", "low": 3, "high": 6},  # 32 to 256
            "top_k": {
                "type": "int",
                "low": 2,
                "high": 4,
            },  # important for time-to-2D reshaping
            "num_kernels": {"type": "int", "low": 4, "high": 8},  # inception kernels
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "PatchTST": {
        "class": PatchTST,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 7},  # 64 to 256
            "e_layers": {"type": "int", "low": 1, "high": 4},
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "patch_len": {"type": "exp", "low": 3, "high": 5},  # 8 to 32
            "stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "top_k": {"type": "int", "low": 2, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "factor": {"type": "int", "low": 1, "high": 2},  # attention factor
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    # "Crossformer": {
    #     "class": Crossformer,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "exp", "low": 6, "high": 7},  # 64 to 256
    #         "n_heads": {"type": "int", "low": 6, "high": 10},
    #         "e_layers": {"type": "int", "low": 1, "high": 4},
    #         "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
    #         "factor": {"type": "int", "low": 1, "high": 2},  # attention factor
    #         "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
}
