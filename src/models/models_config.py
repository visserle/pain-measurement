from src.models.architectures.Crossformer import Crossformer
from src.models.architectures.DLinear import DLinear
from src.models.architectures.ETSformer import ETSformer
from src.models.architectures.iTransformer import iTransformer
from src.models.architectures.LightTS import LightTS
from src.models.architectures.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.architectures.PatchTST import PatchTST
from src.models.architectures.Pyraformer import Pyraformer
from src.models.architectures.Reformer import Reformer
from src.models.architectures.TimesNet import TimesNet

# for exponential search space
# 2^3 = 8
# 2^4 = 16
# 2^5 = 32
# 2^6 = 64
# 2^7 = 128
# 2^8 = 256
# 2^9 = 512
# 2^10 = 1024
# 2^11 = 2048
# 2^12 = 4096
# 2^13 = 8192

MODELS = {
    "MLP": {
        "class": MultiLayerPerceptron,
        "format": "2D",
        "hyperparameters": {
            "hidden_size": {"type": "exp", "low": 8, "high": 12},  # 256 to 8192
            "depth": {"type": "int", "low": 1, "high": 5},
            "dropout_rate": {"type": "float", "low": 0.0, "high": 0.8},
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
            "patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "top_k": {"type": "int", "low": 2, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "factor": {"type": "int", "low": 1, "high": 2},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "iTransformer": {
        "class": iTransformer,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "e_layers": {"type": "int", "low": 1, "high": 4},
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "top_k": {"type": "int", "low": 2, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "factor": {"type": "int", "low": 1, "high": 2},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "Crossformer": {
        "class": Crossformer,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 7},
            "e_layers": {"type": "int", "low": 1, "high": 4},
            "d_ff": {"type": "exp", "low": 7, "high": 9},
            "factor": {"type": "int", "low": 1, "high": 2},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "TimesNet": {
        "class": TimesNet,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 3, "high": 5},
            "e_layers": {"type": "int", "low": 1, "high": 3},
            "d_ff": {"type": "exp", "low": 3, "high": 6},
            "top_k": {"type": "int", "low": 2, "high": 4},
            "num_kernels": {"type": "int", "low": 4, "high": 8},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "Reformer": {
        "class": Reformer,
        "format": "3D",
        "hyperparameters": {
            "e_layers": {"type": "int", "low": 2, "high": 6},
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "d_ff": {"type": "exp", "low": 7, "high": 10},
            "top_k": {"type": "int", "low": 1, "high": 8},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "DLinear": {
        "class": DLinear,
        "format": "3D",
        "hyperparameters": {
            "e_layers": {"type": "int", "low": 1, "high": 5},
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "d_ff": {"type": "exp", "low": 7, "high": 10},
            "top_k": {"type": "int", "low": 1, "high": 5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "ETSformer": {
        "class": ETSformer,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "e_layers": {"type": "int", "low": 1, "high": 6},
            "d_layers": {"type": "int", "low": 1, "high": 6},
            "top_k": {"type": "int", "low": 1, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "Pyraformer": {
        "class": Pyraformer,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "e_layers": {"type": "int", "low": 2, "high": 6},
            "top_k": {"type": "int", "low": 2, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        },
    },
    "LightTS": {
        "class": LightTS,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "e_layers": {"type": "int", "low": 1, "high": 6},
            "top_k": {"type": "int", "low": 1, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    # Commented out as they did not scale well with EEG data:
    # "NonstationaryTransformer": {
    #     "class": NonstationarityTransformer,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
    #         "e_layers": {"type": "int", "low": 1, "high": 4},
    #         "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
    #         "factor": {"type": "int", "low": 1, "high": 2},
    #         "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
    # "Transformer": {
    #     "class": Transformer,
    #     "format": "3D",
    #     "hyperparameters": {
    #         "d_model": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
    #         "e_layers": {"type": "int", "low": 1, "high": 4},
    #         "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
    #         "factor": {"type": "int", "low": 1, "high": 2},
    #         "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    #         "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    #     },
    # },
}
