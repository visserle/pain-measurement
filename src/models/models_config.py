from src.models.architectures.EEGFacePhysioEnsemble import EEGFacePhysioEnsemble
from src.models.architectures.EEGNet import EEGNet
from src.models.architectures.EEGPhysioEnsemble import EEGPhysioEnsemble
from src.models.architectures.FacePhysioEnsemble import FacePhysioEnsemble
from src.models.architectures.iTransformer import iTransformer
from src.models.architectures.LightTS import LightTS
from src.models.architectures.MultiLayerPerceptron import MultiLayerPerceptron
from src.models.architectures.PatchTST import PatchTST
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
            "depth": {"type": "int", "low": 1, "high": 4},
            "dropout": {"type": "float", "low": 0.0, "high": 0.8},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "PatchTST": {
        "class": PatchTST,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 256
            "e_layers": {"type": "int", "low": 1, "high": 3},
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
            "d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 256
            "e_layers": {"type": "int", "low": 1, "high": 3},
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "top_k": {"type": "int", "low": 2, "high": 5},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "factor": {"type": "int", "low": 1, "high": 2},
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
    "LightTS": {
        "class": LightTS,
        "format": "3D",
        "hyperparameters": {
            "d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 256
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "EEGNet": {
        "class": EEGNet,
        "format": "3D",
        "hyperparameters": {
            "F1": {"type": "exp", "low": 3, "high": 4},  # 8 to 16 temporal filters
            "D": {"type": "int", "low": 2, "high": 5},  # 2 to 5 depth multiplier
            "dropout": {"type": "float", "low": 0.1, "high": 0.5},
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "EEGPhysioEnsemble": {
        "class": EEGPhysioEnsemble,
        "format": "3D",
        "hyperparameters": {
            # EEGNet parameters
            "F1": {"type": "exp", "low": 3, "high": 4},  # 8 to 16
            "D": {"type": "int", "low": 2, "high": 5},
            "eeg_dropout": {"type": "float", "low": 0.1, "high": 0.5},
            # PatchTST parameters
            "d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 256
            "e_layers": {"type": "int", "low": 1, "high": 3},
            "d_ff": {"type": "exp", "low": 7, "high": 9},  # 128 to 512
            "patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "physio_dropout": {"type": "float", "low": 0.0, "high": 0.5},
            # Fusion parameters
            "fusion_hidden_dim": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "fusion_dropout": {"type": "float", "low": 0.1, "high": 0.5},
            # Common
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "EEGFacePhysioEnsemble": {
        "class": EEGFacePhysioEnsemble,
        "format": "3D",
        "hyperparameters": {
            # EEGNet parameters
            "F1": {"type": "exp", "low": 3, "high": 4},  # 8 to 16
            "D": {"type": "int", "low": 2, "high": 5},
            "eeg_dropout": {"type": "float", "low": 0.1, "high": 0.5},
            # PatchTST parameters for face data
            "face_d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 128
            "face_e_layers": {"type": "int", "low": 1, "high": 3},
            "face_d_ff": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "face_patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "face_stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "face_dropout": {"type": "float", "low": 0.0, "high": 0.5},
            # PatchTST parameters for physio data
            "physio_d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 128
            "physio_e_layers": {"type": "int", "low": 1, "high": 3},
            "physio_d_ff": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "physio_patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "physio_stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "physio_dropout": {"type": "float", "low": 0.0, "high": 0.5},
            # Fusion parameters
            "fusion_hidden_dim": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "fusion_dropout": {"type": "float", "low": 0.1, "high": 0.5},
            # Common
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
    "FacePhysioEnsemble": {
        "class": FacePhysioEnsemble,
        "format": "3D",
        "hyperparameters": {
            # PatchTST parameters for face data
            "face_d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 128
            "face_e_layers": {"type": "int", "low": 1, "high": 3},
            "face_d_ff": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "face_patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "face_stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "face_top_k": {"type": "int", "low": 2, "high": 5},
            "face_dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "face_factor": {"type": "int", "low": 1, "high": 2},
            # PatchTST parameters for physio data
            "physio_d_model": {"type": "exp", "low": 5, "high": 7},  # 32 to 128
            "physio_e_layers": {"type": "int", "low": 1, "high": 3},
            "physio_d_ff": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "physio_patch_len": {"type": "exp", "low": 2, "high": 5},  # 4 to 32
            "physio_stride": {"type": "exp", "low": 2, "high": 4},  # 4 to 16
            "physio_top_k": {"type": "int", "low": 2, "high": 5},
            "physio_dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "physio_factor": {"type": "int", "low": 1, "high": 2},
            # Fusion parameters
            "fusion_hidden_dim": {"type": "exp", "low": 6, "high": 8},  # 64 to 256
            "fusion_dropout": {"type": "float", "low": 0.1, "high": 0.5},
            # Common
            "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
        },
    },
}
