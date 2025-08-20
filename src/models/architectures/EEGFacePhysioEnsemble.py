import torch
import torch.nn as nn

from src.models.architectures.EEGNet import EEGNet
from src.models.architectures.PatchTST import PatchTST


class EEGFacePhysioEnsemble(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
        # EEGNet parameters
        eeg_channels: int = 8,
        F1: int = 8,
        D: int = 3,
        eeg_dropout: float = 0.2,
        fs: int = 250,
        # PatchTST parameters for face data
        face_channels: int = 5,
        face_patch_len: int = 16,
        face_stride: int = 8,
        face_d_model: int = 128,
        face_e_layers: int = 3,
        face_d_ff: int = 256,
        face_top_k: int = 3,
        face_dropout: float = 0.1,
        face_factor: int = 1,
        # PatchTST parameters for physio data
        physio_patch_len: int = 16,
        physio_stride: int = 8,
        physio_d_model: int = 128,
        physio_e_layers: int = 3,
        physio_d_ff: int = 256,
        physio_top_k: int = 3,
        physio_dropout: float = 0.1,
        physio_factor: int = 1,
        # Fusion parameters
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # Calculate lengths for each modality
        # Assuming input_len is at EEG sampling rate (250Hz)
        self.eeg_len = input_len
        self.face_physio_len = input_len // 25  # 250Hz to 10Hz

        # Determine number of physio channels (after EEG and face)
        self.face_channels = face_channels
        self.physio_channels = input_dim - eeg_channels - face_channels

        # EEGNet for EEG data (250Hz)
        self.eeg_model = EEGNet(
            input_len=self.eeg_len,
            input_dim=eeg_channels,
            F1=F1,
            D=D,
            dropout=eeg_dropout,
            num_classes=num_classes,  # will be replaced by fusion layer
            fs=fs,
        )

        # PatchTST for facial expression data (10Hz)
        self.face_model = PatchTST(
            input_len=self.face_physio_len,
            input_dim=self.face_channels,
            patch_len=face_patch_len,
            stride=face_stride,
            d_model=face_d_model,
            e_layers=face_e_layers,
            d_ff=face_d_ff,
            top_k=face_top_k,
            dropout=face_dropout,
            factor=face_factor,
            num_classes=num_classes,
        )

        # PatchTST for physiological data (10Hz)
        self.physio_model = PatchTST(
            input_len=self.face_physio_len,
            input_dim=self.physio_channels,
            patch_len=physio_patch_len,
            stride=physio_stride,
            d_model=physio_d_model,
            e_layers=physio_e_layers,
            d_ff=physio_d_ff,
            top_k=physio_top_k,
            dropout=physio_dropout,
            factor=physio_factor,
            num_classes=num_classes,
        )

        # Remove final classification layers from sub-models
        self.eeg_model.Dense = nn.Identity()
        self.face_model.projection = nn.Identity()
        self.physio_model.projection = nn.Identity()

        # Calculate feature dimensions from each model
        eeg_features = self.eeg_model.F2 * (self.eeg_len // 32)
        face_features = self.face_model.head_nf * self.face_channels
        physio_features = self.physio_model.head_nf * self.physio_channels

        # Fusion network
        total_features = eeg_features + face_features + physio_features
        self.fusion = nn.Sequential(
            nn.Linear(total_features, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x shape: [batch, time_steps, all_channels]
        # Split into EEG, face, and physio channels
        eeg_data = x[:, :, :8]  # First 8 channels are EEG
        face_data = x[:, :, 8:13]  # Next 5 channels are facial expressions
        physio_data = x[:, :, 13:]  # Remaining channels are physio

        # Downsample face and physio data from 250Hz to 10Hz
        # Take every 25th sample
        face_data_downsampled = face_data[:, ::25, :]
        physio_data_downsampled = physio_data[:, ::25, :]

        # Process each modality
        eeg_features = self.eeg_model(eeg_data)
        face_features = self.face_model(face_data_downsampled)
        physio_features = self.physio_model(physio_data_downsampled)

        # Concatenate features from all three modalities
        combined_features = torch.cat(
            [eeg_features, face_features, physio_features], dim=1
        )

        # Final classification
        output = self.fusion(combined_features)

        return output
