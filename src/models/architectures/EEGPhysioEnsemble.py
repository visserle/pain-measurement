import torch
import torch.nn as nn

from src.models.architectures.EEGNet import EEGNet
from src.models.architectures.PatchTST import PatchTST


class EEGPhysioEnsemble(nn.Module):
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
        # PatchTST parameters
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        e_layers: int = 3,
        d_ff: int = 256,
        top_k: int = 3,
        physio_dropout: float = 0.1,
        factor: int = 1,
        # Fusion parameters
        fusion_hidden_dim: int = 128,
        fusion_dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # Calculate lengths for each modality
        # Assuming input_len is at EEG sampling rate (250Hz)
        self.eeg_len = input_len
        self.physio_len = input_len // 25  # 250Hz to 10Hz

        # Determine number of physio channels
        self.physio_channels = input_dim - eeg_channels

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

        # PatchTST for physiological data (10Hz)
        self.physio_model = PatchTST(
            input_len=self.physio_len,
            input_dim=self.physio_channels,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            e_layers=e_layers,
            d_ff=d_ff,
            top_k=top_k,
            dropout=physio_dropout,
            factor=factor,
            num_classes=num_classes,
        )

        # Remove final classification layers from sub-models
        self.eeg_model.Dense = nn.Identity()
        self.physio_model.projection = nn.Identity()

        # Calculate feature dimensions from each model
        eeg_features = self.eeg_model.F2 * (self.eeg_len // 32)
        physio_features = self.physio_model.head_nf * self.physio_channels

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(eeg_features + physio_features, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        # x shape: [batch, time_steps, all_channels]
        # Split into EEG and physio channels
        eeg_data = x[:, :, :8]  # First 8 channels are EEG
        physio_data = x[:, :, 8:]  # Remaining channels are physio

        # Downsample physio data from 250Hz to 10Hz
        # Take every 25th sample
        physio_data_downsampled = physio_data[:, ::25, :]

        # Process each modality
        eeg_features = self.eeg_model(eeg_data)
        physio_features = self.physio_model(physio_data_downsampled)

        # Concatenate features
        combined_features = torch.cat([eeg_features, physio_features], dim=1)

        # Final classification
        output = self.fusion(combined_features)

        return output
