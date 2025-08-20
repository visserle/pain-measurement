import torch
import torch.nn as nn

from src.models.architectures.PatchTST import PatchTST


class FacePhysioEnsemble(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int,
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
        # Assuming input_len is at sampling rate (10Hz)
        self.input_len = input_len

        # Determine number of physio channels (after face)
        self.face_channels = face_channels
        self.physio_channels = input_dim - face_channels

        # PatchTST for facial expression data (10Hz)
        self.face_model = PatchTST(
            input_len=self.input_len,
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
            input_len=self.input_len,
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
        self.face_model.projection = nn.Identity()
        self.physio_model.projection = nn.Identity()

        # Calculate feature dimensions from each model
        face_features = self.face_model.head_nf * self.face_channels
        physio_features = self.physio_model.head_nf * self.physio_channels

        # Fusion network
        total_features = face_features + physio_features
        self.fusion = nn.Sequential(
            nn.Linear(total_features, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x shape: [batch, time_steps, all_channels]
        # Split into face and physio channels
        face_data = x[
            :, :, : self.face_channels
        ]  # First 5 channels are facial expressions
        physio_data = x[:, :, self.face_channels :]  # Remaining channels are physio

        # Process each modality
        face_features = self.face_model(face_data)
        physio_features = self.physio_model(physio_data)

        # Concatenate features from both modalities
        combined_features = torch.cat([face_features, physio_features], dim=1)

        # Final classification
        output = self.fusion(combined_features)

        return output
