import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """MLP class for time series classification with 3D input

    The input features will be concatenated along the last dimension to form a 1D time
    series before being passed to the network.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # permute to Fortran order
        x = x.flatten(start_dim=1)  # flatten to 2D
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out  # BCEWithLogitsLoss will apply sigmoid


class LongShortTermMemory(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
