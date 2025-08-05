import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """MLP class for time series classification with 3D input

    The input features will be concatenated along the last dimension to form a 1D time
    series before being passed to the network.
    """

    def __init__(
        self,
        input_len: int,
        hidden_size: int,
        input_dim: int = 1,
        depth: int = 2,
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if input_dim != 1:
            raise NotImplementedError("Only 1D input supported.")

        layers = []

        # First layer: input_len -> hidden_size
        layers.append(nn.Linear(input_len, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Middle layers: hidden_size -> hidden_size (depth-1 additional hidden layers)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer: hidden_size -> num_classes
        layers.append(nn.Linear(hidden_size, num_classes))

        # Convert list of layers to a sequential module
        self.model = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # permute to Fortran order
        x = x.flatten(start_dim=1)  # flatten to 2D
        return self.model(x)  # returns logits
