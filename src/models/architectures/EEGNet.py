import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        input_len: int,
        input_dim: int = 8,
        F1: int = 8,
        D: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        fs: int = 250,
    ):
        super().__init__()

        # Store parameters
        self.input_len = input_len
        self.input_dim = input_dim  # number of EEG channels
        self.num_classes = num_classes
        self.F1 = F1  # number of temporal filters
        self.D = D  # depth multiplier (number of spatial filters)
        self.F2 = D * F1  # number of pointwise filters
        self.fs = fs  # sampling frequency

        # Define kernel sizes
        kernel_size_1 = (1, round(fs / 2))
        kernel_size_2 = (input_dim, 1)
        kernel_size_3 = (1, round(fs / 8))
        kernel_size_4 = (1, 1)

        kernel_avgpool_1 = (1, 4)
        kernel_avgpool_2 = (1, 8)

        # Calculate paddings
        ks0 = int(round((kernel_size_1[0] - 1) / 2))
        ks1 = int(round((kernel_size_1[1] - 1) / 2))
        kernel_padding_1 = (ks0, ks1 - 1)

        ks0 = int(round((kernel_size_3[0] - 1) / 2))
        ks1 = int(round((kernel_size_3[1] - 1) / 2))
        kernel_padding_3 = (ks0, ks1)

        # layer 1: Temporal Convolution
        self.conv2d = nn.Conv2d(1, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)

        # layer 2: Spatial Convolution
        self.Depthwise_conv2D = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout)

        # layer 3: Separable Convolution
        self.Separable_conv2D_depth = nn.Conv2d(
            D * F1, D * F1, kernel_size_3, padding=kernel_padding_3, groups=D * F1
        )
        self.Separable_conv2D_point = nn.Conv2d(D * self.F1, self.F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(self.F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)

        # Calculate output size for the dense layer
        final_length = self.input_len // 32  # Approximate reduction from pooling layers

        # layer 4: Classification
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(self.F2 * final_length, num_classes)

    def forward(self, x):
        # Input shape: [batch, time_steps, channels]
        batch_size = x.size(0)

        # Reshape to [batch, 1, channels, time_steps] - the "1" is the input_dim for Conv2D
        x = x.permute(0, 2, 1)  # [batch, channels, time_steps]
        x = x.view(batch_size, 1, self.input_dim, self.input_len)

        # layer 1: Temporal Convolution
        y = self.conv2d(x)
        y = self.Batch_normalization_1(y)

        # layer 2: Spatial Convolution
        y = self.Depthwise_conv2D(y)
        y = self.Batch_normalization_2(y)
        y = self.Elu(y)
        y = self.Average_pooling2D_1(y)
        y = self.Dropout(y)

        # layer 3: Separable Convolution
        y = self.Separable_conv2D_depth(y)
        y = self.Separable_conv2D_point(y)
        y = self.Batch_normalization_3(y)
        y = self.Elu(y)
        y = self.Average_pooling2D_2(y)
        y = self.Dropout(y)

        # layer 4: Classification
        y = self.Flatten(y)
        y = self.Dense(y)

        return y
