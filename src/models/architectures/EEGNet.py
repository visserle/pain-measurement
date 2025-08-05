# Based on: DOI : 10.1088/1741-2552/aace8c
# Modified from: https://github.com/Amir-Hofo/EEGNet

import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        input_len: int,
        num_channels: int = 8,
        input_dim: int = 1,
        F1: int = 8,
        D: int = 3,
        dropout: float = 0.2,
        num_classes: int = 2,
        fs: int = 250,
    ):
        super().__init__()

        # Store parameters
        self.input_len = input_len
        self.num_channels = num_channels  # number of EEG channels
        self.input_dim = (
            input_dim  # number of channels in input (for EEG signal usually 1)
        )
        self.num_classes = num_classes
        self.F1 = F1  # number of temporal filters
        self.D = D  # depth multiplier (number of spatial filters)
        self.F2 = D * F1  # number of pointwise filters
        self.fs = fs  # sampling frequency

        # Define kernel sizes
        kernel_size_1 = (1, round(fs / 2))
        kernel_size_2 = (num_channels, 1)
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

        # layer 1
        self.conv2d = nn.Conv2d(input_dim, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_normalization_1 = nn.BatchNorm2d(F1)

        # layer 2
        self.Depthwise_conv2D = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.Batch_normalization_2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU()
        self.Average_pooling2D_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout = nn.Dropout2d(dropout)

        # layer 3
        self.Separable_conv2D_depth = nn.Conv2d(
            D * F1, D * F1, kernel_size_3, padding=kernel_padding_3, groups=D * F1
        )
        self.Separable_conv2D_point = nn.Conv2d(D * self.F1, self.F2, kernel_size_4)
        self.Batch_normalization_3 = nn.BatchNorm2d(self.F2)
        self.Average_pooling2D_2 = nn.AvgPool2d(kernel_avgpool_2)

        # Calculate output size for the dense layer
        # The formula depends on how pooling reduces the signal length
        final_length = self.input_len // 32  # Approximate reduction from pooling layers

        # layer 4
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(self.F2 * final_length, num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Handle input reshaping
        if len(x.shape) == 3:  # [batch, channels, time]
            # Reshape to [batch, input_dim, channels, time]
            x = x.unsqueeze(1)

        # layer 1
        y = self.Batch_normalization_1(self.conv2d(x))

        # layer 2
        y = self.Batch_normalization_2(self.Depthwise_conv2D(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_1(y))

        # layer 3
        y = self.Separable_conv2D_depth(y)
        y = self.Batch_normalization_3(self.Separable_conv2D_point(y))
        y = self.Elu(y)
        y = self.Dropout(self.Average_pooling2D_2(y))

        # layer 4
        y = self.Flatten(y)
        y = self.Dense(y)
        y = self.Softmax(y)

        return y
