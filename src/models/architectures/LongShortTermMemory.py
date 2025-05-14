import torch.nn as nn


class LongShortTermMemory(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        input_len: int,  # not used (see below)
        num_classes: int = 2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
        # note that the input_size kw from pytorch is the number of features in the
        # input, not the length of the input
        # (LSTMs are recurrent networks that can process sequences of variable length)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
