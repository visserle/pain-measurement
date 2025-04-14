import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.Conv_Blocks import Inception_Block_V1
from src.models.architectures.layers.Embed import DataEmbedding


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels

        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (((self.seq_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq

    Parameters:
    -----------
    input_len : int
        Length of the input time series sequence.
    input_dim : int
        Number of features/variables in the input data.
    d_model : int, default=16
        The dimension of the model's internal representation.
    n_heads : int, default=8
        Number of attention heads for internal operations.
    e_layers : int, default=2
        Number of TimesBlock layers to stack.
    d_ff : int, default=32
        Dimension of the feed-forward network in each TimesBlock.
    d_conv : int, default=4
        Kernel size for convolution operations.
    dropout : float, default=0.1
        Dropout rate for regularization.
    embed : str, default='timeF'
        Type of time embedding to use. Options: 'timeF', 'fixed', 'learned'.
    freq : str, default='s'
        Frequency of the time series for temporal encoding. Options include:
        's' (second), 't' (minute), 'h' (hour), 'd' (day), etc.
    top_k : int, default=3
        Number of top periods to extract from FFT for time-to-2D reshaping.
    num_kernels : int, default=6
        Number of inception kernels in each TimesBlock.
    num_classes : int, default=2
        Number of output classes for classification tasks.
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        d_model: int = 16,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 32,
        d_conv: int = 4,
        dropout: float = 0.1,
        embed: str = "timeF",  # TODO: try also "fixed"
        freq: str = "s",  # TODO: try h for hourly and t for minutely
        top_k: int = 3,
        num_kernels: int = 6,
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.embed = embed
        self.freq = freq
        self.top_k = top_k
        self.num_kernels = num_kernels
        self.num_class = num_classes

        self.model = nn.ModuleList(
            [
                TimesBlock(
                    self.seq_len,
                    self.top_k,
                    self.d_model,
                    self.d_ff,
                    self.num_kernels,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )
        self.layer = self.e_layers
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, self.num_class)

    def forward(self, x_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
