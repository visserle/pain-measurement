import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.Autoformer_EncDec import series_decomp
from src.models.architectures.layers.Embed import DataEmbedding_wo_pos
from src.models.architectures.layers.StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(
        self,
        seq_len,
        down_sampling_window,
        down_sampling_layers,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        # Down-sampling layers
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window**i),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(self.down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(
        self,
        seq_len,
        down_sampling_window,
        down_sampling_layers,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window ** (i + 1)),
                        self.seq_len // (self.down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        self.seq_len // (self.down_sampling_window**i),
                        self.seq_len // (self.down_sampling_window**i),
                    ),
                )
                for i in reversed(range(self.down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        seq_len,
        down_sampling_window,
        d_model,
        dropout,
        channel_independence,
        moving_avg,
        top_k,
        d_ff,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.d_model = d_model
        self.dropout = dropout
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.top_k = top_k

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        self.channel_independence = channel_independence

        if self.decomp_method == "moving_avg":
            self.decompsition = series_decomp(self.moving_avg)
        elif self.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(self.top_k)
        else:
            raise ValueError("decompsition is error")

        if not self.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=self.d_model, out_features=self.d_ff),
                nn.GELU(),
                nn.Linear(in_features=self.d_ff, out_features=self.d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            self.seq_len, self.down_sampling_window, self.down_sampling_layers
        )

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            self.seq_len, self.down_sampling_window, self.down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_ff),
            nn.GELU(),
            nn.Linear(in_features=self.d_ff, out_features=self.d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):
    def __init__(
        self,
        input_len,
        input_dim,
        label_len,
        down_sampling_window,
        channel_independence,
        down_sampling_layers,
        down_sampling_method,
        moving_avg,
        d_model,
        e_layers,
        dropout,
        use_norm,
        top_k,
        d_ff,
        freq="s",
        embed="timeF",
        num_class=2,
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.label_len = label_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_method = down_sampling_method
        self.moving_avg = moving_avg
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.top_k = top_k
        self.dropout = dropout
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    self.seq_len,
                    self.down_sampling_window,
                    self.d_model,
                    self.dropout,
                    self.channel_independence,
                    self.moving_avg,
                    self.top_k,
                    self.d_ff,
                )
                for _ in range(self.e_layers)
            ]
        )
        self.use_norm = use_norm
        self.freq = freq
        self.embed = embed

        self.preprocess = series_decomp(self.moving_avg)

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                1,
                self.d_model,
                self.embed,
                self.freq,
                self.dropout,
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                self.enc_in,
                self.d_model,
                self.embed,
                self.freq,
                self.dropout,
            )

        self.layer = self.e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(
                    self.enc_in,
                    affine=True,
                    non_norm=True if self.use_norm == 0 else False,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, num_class)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(
                self.down_sampling_window, return_indices=False
            )
        elif self.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(
                    x_mark_enc_mark_ori[:, :: self.down_sampling_window, :]
                )
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[
                    :, :: self.down_sampling_window, :
                ]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forward(self, x_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
