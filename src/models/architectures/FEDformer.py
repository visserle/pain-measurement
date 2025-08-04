import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.AutoCorrelation import AutoCorrelationLayer
from src.models.architectures.layers.Autoformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
)
from src.models.architectures.layers.Embed import DataEmbedding
from src.models.architectures.layers.FourierCorrelation import (
    FourierBlock,
    FourierCrossAttention,
)
from src.models.architectures.layers.MultiWaveletCorrelation import (
    MultiWaveletCross,
    MultiWaveletTransform,
)


class FEDformer(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
    """

    def __init__(
        self,
        input_len,
        input_dim,
        d_model,
        dropout,
        d_ff,
        e_layers,
        d_layers,
        pred_len=96,
        moving_avg=25,
        n_heads=8,
        activation="gelu",
        embed="timeF",
        freq="s",
        c_out=7,
        version="fourier",
        mode_select="random",
        modes=32,
        num_class=2,
    ):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.pred_len = pred_len
        self.dec_in = input_dim
        self.d_model = d_model
        self.c_out = c_out
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.e_layers = e_layers
        self.moving_avg = moving_avg
        self.activation = activation
        self.embed = embed
        self.freq = freq
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_class = num_class

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        # Decomp
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )
        self.dec_embedding = DataEmbedding(
            self.dec_in,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )

        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=1, base="legendre"
            )
            decoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=1, base="legendre"
            )
            decoder_cross_att = MultiWaveletCross(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=self.d_model,
                base="legendre",
                activation="tanh",
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                n_heads=self.n_heads,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                n_heads=self.n_heads,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=self.n_heads,
            )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, self.d_model, self.n_heads),
                    AutoCorrelationLayer(decoder_cross_att, self.d_model, self.n_heads),
                    self.d_model,
                    self.c_out,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, self.c_out, bias=True),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, self.num_class)

    def forward(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
