import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.Embed import DataEmbedding
from src.models.architectures.layers.SelfAttention_Family import (
    AttentionLayer,
    DSAttention,
)
from src.models.architectures.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)


class Projector(nn.Module):
    """
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(
        self,
        enc_in,
        seq_len,
        hidden_dims,
        hidden_layers,
        output_dim,
        kernel_size=3,
    ):
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]

        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O

        return y


class NonstationarityTransformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        d_model: int = 128,
        e_layers: int = 3,
        d_ff: int = 256,
        n_heads: int = 8,
        factor: int = 1,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        c_out: int = 2,
        num_class: int = 2,
        hidden_dims: list = [256, 256],
        hidden_layers: int = 2,
        embed: str = "fixed",
        freq: str = "h",
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.factor = factor
        self.dropout = dropout
        self.activation = activation
        self.c_out = c_out
        self.num_class = num_class
        self.p_hidden_dims = hidden_dims
        self.p_hidden_layers = hidden_layers
        self.embed = embed
        self.freq = freq

        # Embedding
        self.enc_embedding = DataEmbedding(
            self.enc_in,
            self.d_model,
            self.embed,
            self.freq,
            self.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model),
        )

        # No decoder needed for classification tasks, only needed for prediction tasks

        self.act = F.gelu
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.seq_len, self.num_class)

        self.tau_learner = Projector(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            hidden_dims=self.p_hidden_dims,
            hidden_layers=self.p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            hidden_dims=self.p_hidden_dims,
            hidden_layers=self.p_hidden_layers,
            output_dim=self.seq_len,
        )

    def forward(self, x_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output
