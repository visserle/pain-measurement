import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.Embed import DataEmbedding
from src.models.architectures.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from src.models.architectures.layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)


class Transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
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
        num_class: int = 2,
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
        self.num_class = num_class
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
                        FullAttention(
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

    def forward(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output  # [B, N]
