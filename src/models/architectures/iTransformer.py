import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.architectures.layers.Embed import DataEmbedding_inverted
from src.models.architectures.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from src.models.architectures.layers.Transformer_EncDec import Encoder, EncoderLayer


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        d_model: int = 128,
        e_layers: int = 3,
        d_ff: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
        factor: int = 1,
        activation: nn.Module = nn.GELU(),
        embed: str = "timeF",
        freq: str = "h",
        num_classes: int = 2,
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = dropout
        self.factor = factor
        self.activation = activation
        self.embed = embed
        self.freq = freq
        self.num_class = num_classes

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
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
        # Decoder
        self.act = F.gelu
        self.dropout_layer = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.d_model * self.enc_in, self.num_class)

    def forward(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout_layer(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
