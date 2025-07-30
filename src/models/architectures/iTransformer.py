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
        pred_len: int = 96,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 2048,
        factor: int = 1,
        dropout: float = 0.1,
        embed: str = "timeF",
        freq: str = "h",
        activation: str = "gelu",
        num_class: int = 2,
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.pred_len = pred_len

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            self.seq_len,
            d_model,
            embed,
            freq,
            dropout,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * self.enc_in, num_class)

    def forward(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, c_in * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
