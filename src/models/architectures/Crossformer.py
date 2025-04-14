from math import ceil

import torch
import torch.nn as nn
from einops import rearrange

from src.models.architectures.layers.Crossformer_EncDec import (
    Decoder,
    DecoderLayer,
    Encoder,
    scale_block,
)
from src.models.architectures.layers.Embed import PatchEmbedding
from src.models.architectures.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    TwoStageAttentionLayer,
)


class Crossformer(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        pred_len: int = 96,  # default value from tsl run.py
        seg_len: int = 12,  # hard-coded class attribute
        win_size: int = 2,  # hard-coded class attribute
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        factor: int = 1,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        num_class: int = 2,
    ):
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        self.pred_len = pred_len
        self.seg_len = seg_len
        self.win_size = win_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.num_class = num_class
        self.activation = activation

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (self.e_layers - 1))
        )
        self.head_nf = self.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            self.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - self.seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, self.d_model)
        )
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    1 if l == 0 else self.win_size,
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    1,
                    self.dropout,
                    self.in_seg_num
                    if l == 0
                    else ceil(self.in_seg_num / self.win_size**l),
                    self.factor,
                )
                for l in range(self.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, self.enc_in, (self.pad_out_len // self.seg_len), self.d_model
            )
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        (self.pad_out_len // self.seg_len),
                        self.factor,
                        self.d_model,
                        self.n_heads,
                        self.d_ff,
                        self.dropout,
                    ),
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
                    self.seg_len,
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    # activation=self.activation,
                )
                for l in range(self.e_layers + 1)
            ],
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.head_nf * self.enc_in, self.num_class)

    def forward(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output
