import torch
from torch import nn

from src.models.architectures.layers.Embed import PatchEmbedding
from src.models.architectures.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
)
from src.models.architectures.layers.Transformer_EncDec import Encoder, EncoderLayer


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(
        self,
        input_len: int,
        input_dim: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        e_layers: int = 3,
        d_ff: int = 256,
        top_k: int = 3,
        dropout: float = 0.1,
        factor: int = 1,
        activation: nn.Module = nn.GELU(),
        num_classes: int = 2,
    ):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = input_len
        self.enc_in = input_dim
        padding = stride
        self.patch_len = patch_len
        self.d_model = d_model
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.top_k = top_k
        self.dropout = dropout
        self.factor = factor
        self.n_heads = 8
        self.num_class = num_classes
        self.activation = activation

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            self.d_model, self.patch_len, stride, padding, self.dropout
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
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)
            ),
        )

        # Prediction Head
        self.head_nf = self.d_model * int((self.seq_len - patch_len) / stride + 2)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(self.dropout)
        self.projection = nn.Linear(self.head_nf * self.enc_in, self.num_class)

    def forward(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output
