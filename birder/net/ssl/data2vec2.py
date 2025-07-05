"""
data2vec 2.0, adapted from
https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py

Paper "Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language",
https://arxiv.org/abs/2212.07525
"""

# Reference license: MIT

import torch
from torch import nn

from birder.layers import LayerNorm2d


class Decoder2d(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, num_layers: int, H: int, W: int) -> None:
        super().__init__()

        self.H = H
        self.W = W

        self.blocks = nn.ModuleList()
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16),
                LayerNorm2d(embed_dim),
                nn.GELU(),
            )
        )
        for _ in range(num_layers - 1):
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16),
                    LayerNorm2d(embed_dim),
                    nn.GELU(),
                )
            )

        self.proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, _, C) = x.size()  # B, N, C

        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)

        residual = x
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i > 0:
                x = x + residual

            residual = x

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)

        return x
