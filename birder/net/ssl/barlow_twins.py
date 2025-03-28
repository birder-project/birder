"""
Barlow Twins, adapted from
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

Paper "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", https://arxiv.org/abs/2103.03230
"""

# Reference license: MIT

import torch
from torch import nn

from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # Return a flattened view of the off-diagonal elements of a square matrix
    (n, _) = x.size()
    # assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(SSLBaseNet):
    default_size = (224, 224)

    def __init__(self, input_channels: int, backbone: BaseNet, sizes: list[int], off_lambda: float) -> None:
        super().__init__(input_channels)
        self.backbone = backbone
        self.off_lambda = off_lambda
        sizes = [self.backbone.embedding_size] + sizes

        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    # pylint: disable=arguments-differ
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        z1 = self.projector(self.backbone.embedding(x1))
        z2 = self.projector(self.backbone.embedding(x2))

        # Cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(x1.size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.off_lambda * off_diag

        return loss
