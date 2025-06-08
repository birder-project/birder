"""
Barlow Twins, adapted from
https://github.com/facebookresearch/barlowtwins/blob/main/main.py

Paper "Barlow Twins: Self-Supervised Learning via Redundancy Reduction", https://arxiv.org/abs/2103.03230
"""

# Reference license: MIT

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn

from birder.common import training_utils
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # Return a flattened view of the off-diagonal elements of a square matrix
    (n, _) = x.size()
    # assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"

        projector_sizes: list[int] = self.config["projector_sizes"]
        self.off_lambda: float = self.config["off_lambda"]

        sizes = [self.backbone.embedding_size] + projector_sizes
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
        if training_utils.is_dist_available_and_initialized() is True:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        z1 = self.projector(self.backbone.embedding(x1))
        z2 = self.projector(self.backbone.embedding(x2))

        # Cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c = c / (x1.size(0) * world_size)
        if training_utils.is_dist_available_and_initialized() is True:
            dist.nn.all_reduce(c)  # https://github.com/pytorch/pytorch/issues/58005#issuecomment-1778029156

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.off_lambda * off_diag

        return loss
