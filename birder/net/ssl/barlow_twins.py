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
import torch.distributed._functional_collectives as funcol
from torch import nn

from birder.common import training_utils
from birder.layers.moe import MoETrainingOutputType
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet
from birder.net.ssl.base import combine_moe_training_outputs


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # Return a flattened view of the off-diagonal elements of a square matrix
    n, _ = x.size()
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

    def _project(
        self,
        x: torch.Tensor,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> tuple[torch.Tensor, Optional[MoETrainingOutputType]]:
        moe_training_output: Optional[MoETrainingOutputType] = None
        if grid_sizes is None:
            if return_moe_training_output is True:
                features, moe_training_output = self.backbone.forward_features(
                    x, return_moe_training_output=True  # type: ignore[call-arg]
                )
            else:
                features = self.backbone.forward_features(x)
        else:
            if return_moe_training_output is True:
                features, moe_training_output = self.backbone.forward_features(  # type: ignore[call-arg]
                    x, grid_sizes=grid_sizes, valid_mask=valid_mask, return_moe_training_output=True
                )
            else:
                features = self.backbone.forward_features(  # type: ignore[call-arg]
                    x, grid_sizes=grid_sizes, valid_mask=valid_mask
                )

        if grid_sizes is None:
            embedding = self.backbone.embedding_from_features(features)
        else:
            embedding = self.backbone.embedding_from_features(features, valid_mask)  # type: ignore[call-arg]

        return (self.projector(embedding), moe_training_output)

    # pylint: disable-next=arguments-differ
    def forward(  # type: ignore[override]
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        grid_sizes1: Optional[torch.Tensor] = None,
        valid_mask1: Optional[torch.Tensor] = None,
        grid_sizes2: Optional[torch.Tensor] = None,
        valid_mask2: Optional[torch.Tensor] = None,
        *,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        world_size = training_utils.get_world_size()

        z1, moe_training_output = self._project(x1, grid_sizes1, valid_mask1, return_moe_training_output)
        z2, z2_moe_training_output = self._project(x2, grid_sizes2, valid_mask2, return_moe_training_output)
        if z2_moe_training_output is not None:
            moe_training_output = combine_moe_training_outputs(moe_training_output, z2_moe_training_output)

        # Cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c = c / (x1.size(0) * world_size)
        if training_utils.is_dist_available_and_initialized() is True:
            c = funcol.all_reduce(c, reduceOp="sum", group=dist.group.WORLD)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.off_lambda * off_diag

        if moe_training_output is not None:
            return (loss, moe_training_output)

        return loss
