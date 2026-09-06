"""
VICReg, adapted from
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

Paper "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning",
https://arxiv.org/abs/2105.04906
"""

# Reference license: MIT

from typing import Any
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.layers.moe import MoETrainingOutputType
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet
from birder.net.ssl.base import combine_moe_training_outputs


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    # assert x.size(0) == x.size(1)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class FullGatherLayer(torch.autograd.Function):
    # pylint: disable=abstract-method,arguments-differ

    @staticmethod
    def forward(_ctx, x):  # type: ignore
        if training_utils.is_dist_available_and_initialized() is False:
            return (x,)

        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(_ctx, *grads):  # type: ignore
        if training_utils.is_dist_available_and_initialized() is False:
            return grads[0]

        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class VICReg(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"

        mlp_dim: int = self.config["mlp_dim"]
        sim_coeff: float = self.config["sim_coeff"]
        std_coeff: float = self.config["std_coeff"]
        cov_coeff: float = self.config["cov_coeff"]
        sync_batches: bool = self.config.get("sync_batches", True)

        self.num_features = mlp_dim
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.sync_batches = sync_batches

        self.projector = nn.Sequential(
            nn.Linear(self.backbone.embedding_size, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim, bias=False),
        )

    def _backbone_embedding(
        self,
        x: torch.Tensor,
        grid_sizes: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        return_moe_training_output: bool,
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

        return (embedding, moe_training_output)

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        grid_sizes1: Optional[torch.Tensor] = None,
        valid_mask1: Optional[torch.Tensor] = None,
        grid_sizes2: Optional[torch.Tensor] = None,
        valid_mask2: Optional[torch.Tensor] = None,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        # pylint: disable=arguments-differ

        x, moe_training_output = self._backbone_embedding(x, grid_sizes1, valid_mask1, return_moe_training_output)
        y, y_moe_training_output = self._backbone_embedding(y, grid_sizes2, valid_mask2, return_moe_training_output)
        if y_moe_training_output is not None:
            moe_training_output = combine_moe_training_outputs(moe_training_output, y_moe_training_output)

        x = self.projector(x)
        y = self.projector(y)

        repr_loss = F.mse_loss(x, y)

        if self.sync_batches is True:
            x = torch.concat(FullGatherLayer.apply(x), dim=0)
            y = torch.concat(FullGatherLayer.apply(y), dim=0)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (y.size(0) - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(
            self.num_features
        )

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        if moe_training_output is not None:
            return (loss, moe_training_output)

        return loss
