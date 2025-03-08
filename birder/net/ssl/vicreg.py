"""
VICReg, adapted from
https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

Paper "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning",
https://arxiv.org/abs/2105.04906
"""

# Reference license: MIT

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from birder.common import training_utils
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    # assert x.size(0) == x.size(1)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# pylint: disable=abstract-method,arguments-differ
class FullGatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(_ctx, x):  # type: ignore
        if training_utils.is_dist_available_and_initialized() is False:
            return (x,)

        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(_ctx, *grads):  # type: ignore
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


class VICReg(SSLBaseNet):
    default_size = (224, 224)

    def __init__(
        self,
        input_channels: int,
        backbone: BaseNet,
        mlp_dim: int,
        batch_size: int,
        sim_coeff: float,
        std_coeff: float,
        cov_coeff: float,
        sync_batches: bool = False,
    ) -> None:
        super().__init__(input_channels)
        self.backbone = backbone
        self.num_features = mlp_dim
        self.world_batch_size = batch_size * training_utils.get_world_size()
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.projector(self.backbone.embedding(x))
        y = self.projector(self.backbone.embedding(y))

        repr_loss = F.mse_loss(x, y)

        if self.sync_batches is True:
            x = torch.concat(FullGatherLayer.apply(x), dim=0)
            y = torch.concat(FullGatherLayer.apply(y), dim=0)

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.world_batch_size - 1)
        cov_y = (y.T @ y) / (self.world_batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(
            self.num_features
        )

        loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        return loss
