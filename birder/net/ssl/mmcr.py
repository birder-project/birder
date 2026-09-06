"""
Maximum Manifold Capacity Representations (MMCR), adapted from
https://github.com/ThomasYerxa/mmcr/blob/master/mmcr/imagenet/models.py

Paper "Learning Efficient Coding of Natural Images with Maximum Manifold Capacity Representations",
https://arxiv.org/abs/2303.03307
"""

# Reference license: MIT

import copy
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


class MMCRMomentumLoss(nn.Module):
    def __init__(self, lambda_coeff: float, n_aug: int) -> None:
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.n_aug = n_aug

    def forward(self, z: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=-1)
        z_m = F.normalize(z_m, dim=-1)

        z_local_ = z.reshape(z.size(0) // self.n_aug, self.n_aug, -1).permute(0, 2, 1).contiguous()
        z_local_m = z_m.reshape(z_m.size(0) // self.n_aug, self.n_aug, -1).permute(0, 2, 1).contiguous()

        # Gather across devices into list
        if training_utils.is_dist_available_and_initialized() is True:
            z_list = [torch.zeros_like(z_local_) for _ in range(dist.get_world_size())]
            dist.all_gather(z_list, z_local_, async_op=False)
            z_list[dist.get_rank()] = z_local_

            # Gather momentum outputs
            z_m_list = [torch.zeros_like(z_local_m) for _ in range(dist.get_world_size())]
            dist.all_gather(z_m_list, z_local_m, async_op=False)
            z_m_list[dist.get_rank()] = z_local_m

            # Append all
            z_local = torch.concat(z_list, dim=0)
            z_m_local = torch.concat(z_m_list, dim=0)

        else:
            z_local = z_local_
            z_m_local = z_local_m

        if self.lambda_coeff == 0:
            local_nuc = 0
        else:
            local_nuc = torch.linalg.svdvals(z_local).sum()  # pylint: disable=not-callable

        centroids = (torch.mean(z_local, dim=-1) + torch.mean(z_m_local, dim=-1)) * 0.5

        # Filter infs and NaNs
        selected = centroids.isfinite().all(dim=1)
        centroids = centroids[selected]

        if selected.sum() != centroids.size(0):
            # NaN filtered
            pass

        global_nuc = torch.linalg.svdvals(centroids).sum()  # pylint: disable=not-callable

        loss = -1.0 * global_nuc + self.lambda_coeff * local_nuc / z_local.size(0)

        return loss


class MMCREncoder(nn.Module):
    def __init__(self, backbone: BaseNet, projector_dims: list[int]) -> None:
        super().__init__()
        self.backbone = backbone
        sizes = [self.backbone.embedding_size] + projector_dims

        # Projection head
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        *,
        return_moe_training_output: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
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

        embedding = torch.flatten(embedding, start_dim=1)
        embedding = self.projector(embedding)

        if moe_training_output is not None:
            return (embedding, moe_training_output)

        return embedding


class MMCR(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"

        projector_dims: list[int] = self.config["projector_dims"]

        momentum_backbone = copy.deepcopy(self.backbone)

        encoder = MMCREncoder(self.backbone, projector_dims=projector_dims)
        momentum_encoder = MMCREncoder(momentum_backbone, projector_dims=projector_dims)

        self.encoder = encoder
        self.momentum_encoder = momentum_encoder

        # Weights initialization
        self.momentum_encoder.load_state_dict(self.encoder.state_dict())

    def forward(
        self,
        x: torch.Tensor | list[torch.Tensor],
        *,
        grid_sizes: Optional[list[torch.Tensor]] = None,
        valid_masks: Optional[list[torch.Tensor]] = None,
        return_moe_training_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, MoETrainingOutputType]:
        moe_training_output: Optional[MoETrainingOutputType] = None
        if grid_sizes is None:
            assert isinstance(x, torch.Tensor)

            C, H, W = x.shape[-3:]  # B, num_views, C, H, W
            x = x.reshape(-1, C, H, W)
            if return_moe_training_output is True:
                z, moe_training_output = self.encoder(x, return_moe_training_output=True)
            else:
                z = self.encoder(x)

            with torch.no_grad():
                z_m = self.momentum_encoder(x)

        else:
            assert isinstance(x, list)

            z_views: list[torch.Tensor] = []
            z_m_views: list[torch.Tensor] = []
            for view, view_grid_sizes, view_valid_mask in zip(
                x, grid_sizes, valid_masks, strict=True  # type: ignore[arg-type]
            ):
                if return_moe_training_output is True:
                    view_z, view_moe_training_output = self.encoder(
                        view, grid_sizes=view_grid_sizes, valid_mask=view_valid_mask, return_moe_training_output=True
                    )
                    moe_training_output = combine_moe_training_outputs(moe_training_output, view_moe_training_output)
                else:
                    view_z = self.encoder(view, grid_sizes=view_grid_sizes, valid_mask=view_valid_mask)

                with torch.no_grad():
                    view_z_m = self.momentum_encoder(view, grid_sizes=view_grid_sizes, valid_mask=view_valid_mask)

                z_views.append(view_z)
                z_m_views.append(view_z_m)

            z = torch.stack(z_views, dim=1).flatten(0, 1)
            z_m = torch.stack(z_m_views, dim=1).flatten(0, 1)

        if moe_training_output is not None:
            return (z, z_m, moe_training_output)

        return (z, z_m)
