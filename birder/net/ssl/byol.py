"""
BYOL, adapted from
https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
and
https://github.com/google-deepmind/deepmind-research/blob/master/byol/utils/networks.py

Paper "Bootstrap your own latent: A new approach to self-supervised Learning",
https://arxiv.org/abs/2006.07733
"""

# Reference license: MIT and Apache-2.0

import copy
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from birder.layers.moe import MoETrainingOutputType
from birder.net.base import BaseNet
from birder.net.ssl.base import SSLBaseNet


def loss_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = nn.BatchNorm1d(hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class BYOLEncoder(nn.Module):
    def __init__(self, backbone: BaseNet, projection_size: int, projection_hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(backbone.embedding_size, projection_hidden_size, projection_size)

    def forward(
        self, x: torch.Tensor, *, return_moe_training_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        moe_training_output: Optional[MoETrainingOutputType] = None
        if return_moe_training_output is True:
            features, moe_training_output = self.backbone.forward_features(
                x, return_moe_training_output=True  # type: ignore[call-arg]
            )
        else:
            features = self.backbone.forward_features(x)

        embedding = self.backbone.embedding_from_features(features)
        projection = self.projector(embedding)
        if moe_training_output is not None:
            return (projection, moe_training_output)

        return projection


class BYOL(SSLBaseNet):
    def __init__(
        self,
        backbone: BaseNet,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(backbone, config=config, size=size)
        assert self.config is not None, "must set config"

        projection_size: int = self.config["projection_size"]
        projection_hidden_size: int = self.config["projection_hidden_size"]

        target_encoder_backbone = copy.deepcopy(self.backbone)

        self.online_encoder = BYOLEncoder(self.backbone, projection_size, projection_hidden_size)
        self.target_encoder = BYOLEncoder(target_encoder_backbone, projection_size, projection_hidden_size)
        self.online_predictor = MLP(projection_size, projection_hidden_size, projection_size)

        # Weights initialization
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())

    def forward(
        self, x: torch.Tensor, *, return_moe_training_output: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, MoETrainingOutputType]:
        moe_training_output: Optional[MoETrainingOutputType] = None
        if return_moe_training_output is True:
            projection, moe_training_output = self.online_encoder(x, return_moe_training_output=True)
        else:
            projection = self.online_encoder(x)

        online_predictions = self.online_predictor(projection)
        online_pred_one, online_pred_two = online_predictions.chunk(2, dim=0)

        with torch.no_grad():
            target_projections = self.target_encoder(x)
            target_proj_one, target_proj_two = target_projections.chunk(2, dim=0)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two

        if moe_training_output is not None:
            return (loss.mean(), moe_training_output)

        return loss.mean()
