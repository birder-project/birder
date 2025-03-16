"""
BYOL, adapted from
https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py

Paper "Bootstrap your own latent: A new approach to self-supervised Learning",
https://arxiv.org/abs/2006.07733
"""

# Reference license: MIT

import torch
import torch.nn.functional as F
from torch import nn

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
        self.fc2 = nn.Linear(hidden_features, out_features)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.embedding(x)
        return self.projector(x)


class BYOL(SSLBaseNet):
    default_size = (224, 224)

    def __init__(
        self,
        input_channels: int,
        backbone: BYOLEncoder,
        target_encoder: BYOLEncoder,
        projection_size: int,
        projection_hidden_size: int,
    ) -> None:
        super().__init__(input_channels)
        self.backbone = backbone
        self.target_encoder = target_encoder
        self.online_predictor = MLP(projection_size, projection_hidden_size, projection_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = self.backbone(x)
        online_predictions = self.online_predictor(projection)
        (online_pred_one, online_pred_two) = online_predictions.chunk(2, dim=0)

        target_projections = self.target_encoder(x)
        (target_proj_one, target_proj_two) = target_projections.chunk(2, dim=0)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        loss = loss_one + loss_two

        return loss.mean()
