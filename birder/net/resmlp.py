"""
ResMLP, adapted from
https://github.com/facebookresearch/deit/blob/main/resmlp_models.py

Paper "ResMLP: Feedforward networks for image classification with data-efficient training",
https://arxiv.org/abs/2105.03404
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import MLP
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.cait import PatchEmbed


class Affine(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class LayerScaleMLP(nn.Module):
    def __init__(self, dim: int, num_patches: int, drop: float, drop_path: float, init_value: float) -> None:
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = StochasticDepth(drop_path, mode="row")
        self.norm2 = Affine(dim)
        self.mlp = MLP(dim, [int(dim * 4.0), dim], activation_layer=nn.GELU, dropout=drop)
        self.gamma_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class ResMLP(BaseNet):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        init_value = 1e-4
        drop_rate = 0.0
        embed_dim: int = self.config["embed_dim"]
        depth: int = self.config["depth"]
        patch_size: tuple[int, int] = self.config["patch_size"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.patch_embed = PatchEmbed(self.input_channels, embed_dim, patch_size)
        num_patches = (self.size[0] // patch_size[0]) * (self.size[1] // patch_size[1])

        dpr = [drop_path_rate for _ in range(depth)]
        blocks = []
        for i in range(depth):
            blocks.append(
                LayerScaleMLP(
                    embed_dim, num_patches=num_patches, drop=drop_rate, drop_path=dpr[i], init_value=init_value
                )
            )

        blocks.append(Affine(embed_dim))

        self.body = nn.Sequential(*blocks)
        self.features = nn.Sequential(
            Permute([0, 2, 1]),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(1),
        )
        self.embedding_size = embed_dim
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)

    def set_dynamic_size(self, dynamic_size: bool = True) -> None:
        assert dynamic_size is False, "Dynamic size not supported for this network"

    def adjust_size(self, new_size: tuple[int, int]) -> None:
        if new_size == self.size:
            return

        raise RuntimeError("Model resizing not supported")


registry.register_model_config(
    "resmlp_12",
    ResMLP,
    config={"embed_dim": 384, "depth": 12, "patch_size": (16, 16), "init_value": 0.1, "drop_path_rate": 0.0},
)
registry.register_model_config(
    "resmlp_24",
    ResMLP,
    config={"embed_dim": 384, "depth": 24, "patch_size": (16, 16), "init_value": 1e-5, "drop_path_rate": 0.1},
)
registry.register_model_config(
    "resmlp_36",
    ResMLP,
    config={"embed_dim": 384, "depth": 36, "patch_size": (16, 16), "init_value": 1e-6, "drop_path_rate": 0.2},
)
registry.register_model_config(
    "resmlp_24_big",
    ResMLP,
    config={"embed_dim": 768, "depth": 24, "patch_size": (8, 8), "init_value": 1e-6, "drop_path_rate": 0.2},
)
