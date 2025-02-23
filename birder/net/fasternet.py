"""
FasterNet, adapted from
https://github.com/JierunChen/FasterNet/blob/master/models/fasternet.py

Paper "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks",
https://arxiv.org/abs/2303.03667

Changes from original:
* No extra norm's for detection
"""

# Reference license: MIT

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import get_activation_module


class PartialConv(nn.Module):
    def __init__(self, dim: int, n_div: int) -> None:
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x1, x2) = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.concat((x1, x2), dim=1)

        return x


class MLPBlock(nn.Module):
    def __init__(
        self, dim: int, n_div: int, mlp_ratio: float, drop_path: float, act_layer: Callable[..., nn.Module]
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.spatial_mixing = PartialConv(dim, n_div)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            nn.BatchNorm2d(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
        )
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, in_channels: int, embed_dim: int, patch_size: tuple[int, int], patch_stride: tuple[int, int]
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=(0, 0), bias=False
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, dim: int, patch_size: tuple[int, int], patch_stride: tuple[int, int]) -> None:
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=patch_size, stride=patch_stride, padding=(0, 0), bias=False
        )
        self.norm = nn.BatchNorm2d(2 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduction(x)
        x = self.norm(x)

        return x


class FasterNetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        n_div: int,
        mlp_ratio: float,
        drop_path: list[int],
        act_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(
                MLPBlock(dim=dim, n_div=n_div, mlp_ratio=mlp_ratio, drop_path=drop_path[i], act_layer=act_layer)
            )

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class FasterNet(DetectorBackbone):
    block_group_regex = r"body\.stage\d+\.(\d+)"

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

        patch_size = (4, 4)
        patch_stride = (4, 4)
        n_div = 4
        mlp_ratio = 2.0
        feature_dim = 1280
        depths: list[int] = self.config["depths"]
        embed_dim: int = self.config["embed_dim"]
        act_layer_name: str = self.config["act_layer_name"]
        drop_path_rate: float = self.config["drop_path_rate"]

        act_layer = get_activation_module(act_layer_name)

        self.stem = PatchEmbed(
            in_channels=self.input_channels, embed_dim=embed_dim, patch_size=patch_size, patch_stride=patch_stride
        )

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        num_stages = len(depths)
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        layers = []
        for i in range(num_stages):
            layers.append(
                FasterNetBlock(
                    dim=int(embed_dim * 2**i),
                    depth=depths[i],
                    n_div=n_div,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    act_layer=act_layer,
                )
            )
            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(int(embed_dim * 2**i))
            layers = []
            if i < num_stages - 1:
                layers.append(PatchMerging(dim=int(embed_dim * 2**i), patch_size=(2, 2), patch_stride=(2, 2)))

        num_features = int(embed_dim * 2 ** (num_stages - 1))
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(num_features, feature_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            act_layer(),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = feature_dim
        self.classifier = self.create_classifier()

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)

        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for param in self.stem.parameters():
            param.requires_grad = False

        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias(
    "fasternet_t0",
    FasterNet,
    config={"depths": [1, 2, 8, 2], "embed_dim": 40, "act_layer_name": "gelu", "drop_path_rate": 0.0},
)
registry.register_alias(
    "fasternet_t1",
    FasterNet,
    config={"depths": [1, 2, 8, 2], "embed_dim": 64, "act_layer_name": "gelu", "drop_path_rate": 0.02},
)
registry.register_alias(
    "fasternet_t2",
    FasterNet,
    config={"depths": [1, 2, 8, 2], "embed_dim": 96, "act_layer_name": "relu", "drop_path_rate": 0.05},
)
registry.register_alias(
    "fasternet_s",
    FasterNet,
    config={"depths": [1, 2, 13, 2], "embed_dim": 128, "act_layer_name": "relu", "drop_path_rate": 0.1},
)
registry.register_alias(
    "fasternet_m",
    FasterNet,
    config={"depths": [3, 4, 18, 3], "embed_dim": 144, "act_layer_name": "relu", "drop_path_rate": 0.2},
)
registry.register_alias(
    "fasternet_l",
    FasterNet,
    config={"depths": [3, 4, 18, 3], "embed_dim": 192, "act_layer_name": "relu", "drop_path_rate": 0.3},
)

registry.register_weights(
    "fasternet_t0_il-common",
    {
        "description": "FasterNet T0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 11.9,
                "sha256": "958bcddf6d5d272490972e7a5a1f77f0d39d65a3a9ffdb59c4b83c2d3562a2a9",
            }
        },
        "net": {"network": "fasternet_t0", "tag": "il-common"},
    },
)
registry.register_weights(
    "fasternet_t1_il-common",
    {
        "description": "FasterNet T1 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 26.0,
                "sha256": "6d7bce44e9eb02362290e9c46a80ac324176146727c16c753d2dcd0e039d155b",
            }
        },
        "net": {"network": "fasternet_t1", "tag": "il-common"},
    },
)
