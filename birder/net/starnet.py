"""
StarNet, adapted from
https://github.com/ma-xu/Rewrite-the-Stars/blob/main/imagenet/starnet.py

Paper "Rewrite the Stars", https://arxiv.org/abs/2403.19967

Changes from original:
* Removed bias term from conv before batchnorm
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class StarNetBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop_path: float) -> None:
        super().__init__()
        self.dwconv1 = Conv2dNormActivation(
            dim, dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=dim, activation_layer=None
        )
        self.f1 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.f2 = nn.Conv2d(dim, int(mlp_ratio * dim), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.g = Conv2dNormActivation(
            int(mlp_ratio * dim), dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), activation_layer=None
        )
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv1(x)
        x1 = self.f1(x)
        x2 = self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = identity + self.drop_path(x)

        return x


class StarNetStage(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, depth: int, mlp_ratio: float, drop_path: list[float]) -> None:
        super().__init__()

        self.downsample = Conv2dNormActivation(
            in_channels, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), activation_layer=None
        )
        layers = []
        for i in range(depth):
            layers.append(StarNetBlock(embed_dim, mlp_ratio=mlp_ratio, drop_path=drop_path[i]))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


class StarNet(DetectorBackbone):
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

        first_channels = 32
        base_dim: int = self.config["base_dim"]
        depths: list[int] = self.config["depths"]
        mlp_ratio: float = self.config["mlp_ratio"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            first_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=nn.ReLU6,
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        num_stages = len(depths)

        prev_channels = first_channels
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            embed_dim = base_dim * 2**i
            stages[f"stage{i+1}"] = StarNetStage(
                prev_channels, embed_dim=embed_dim, depth=depths[i], mlp_ratio=mlp_ratio, drop_path=dpr[i]
            )
            return_channels.append(embed_dim)
            prev_channels = embed_dim

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.BatchNorm2d(embed_dim),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dim
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


# Extremely Small Models
registry.register_model_config(
    "starnet_esm05", StarNet, config={"base_dim": 16, "depths": [1, 1, 3, 1], "mlp_ratio": 3.0, "drop_path_rate": 0.0}
)
registry.register_model_config(
    "starnet_esm10", StarNet, config={"base_dim": 20, "depths": [1, 2, 4, 1], "mlp_ratio": 4.0, "drop_path_rate": 0.0}
)
registry.register_model_config(
    "starnet_esm15", StarNet, config={"base_dim": 24, "depths": [1, 2, 4, 2], "mlp_ratio": 3.0, "drop_path_rate": 0.0}
)

# StarNets
registry.register_model_config(
    "starnet_s1", StarNet, config={"base_dim": 24, "depths": [2, 2, 8, 3], "mlp_ratio": 4.0, "drop_path_rate": 0.0}
)
registry.register_model_config(
    "starnet_s2", StarNet, config={"base_dim": 32, "depths": [1, 2, 6, 2], "mlp_ratio": 4.0, "drop_path_rate": 0.0}
)
registry.register_model_config(
    "starnet_s3", StarNet, config={"base_dim": 32, "depths": [2, 2, 8, 4], "mlp_ratio": 4.0, "drop_path_rate": 0.0}
)
registry.register_model_config(
    "starnet_s4", StarNet, config={"base_dim": 32, "depths": [3, 3, 12, 5], "mlp_ratio": 4.0, "drop_path_rate": 0.1}
)

registry.register_weights(
    "starnet_esm10_il-common",
    {
        "description": "StarNet extremely small model 1M trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 3.7,
                "sha256": "74f49aff37993c8cb3d76adba72318a37e09816173505b180bb4353699a4dbdc",
            }
        },
        "net": {"network": "starnet_esm10", "tag": "il-common"},
    },
)
registry.register_weights(
    "starnet_s1_il-common",
    {
        "description": "StarNet S1 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 10.6,
                "sha256": "b997d7ec49c1b6dc1dbc1f12204d34d1fce8114072e7e72b9fe6df989863143b",
            }
        },
        "net": {"network": "starnet_s1", "tag": "il-common"},
    },
)
