"""
VAN (Visual Attention Network), adapted from
https://github.com/Visual-Attention-Network/VAN-Classification/blob/main/models/van.py

Paper "Visual Attention Network", https://arxiv.org/abs/2202.09741
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import StochasticDepth

from birder.layers import LayerNorm2d
from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class DWConvMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=hidden_features,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim,
            dim,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(9, 9),
            dilation=(3, 3),
            groups=dim,
        )
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size: tuple[int, int], stride: tuple[int, int], in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_dim, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)

        return x


class VANBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float, drop_path: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = StochasticDepth(drop_path, mode="row")

        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = DWConvMLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))

        return x


class VANStage(nn.Module):
    def __init__(
        self,
        patch_size: tuple[int, int],
        stride: tuple[int, int],
        in_dim: int,
        embed_dim: int,
        depth: int,
        mlp_ratio: float,
        drop: float,
        drop_path: list[float],
    ) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(patch_size=patch_size, stride=stride, in_dim=in_dim, embed_dim=embed_dim)

        layers = []
        for i in range(depth):
            layers.append(VANBlock(embed_dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i]))

        self.block = nn.Sequential(*layers)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.block(x)
        x = self.norm(x)

        return x


class VAN(DetectorBackbone):
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

        drop_rate = 0.0
        mlp_ratios = [8.0, 8.0, 4.0, 4.0]
        embed_dims: list[int] = self.config["embed_dims"]
        depths: list[int] = self.config["depths"]
        drop_path_rate: float = self.config["drop_path_rate"]

        num_stages = len(depths)
        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i in range(num_stages):
            stages[f"stage{i+1}"] = VANStage(
                patch_size=(7, 7) if i == 0 else (3, 3),
                stride=(4, 4) if i == 0 else (2, 2),
                in_dim=self.input_channels if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                depth=depths[i],
                mlp_ratio=mlp_ratios[i],
                drop=drop_rate,
                drop_path=dpr[i],
            )
            return_channels.append(embed_dims[i])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = embed_dims[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.constant_(m.weight, 1.0)

            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.trunc_normal_(m.weight, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def detection_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.return_stages:
                out[name] = x

        return out

    def freeze_stages(self, up_to_stage: int) -> None:
        for idx, module in enumerate(self.body.children()):
            if idx >= up_to_stage:
                break

            for param in module.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "van_b0", VAN, config={"embed_dims": [32, 64, 160, 256], "depths": [3, 3, 5, 2], "drop_path_rate": 0.1}
)
registry.register_model_config(
    "van_b1", VAN, config={"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 4, 2], "drop_path_rate": 0.1}
)
registry.register_model_config(
    "van_b2", VAN, config={"embed_dims": [64, 128, 320, 512], "depths": [3, 3, 12, 3], "drop_path_rate": 0.1}
)
registry.register_model_config(
    "van_b3", VAN, config={"embed_dims": [64, 128, 320, 512], "depths": [3, 5, 27, 3], "drop_path_rate": 0.2}
)
registry.register_model_config(
    "van_b4", VAN, config={"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "drop_path_rate": 0.3}
)
registry.register_model_config(
    "van_b5", VAN, config={"embed_dims": [96, 192, 480, 768], "depths": [3, 3, 24, 3], "drop_path_rate": 0.3}
)
registry.register_model_config(
    "van_b6", VAN, config={"embed_dims": [96, 192, 384, 768], "depths": [6, 6, 90, 6], "drop_path_rate": 0.5}
)

registry.register_weights(
    "van_b0_il-common",
    {
        "description": "VAN B0 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 15.2,
                "sha256": "6ab691627500cbba1e289e5293039630f76fba06b93222dddb5b06e711ccc31e",
            },
        },
        "net": {"network": "van_b0", "tag": "il-common"},
    },
)
registry.register_weights(
    "van_b2_arabian-peninsula256px",
    {
        "url": "https://huggingface.co/birder-project/van_b2_arabian-peninsula/resolve/main",
        "description": "VAN B2 model trained on the arabian-peninsula dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 101.2,
                "sha256": "fda34cdc4f87833dcb3eab29480305ea63d23575f0947c2757edbafed3ab5759",
            },
        },
        "net": {"network": "van_b2", "tag": "arabian-peninsula256px"},
    },
)
registry.register_weights(
    "van_b2_arabian-peninsula",
    {
        "url": "https://huggingface.co/birder-project/van_b2_arabian-peninsula/resolve/main",
        "description": "VAN B2 model trained on the arabian-peninsula dataset",
        "resolution": (384, 384),
        "formats": {
            "pt": {
                "file_size": 101.2,
                "sha256": "a119b788874a2dcf6454fb1357d7e128db450778fd2947d8b2bbdde5673a6cc7",
            },
        },
        "net": {"network": "van_b2", "tag": "arabian-peninsula"},
    },
)
