"""
InceptionNeXt, adapted from
https://github.com/sail-sg/inceptionnext/blob/main/models/inceptionnext.py

Paper "InceptionNeXt: When Inception Meets ConvNeXt", https://arxiv.org/abs/2303.16900
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import StochasticDepth

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class InceptionDWConv2d(nn.Module):
    def __init__(self, in_channels: int, square_kernel_size: int, band_kernel_size: int, branch_ratio: float) -> None:
        super().__init__()

        branch_channels = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(square_kernel_size, square_kernel_size),
            stride=(1, 1),
            padding=square_kernel_size // 2,
            groups=branch_channels,
            bias=True,
        )
        self.dwconv_w = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, band_kernel_size),
            stride=(1, 1),
            padding=(0, band_kernel_size // 2),
            groups=branch_channels,
            bias=True,
        )
        self.dwconv_h = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(band_kernel_size, 1),
            stride=(1, 1),
            padding=(band_kernel_size // 2, 0),
            groups=branch_channels,
            bias=True,
        )
        self.split_indexes = (
            in_channels - (3 * branch_channels),
            branch_channels,
            branch_channels,
            branch_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (x_id, x_hw, x_w, x_h) = torch.split(x, self.split_indexes, dim=1)
        x_hw = self.dwconv_hw(x_hw)
        x_w = self.dwconv_w(x_w)
        x_h = self.dwconv_h(x_h)

        return torch.concat((x_id, x_hw, x_w, x_h), dim=1)


class ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(
            hidden_features, out_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class InceptionNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float,
        layer_scale: float,
        band_kernel_size: int,
        branch_ratio: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            InceptionDWConv2d(
                channels, square_kernel_size=3, band_kernel_size=band_kernel_size, branch_ratio=branch_ratio
            ),
            nn.BatchNorm2d(channels),
            ConvMLP(channels, hidden_features=int(mlp_ratio * channels), out_features=channels),
        )
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale, requires_grad=True)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode="row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.layer_scale * self.block(x)
        x = self.stochastic_depth(x) + identity

        return x


class InceptionNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        depth: int,
        drop_path_rates: list[float],
        mlp_ratio: float,
        layer_scale: float,
        band_kernel_size: int,
        branch_ratio: float,
    ) -> None:
        super().__init__()
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(stride, stride),
                    stride=(stride, stride),
                    padding=(0, 0),
                    bias=True,
                ),
            )

        else:
            self.downsample = nn.Identity()

        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                InceptionNeXtBlock(
                    channels=out_channels,
                    mlp_ratio=mlp_ratio,
                    layer_scale=layer_scale,
                    band_kernel_size=band_kernel_size,
                    branch_ratio=branch_ratio,
                    stochastic_depth_prob=drop_path_rates[i],
                )
            )

        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)

        return x


# pylint: disable=invalid-name
class Inception_NeXt(DetectorBackbone):
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

        layer_scale = 1e-6
        mlp_ratios = [4, 4, 4, 3]
        channels: list[int] = self.config["channels"]
        num_layers: list[int] = self.config["num_layers"]
        band_kernel_size: int = self.config["band_kernel_size"]
        branch_ratio: float = self.config["branch_ratio"]
        drop_path_rate: float = self.config["drop_path_rate"]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            channels[0],
            kernel_size=(4, 4),
            stride=(4, 4),
            padding=(0, 0),
            bias=True,
            activation_layer=None,
        )

        num_stage = len(num_layers)
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_layers)).split(num_layers)]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        prev_chs = channels[0]
        for i in range(num_stage):
            out_chs = channels[i]
            if i > 0:
                stride = 2
            else:
                stride = 1

            stages[f"stage{i+1}"] = InceptionNeXtStage(
                prev_chs,
                out_chs,
                stride=stride,
                depth=num_layers[i],
                drop_path_rates=dp_rates[i],
                mlp_ratio=mlp_ratios[i],
                layer_scale=layer_scale,
                band_kernel_size=band_kernel_size,
                branch_ratio=branch_ratio,
            )
            return_channels.append(out_chs)
            prev_chs = out_chs

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = channels[-1]
        self.last_mlp_ratio = mlp_ratios[-1]
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
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

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.Linear(embed_dim, self.last_mlp_ratio * embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.last_mlp_ratio * embed_dim, eps=1e-6),
            nn.Linear(self.last_mlp_ratio * embed_dim, self.num_classes),
        )


registry.register_model_config(
    "inception_next_a",
    Inception_NeXt,
    config={
        "channels": [40, 80, 160, 320],
        "num_layers": [2, 2, 6, 2],
        "band_kernel_size": 9,
        "branch_ratio": 0.25,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "inception_next_t",
    Inception_NeXt,
    config={
        "channels": [96, 192, 384, 768],
        "num_layers": [3, 3, 9, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "drop_path_rate": 0.1,
    },
)
registry.register_model_config(
    "inception_next_s",
    Inception_NeXt,
    config={
        "channels": [96, 192, 384, 768],
        "num_layers": [3, 3, 27, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "drop_path_rate": 0.3,
    },
)
registry.register_model_config(
    "inception_next_b",
    Inception_NeXt,
    config={
        "channels": [128, 256, 512, 1024],
        "num_layers": [3, 3, 27, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "drop_path_rate": 0.4,
    },
)
registry.register_model_config(
    "inception_next_l",
    Inception_NeXt,
    config={
        "channels": [192, 384, 768, 1536],
        "num_layers": [3, 3, 27, 3],
        "band_kernel_size": 11,
        "branch_ratio": 0.125,
        "drop_path_rate": 0.5,
    },
)

registry.register_weights(
    "inception_next_a_il-common",
    {
        "description": "InceptionNeXt atto model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.7,
                "sha256": "c9e451c8dc44727acedfac400cee9d14eac5ba6b4edbfe79efcfd8802453c506",
            }
        },
        "net": {"network": "inception_next_a", "tag": "il-common"},
    },
)
