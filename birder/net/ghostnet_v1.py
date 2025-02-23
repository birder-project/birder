"""
GhostNet, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/ghostnet.py

Paper "GhostNet: More Features from Cheap Operations", https://arxiv.org/abs/1911.11907
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class GhostModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        ratio: int,
        dw_size: tuple[int, int],
        use_act: bool,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        activation_layer = None
        if use_act is True:
            activation_layer = nn.ReLU

        self.primary_conv = Conv2dNormActivation(
            in_channels,
            init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            bias=False,
            activation_layer=activation_layer,
        )

        self.cheap_operation = Conv2dNormActivation(
            init_channels,
            new_channels,
            kernel_size=dw_size,
            stride=(1, 1),
            padding=((dw_size[0] - 1) // 2, (dw_size[1] - 1) // 2),
            groups=init_channels,
            bias=False,
            activation_layer=activation_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.concat([x1, x2], dim=1)
        out = out[:, : self.out_channels, :, :]

        return out


class GhostBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dw_kernel_size: tuple[int, int],
        stride: tuple[int, int],
        se_ratio: float,
    ) -> None:
        super().__init__()

        # Point-wise expansion
        self.ghost1 = GhostModule(
            in_channels, mid_channels, kernel_size=(1, 1), ratio=2, dw_size=(3, 3), stride=(1, 1), use_act=True
        )

        # Depth-wise convolution
        if stride[0] > 1 or stride[1] > 1:
            self.conv_dw = nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=dw_kernel_size,
                stride=stride,
                padding=((dw_kernel_size[0] - 1) // 2, (dw_kernel_size[1] - 1) // 2),
                groups=mid_channels,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_channels)

        else:
            self.conv_dw = nn.Identity()
            self.bn_dw = nn.Identity()

        # Squeeze-and-excitation
        if se_ratio > 0:
            squeeze_channels = make_divisible(se_ratio * mid_channels, 4)
            self.se = SqueezeExcitation(mid_channels, squeeze_channels, scale_activation=nn.Hardsigmoid)

        else:
            self.se = nn.Identity()

        # Point-wise linear projection
        self.ghost2 = GhostModule(
            mid_channels, out_channels, kernel_size=(1, 1), ratio=2, dw_size=(3, 3), stride=(1, 1), use_act=False
        )

        # Shortcut
        if in_channels == out_channels and stride[0] == 1 and stride[1] == 1:
            self.shortcut = nn.Identity()

        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=dw_kernel_size,
                    stride=stride,
                    padding=((dw_kernel_size[0] - 1) // 2, (dw_kernel_size[1] - 1) // 2),
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ghost1(x)
        x = self.conv_dw(x)
        x = self.bn_dw(x)
        x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(shortcut)

        return x


# pylint: disable=invalid-name
class GhostNet_v1(DetectorBackbone):
    auto_register = True

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
        assert self.net_param is not None, "must set net-param"
        assert self.config is None, "config not supported"
        width = self.net_param

        block_config: list[list[tuple[int, int, int, float, int]]] = [
            # kernel, expansion, channels, se ratio, stride
            # Stage 1
            [(3, 16, 16, 0, 1)],
            # Stage 2
            [(3, 48, 24, 0, 2)],
            [(3, 72, 24, 0, 1)],
            # Stage 3
            [(5, 72, 40, 0.25, 2)],
            [(5, 120, 40, 0.25, 1)],
            # Stage 4
            [(3, 240, 80, 0, 2)],
            [
                (3, 200, 80, 0, 1),
                (3, 184, 80, 0, 1),
                (3, 184, 80, 0, 1),
                (3, 480, 112, 0.25, 1),
                (3, 672, 112, 0.25, 1),
            ],
            # Stage 5
            [(5, 672, 160, 0.25, 2)],
            [
                (5, 960, 160, 0, 1),
                (5, 960, 160, 0.25, 1),
                (5, 960, 160, 0, 1),
                (5, 960, 160, 0.25, 1),
            ],
        ]

        stem_channels = make_divisible(16 * width, 4)
        self.stem = Conv2dNormActivation(
            self.input_channels, stem_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        i = 0
        layers: list[nn.Module] = []
        prev_channels = stem_channels
        for cfg in block_config:
            for k, exp_size, c, se_ratio, s in cfg:
                if s > 1:
                    stages[f"stage{i}"] = nn.Sequential(*layers)
                    return_channels.append(prev_channels)
                    layers = []
                    i += 1

                out_channels = make_divisible(c * width, 4)
                mid_channels = make_divisible(exp_size * width, 4)
                layers.append(
                    GhostBottleneck(prev_channels, mid_channels, out_channels, (k, k), (s, s), se_ratio=se_ratio)
                )
                prev_channels = out_channels

        out_channels = make_divisible(exp_size * width, 4)
        layers.append(
            Conv2dNormActivation(prev_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )
        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(out_channels)

        self.body = nn.Sequential(stages)
        prev_channels = out_channels
        out_channels = 1280
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(prev_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = out_channels
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
