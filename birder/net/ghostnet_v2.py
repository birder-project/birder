"""
GhostNet v2, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/ghostnet.py

Paper "GhostNetV2: Enhance Cheap Operation with Long-Range Attention", https://arxiv.org/abs/2211.12905
"""

# Reference license: Apache-2.0

import math
from collections import OrderedDict
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible
from birder.net.ghostnet_v1 import GhostModule


# pylint: disable=invalid-name
class GhostModule_v2(nn.Module):
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
        self.gate_fn = nn.Sigmoid()
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
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.short_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 5),
                stride=(1, 1),
                padding=(0, 2),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(5, 1),
                stride=(1, 1),
                padding=(2, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.short_conv(self.avg_pool(x))
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.concat([x1, x2], dim=1)
        out = out[:, : self.out_channels, :, :] * F.interpolate(
            self.gate_fn(res), size=(out.shape[-2], out.shape[-1]), mode="nearest"
        )

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
        module_version: Literal["v1", "v2"],
    ) -> None:
        super().__init__()

        # Point-wise expansion
        if module_version == "v1":
            self.ghost1 = GhostModule(
                in_channels, mid_channels, kernel_size=(1, 1), ratio=2, dw_size=(3, 3), stride=(1, 1), use_act=True
            )

        else:
            self.ghost1 = GhostModule_v2(
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

        # shortcut
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
class GhostNet_v2(DetectorBackbone):
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
        layer_idx = 0
        for cfg in block_config:
            for k, exp_size, c, se_ratio, s in cfg:
                if s > 1:
                    stages[f"stage{i}"] = nn.Sequential(*layers)
                    return_channels.append(prev_channels)
                    layers = []
                    i += 1

                out_channels = make_divisible(c * width, 4)
                mid_channels = make_divisible(exp_size * width, 4)
                if layer_idx > 1:
                    module_version: Literal["v1", "v2"] = "v2"

                else:
                    module_version = "v1"

                layers.append(
                    GhostBottleneck(
                        prev_channels,
                        mid_channels,
                        out_channels,
                        (k, k),
                        (s, s),
                        se_ratio=se_ratio,
                        module_version=module_version,
                    )
                )
                prev_channels = out_channels
                layer_idx += 1

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_weights(
    "ghostnet_v2_1_il-common",
    {
        "description": "GhostNet v2 1.0x model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 20.9,
                "sha256": "efb438440a88b7e159b36bbba7de1d0748c8caae16dbe85b0330ccf16ade93c0",
            }
        },
        "net": {"network": "ghostnet_v2", "net_param": 1, "tag": "il-common"},
    },
)
