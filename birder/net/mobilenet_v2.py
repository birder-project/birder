"""
MobileNet v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

Paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/abs/1801.04381
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        expansion_factor: float,
        shortcut: bool,
        activation_layer: Callable[..., nn.Module] = nn.ReLU6,
    ) -> None:
        super().__init__()
        num_expfilter = int(round(in_channels * expansion_factor))

        self.shortcut = shortcut
        self.block = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                num_expfilter,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=activation_layer,
            ),
            Conv2dNormActivation(
                num_expfilter,
                num_expfilter,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=num_expfilter,
                bias=False,
                activation_layer=activation_layer,
            ),
            Conv2dNormActivation(
                num_expfilter,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is True:
            return x + self.block(x)

        return self.block(x)


# pylint: disable=invalid-name
class MobileNet_v2(DetectorBackbone):
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
        alpha = net_param
        alpha_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert alpha in alpha_values, f"alpha = {alpha} not supported"

        # t - expansion factor
        # c - num_filters (channels)
        # n - number of repetitions
        # s - stride
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        base = make_divisible(32 * alpha, 8)
        self.stem = Conv2dNormActivation(
            self.input_channels,
            base,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
            activation_layer=nn.ReLU6,
        )

        layers: list[nn.Module] = []
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        i = 0
        for t, c, n, s in inverted_residual_setting:
            if s > 1:
                stages[f"stage{i}"] = nn.Sequential(*layers)
                return_channels.append(base)
                layers = []
                i += 1

            c = make_divisible(c * alpha, 8)
            layers.append(
                InvertedResidual(
                    base,
                    c,
                    kernel_size=(3, 3),
                    stride=(s, s),
                    padding=(1, 1),
                    expansion_factor=t,
                    shortcut=False,
                )
            )
            for _ in range(1, n):
                layers.append(
                    InvertedResidual(
                        c,
                        c,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        expansion_factor=t,
                        shortcut=True,
                    )
                )

            base = c

        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(base)

        last_channels = make_divisible(1280 * max(1.0, alpha), 8)
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                c,
                last_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=nn.ReLU6,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = last_channels
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_weights(
    "mobilenet_v2_1_il-common",
    {
        "description": "MobileNet v2 (1.0 multiplier) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 10.6,
                "sha256": "d6182293e98c102026f7cdc0d446aaf0e511232173c4b98c1a882c9f147be6e7",
            }
        },
        "net": {"network": "mobilenet_v2", "net_param": 1, "tag": "il-common"},
    },
)
