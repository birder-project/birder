"""
MnasNet (B variant), adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py

Paper "MnasNet: Platform-Aware Neural Architecture Search for Mobile", https://arxiv.org/abs/1807.11626

Changes from original:
* Relaxed the paper suggestion of 0.9997 momentum (1.0 - 0.9997 for Pytorch), using 0.99
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    """
    Asymmetric rounding to make val divisible by divisor.
    """

    if 0.0 >= round_up_bias or round_up_bias >= 1.0:
        raise ValueError(f"round_up_bias should be greater than 0.0 and smaller than 1.0 instead of {round_up_bias}")

    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    if new_val >= round_up_bias * val:
        return new_val

    return new_val + divisor


def _get_depths(alpha: float) -> list[int]:
    """
    Scales tensor depths as in reference MobileNet code, prefers rounding up rather than down
    """

    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        expansion_factor: float,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion_factor
        if in_channels == out_channels and stride[0] == 1:
            self.shortcut = True
        else:
            self.shortcut = False

        self.block = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                mid_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv2dNormActivation(
                mid_channels,
                mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=mid_channels,
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv2dNormActivation(
                mid_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                norm_layer=norm_layer,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is True:
            return x + self.block(x)

        return self.block(x)


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        expansion_factor: float,
        repeats: int,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        layers = []
        layers.append(
            InvertedResidual(in_channels, out_channels, kernel_size, stride, padding, expansion_factor, norm_layer)
        )

        for _ in range(1, repeats):
            layers.append(
                InvertedResidual(out_channels, out_channels, kernel_size, (1, 1), padding, expansion_factor, norm_layer)
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MNASNet(DetectorBackbone):
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
        assert self.net_param > 0.0, "alpha should be greater than 0"
        alpha = self.net_param

        depths = _get_depths(alpha)
        norm_layer = partial(nn.BatchNorm2d, momentum=1 - 0.99)

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                depths[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv2dNormActivation(
                depths[0],
                depths[0],
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=depths[0],
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv2dNormActivation(
                depths[0],
                depths[1],
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                norm_layer=norm_layer,
                activation_layer=None,
            ),
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        stages["stage1"] = nn.Sequential(
            InvertedResidualBlock(depths[1], depths[2], (3, 3), (2, 2), (1, 1), 3, 3, norm_layer),
        )
        return_channels.append(depths[2])
        stages["stage2"] = nn.Sequential(
            InvertedResidualBlock(depths[2], depths[3], (5, 5), (2, 2), (2, 2), 3, 3, norm_layer),
        )
        return_channels.append(depths[3])
        stages["stage3"] = nn.Sequential(
            InvertedResidualBlock(depths[3], depths[4], (5, 5), (2, 2), (2, 2), 6, 3, norm_layer),
            InvertedResidualBlock(depths[4], depths[5], (3, 3), (1, 1), (1, 1), 6, 2, norm_layer),
        )
        return_channels.append(depths[5])
        stages["stage4"] = nn.Sequential(
            InvertedResidualBlock(depths[5], depths[6], (5, 5), (2, 2), (2, 2), 6, 4, norm_layer),
            InvertedResidualBlock(depths[6], depths[7], (3, 3), (1, 1), (1, 1), 6, 1, norm_layer),
        )
        return_channels.append(depths[7])
        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                depths[7],
                1280,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.return_channels = return_channels
        self.embedding_size = 1280
        self.classifier = self.create_classifier()

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
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


registry.register_weights(
    "mnasnet_0.5_il-common",
    {
        "description": "MnasNet with depth multiplier of 0.5 trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 5.6,
                "sha256": "bec1d292d6400b54958206e9f89ec50a31e42abd94447fb466161dd0623cd75a",
            }
        },
        "net": {"network": "mnasnet", "net_param": 0.5, "tag": "il-common"},
    },
)
registry.register_weights(
    "mnasnet_1_il-common",
    {
        "description": "MnasNet with depth multiplier of 1 trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 13.9,
                "sha256": "351256cf362e5ed64838fb72e1455aee05793d2eff3c2c02f5a4c0f89cce7e53",
            }
        },
        "net": {"network": "mnasnet", "net_param": 1, "tag": "il-common"},
    },
)
