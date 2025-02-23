"""
MobileNet v3, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

Paper "Searching for MobileNetV3", https://arxiv.org/abs/1905.02244
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class InvertedResidualConfig:
    """
    Stores information listed at Tables 1 and 2 of the MobileNet v3 paper
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expanded_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        width_multi: float,
        use_se: bool,
        activation: Callable[..., nn.Module] = nn.ReLU,
    ):
        self.in_channels = make_divisible(in_channels * width_multi, 8)
        self.out_channels = make_divisible(out_channels * width_multi, 8)
        self.expanded_channels = make_divisible(expanded_channels * width_multi, 8)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_se = use_se
        self.activation = activation


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig) -> None:
        super().__init__()
        if cnf.stride[0] == 1 and cnf.stride[1] == 1 and cnf.in_channels == cnf.out_channels:
            self.shortcut = True
        else:
            self.shortcut = False

        layers = []
        # Expand
        if cnf.expanded_channels != cnf.in_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.in_channels,
                    cnf.expanded_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    activation_layer=cnf.activation,
                )
            )

        # Depthwise
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel_size,
                stride=cnf.stride,
                padding=cnf.padding,
                groups=cnf.expanded_channels,
                bias=False,
                activation_layer=cnf.activation,
            )
        )

        if cnf.use_se is True:
            squeeze_channels = make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(cnf.expanded_channels, squeeze_channels, scale_activation=nn.Hardsigmoid))

        # Project
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is True:
            return x + self.block(x)

        return self.block(x)


# pylint: disable=invalid-name
class MobileNet_v3_Large(DetectorBackbone):
    auto_register = True

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
        large: bool = True,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param=net_param, config=config, size=size)
        assert self.net_param is not None, "must set net-param"
        assert self.config is None, "config not supported"
        alpha = net_param
        alpha_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert alpha in alpha_values, f"alpha = {alpha} not supported"

        if large is True:
            last_channels = int(round(1280 * max(1.0, alpha)))
            net_settings = [
                InvertedResidualConfig(16, 16, 16, (3, 3), (1, 1), (1, 1), alpha, False, nn.ReLU),
                InvertedResidualConfig(16, 24, 64, (3, 3), (2, 2), (1, 1), alpha, False, nn.ReLU),
                InvertedResidualConfig(24, 24, 72, (3, 3), (1, 1), (1, 1), alpha, False, nn.ReLU),
                InvertedResidualConfig(24, 40, 72, (5, 5), (2, 2), (2, 2), alpha, True, nn.ReLU),
                InvertedResidualConfig(40, 40, 120, (5, 5), (1, 1), (2, 2), alpha, True, nn.ReLU),
                InvertedResidualConfig(40, 40, 120, (5, 5), (1, 1), (2, 2), alpha, True, nn.ReLU),
                InvertedResidualConfig(40, 80, 240, (3, 3), (2, 2), (1, 1), alpha, False, nn.Hardswish),
                InvertedResidualConfig(80, 80, 200, (3, 3), (1, 1), (1, 1), alpha, False, nn.Hardswish),
                InvertedResidualConfig(80, 80, 184, (3, 3), (1, 1), (1, 1), alpha, False, nn.Hardswish),
                InvertedResidualConfig(80, 80, 184, (3, 3), (1, 1), (1, 1), alpha, False, nn.Hardswish),
                InvertedResidualConfig(80, 112, 480, (3, 3), (1, 1), (1, 1), alpha, True, nn.Hardswish),
                InvertedResidualConfig(112, 112, 672, (3, 3), (1, 1), (1, 1), alpha, True, nn.Hardswish),
                InvertedResidualConfig(112, 160, 672, (5, 5), (2, 2), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(160, 160, 960, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(160, 160, 960, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
            ]
        else:
            last_channels = int(round(1024 * max(1.0, alpha)))
            net_settings = [
                InvertedResidualConfig(16, 16, 16, (3, 3), (2, 2), (1, 1), alpha, True, nn.ReLU),
                InvertedResidualConfig(16, 24, 72, (3, 3), (2, 2), (1, 1), alpha, False, nn.ReLU),
                InvertedResidualConfig(24, 24, 88, (3, 3), (1, 1), (1, 1), alpha, False, nn.ReLU),
                InvertedResidualConfig(24, 40, 96, (5, 5), (2, 2), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(40, 40, 240, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(40, 40, 240, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(40, 48, 120, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(48, 48, 144, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(48, 96, 288, (5, 5), (2, 2), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(96, 96, 576, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
                InvertedResidualConfig(96, 96, 576, (5, 5), (1, 1), (2, 2), alpha, True, nn.Hardswish),
            ]

        self.stem = Conv2dNormActivation(
            self.input_channels,
            net_settings[0].in_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False,
            activation_layer=nn.Hardswish,
        )

        layers: list[nn.Module] = []
        i = 0
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for idx, block_settings in enumerate(net_settings):
            if block_settings.stride[0] > 1 or block_settings.stride[1] > 1:
                stages[f"stage{i}"] = nn.Sequential(*layers)
                return_channels.append(net_settings[idx - 1].out_channels)
                layers = []
                i += 1

            layers.append(InvertedResidual(block_settings))

        stages[f"stage{i}"] = nn.Sequential(*layers)
        return_channels.append(net_settings[-1].out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                net_settings[-1].out_channels,
                net_settings[-1].out_channels * 6,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=nn.Hardswish,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels[1:5]
        self.embedding_size = net_settings[-1].out_channels * 6
        self.last_channels = last_channels
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

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.Linear(embed_dim, self.last_channels),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(self.last_channels, self.num_classes),
        )


registry.register_weights(
    "mobilenet_v3_large_0.75_il-common",
    {
        "description": "MobileNet v3 large (0.75 multiplier) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 12.1,
                "sha256": "92412316f3dbcc41e4f3186acb50027e87ce0ea3ea1d6b5a726ea883fea20b8e",
            }
        },
        "net": {"network": "mobilenet_v3_large", "net_param": 0.75, "tag": "il-common"},
    },
)
registry.register_weights(
    "mobilenet_v3_large_1_il-common",
    {
        "description": "MobileNet v3 large (1.0 multiplier) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 18.1,
                "sha256": "85bc65482716368c4fd69d4c699b61330fdf806d99e6c88ffbeda54886c2f3d4",
            }
        },
        "net": {"network": "mobilenet_v3_large", "net_param": 1, "tag": "il-common"},
    },
)
