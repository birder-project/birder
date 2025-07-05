"""
SqueezeNext 23v5 version.

Paper "SqueezeNext: Hardware-Aware Neural Network Design",  https://arxiv.org/abs/1803.10615
"""

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


class SqnxtUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride == 2:
            reduction = 1
            self.identity = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding=(0, 0),
                bias=False,
            )

        elif in_channels > out_channels:
            reduction = 4
            self.identity = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding=(0, 0),
                bias=False,
            )

        else:
            reduction = 2
            self.identity = nn.Identity()

        self.block = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                in_channels // reduction,
                kernel_size=(1, 1),
                stride=(stride, stride),
                padding=(0, 0),
                bias=False,
            ),
            Conv2dNormActivation(
                in_channels // reduction,
                in_channels // (2 * reduction),
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            Conv2dNormActivation(
                in_channels // (2 * reduction),
                in_channels // reduction,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
                bias=False,
            ),
            Conv2dNormActivation(
                in_channels // reduction,
                in_channels // reduction,
                kernel_size=(3, 1),
                stride=(1, 1),
                padding=(1, 0),
                bias=False,
            ),
            Conv2dNormActivation(
                in_channels // reduction,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.identity(x)
        x = self.block(x)
        x = x + identity
        x = self.relu(x)

        return x


class SqueezeNext(DetectorBackbone):
    default_size = (227, 227)
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
        width_scale = self.net_param
        width_scale_values = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        assert width_scale in width_scale_values, f"width scale = {width_scale} not supported"

        channels_per_layers = [32, 64, 128, 256]
        layers_per_stage = [2, 4, 14, 1]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                int(64 * width_scale),
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True),
        )

        in_channels = int(64 * width_scale)
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, lps in enumerate(layers_per_stage):
            layers = []
            for j in range(lps):
                if j == 0 and i != 0:
                    stride = 2
                else:
                    stride = 1

                out_channels = int(channels_per_layers[i] * width_scale)
                layers.append(SqnxtUnit(in_channels, out_channels, stride))
                in_channels = out_channels

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(out_channels)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                int(128 * width_scale),
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = int(128 * width_scale)
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
    "squeezenext_1_il-common",
    {
        "description": "SqueezeNext v2 1.0x output channels model trained on the il-common dataset",
        "resolution": (259, 259),
        "formats": {
            "pt": {
                "file_size": 3.5,
                "sha256": "da01d1cd05c71b80b5e4e6ca66400f64fa3f6179d0e90834c4f6942c8095557a",
            }
        },
        "net": {"network": "squeezenext", "net_param": 1, "tag": "il-common"},
    },
)
