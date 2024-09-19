"""
SqueezeNext 23v5 version.

Paper "SqueezeNext: Hardware-Aware Neural Network Design",  https://arxiv.org/abs/1803.10615
"""

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


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


class SqueezeNext(BaseNet):
    default_size = 227

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
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
        layers = []
        for i, lps in enumerate(layers_per_stage):
            for j in range(lps):
                if j == 0 and i != 0:
                    stride = 2
                else:
                    stride = 1

                out_channels = int(channels_per_layers[i] * width_scale)
                layers.append(SqnxtUnit(in_channels, out_channels, stride))
                in_channels = out_channels

        self.body = nn.Sequential(*layers)
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
        self.embedding_size = int(128 * width_scale)
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
