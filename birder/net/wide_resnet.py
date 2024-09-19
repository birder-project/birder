"""
Wide ResNet, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Paper "Wide Residual Networks", https://arxiv.org/abs/1605.07146
"""

# Reference license: BSD 3-Clause

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int], bottle_neck: bool) -> None:
        super().__init__()
        if bottle_neck is True:
            self.block1 = nn.Sequential(
                Conv2dNormActivation(
                    in_channels,
                    out_channels // 2,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                Conv2dNormActivation(
                    out_channels // 2,
                    out_channels // 2,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=(1, 1),
                    bias=False,
                ),
                nn.Conv2d(
                    out_channels // 2,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.block1 = nn.Sequential(
                Conv2dNormActivation(
                    in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False
                ),
                nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        if in_channels == out_channels:
            self.block2 = nn.Identity()
        else:
            self.block2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, padding=(0, 0), bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block1(x)
        identity = self.block2(identity)
        x += identity
        x = self.relu(x)

        return x


# pylint: disable=invalid-name
class Wide_ResNet(BaseNet):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        num_layers = int(self.net_param)

        bottle_neck = True
        filter_list = [64, 256, 512, 1024, 2048]
        if num_layers == 50:
            units = [3, 4, 6, 3]

        elif num_layers == 101:
            units = [3, 4, 23, 3]

        elif num_layers == 152:
            units = [3, 8, 36, 3]

        elif num_layers == 200:
            units = [3, 24, 36, 3]

        elif num_layers == 269:
            units = [3, 30, 48, 8]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        assert len(units) + 1 == len(filter_list)
        num_unit = len(units)

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                filter_list[0],
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Generate body layers
        layers = []
        for i in range(num_unit):
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            layers.append(ResidualBlock(filter_list[i], filter_list[i + 1], stride=stride, bottle_neck=bottle_neck))
            for _ in range(1, units[i]):
                layers.append(
                    ResidualBlock(filter_list[i + 1], filter_list[i + 1], stride=(1, 1), bottle_neck=bottle_neck)
                )

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = filter_list[-1]
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
