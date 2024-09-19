"""
Xception, adapted from
https://github.com/keras-team/keras/blob/r2.15/keras/applications/xception.py

Paper "Xception: Deep Learning with Depthwise Separable Convolutions", https://arxiv.org/abs/1610.02357
"""

# Reference license: Apache-2.0

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, repeats: int, stride: tuple[int, int], grow_first: bool
    ) -> None:
        super().__init__()

        if out_channels != in_channels or stride[0] != 1 or stride[1] != 1:
            self.skip = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=stride,
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            )

        else:
            self.skip = nn.Identity()

        layers = []
        for i in range(repeats):
            if grow_first is True:
                out_c = out_channels
                if i == 0:
                    in_c = in_channels
                else:
                    in_c = out_channels

            else:
                in_c = in_channels
                if i < (repeats - 1):
                    out_c = in_channels
                else:
                    out_c = out_channels

            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_c, out_c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.append(nn.BatchNorm2d(out_c))

        if stride[0] != 1 or stride[1] != 1:
            layers.append(nn.MaxPool2d(kernel_size=(3, 3), stride=stride, padding=(1, 1)))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = self.block(x)
        identity = self.skip(x)
        x = branch + identity

        return x


class Xception(BaseNet):
    default_size = 299

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is None, "net-param not supported"

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
            ),
            Conv2dNormActivation(
                32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False, activation_layer=None
            ),  # Remove ReLU here, first Xception block starts with ReLU
        )
        self.body = nn.Sequential(
            XceptionBlock(64, 128, repeats=2, stride=(2, 2), grow_first=True),
            XceptionBlock(128, 256, repeats=2, stride=(2, 2), grow_first=True),
            XceptionBlock(256, 728, repeats=2, stride=(2, 2), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 728, repeats=3, stride=(1, 1), grow_first=True),
            XceptionBlock(728, 1024, repeats=2, stride=(2, 2), grow_first=False),
            SeparableConv2d(1024, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = 2048
        self.classifier = self.create_classifier()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) is True:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, nn.BatchNorm2d) is True:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
