"""
Inception-ResNet v2, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_resnet_v2.py

Paper "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning",
https://arxiv.org/abs/1602.07261
"""

# Reference license: Apache-2.0

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


class StemBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = Conv2dNormActivation(
            in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
        )
        self.branch2 = nn.Sequential(
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch0, branch1, branch2, branch_pool), dim=1)

        return x


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = nn.Sequential(
            Conv2dNormActivation(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_3x3dbl = self.branch_3x3dbl(x)
        x = torch.concat((branch_1x1, branch_3x3, branch_3x3dbl), dim=1)
        x = self.conv2d(x)
        x = (x * self.scale) + identity
        x = self.relu(x)

        return x


class InceptionReductionBlockA(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )
        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(256, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch0, branch1, branch_pool), dim=1)

        return x


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(128, 160, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(160, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branch_1x1 = self.branch_1x1(x)
        branch_7x7 = self.branch_7x7(x)
        x = torch.concat((branch_1x1, branch_7x7), dim=1)
        x = self.conv2d(x)
        x = (x * self.scale) + identity
        x = self.relu(x)

        return x


class InceptionReductionBlockB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch1 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 288, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch2 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(288, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch0, branch1, branch2, branch_pool), dim=1)

        return x


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels: int, scale: float, last_relu: bool) -> None:
        super().__init__()
        self.scale = scale
        self.last_relu = last_relu
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 224, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            Conv2dNormActivation(224, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        x = torch.concat((branch_1x1, branch_3x3), dim=1)
        x = self.conv2d(x)
        x = (x * self.scale) + identity
        if self.last_relu is True:
            x = self.relu(x)

        return x


# pylint: disable=invalid-name
class Inception_ResNet_v2(BaseNet):
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
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            Conv2dNormActivation(64, 80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(80, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            StemBlock(192),
        )

        # Stage 1
        layers = []
        for _ in range(10):
            layers.append(InceptionBlockA(320, scale=0.17))

        layers.append(InceptionReductionBlockA(320))
        stage1 = nn.Sequential(*layers)

        # Stage 2
        layers = []
        for _ in range(20):
            layers.append(InceptionBlockB(1088, scale=0.1))

        layers.append(InceptionReductionBlockB(1088))
        stage2 = nn.Sequential(*layers)

        # Stage 3
        layers = []
        for _ in range(9):
            layers.append(InceptionBlockC(2080, scale=0.2, last_relu=True))

        layers.append(InceptionBlockC(2080, scale=1.0, last_relu=False))
        layers.append(Conv2dNormActivation(2080, 1536, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False))
        stage3 = nn.Sequential(*layers)
        self.body = nn.Sequential(stage1, stage2, stage3)

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.embedding_size = 1536
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
