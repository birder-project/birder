"""
Inception v3, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py

Paper "Rethinking the Inception Architecture for Computer Vision", https://arxiv.org/abs/1512.00567
"""

# Reference license: BSD 3-Clause

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import BaseNet


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_5x5 = nn.Sequential(
            Conv2dNormActivation(in_channels, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False),
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(
                in_channels, pool_features, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_1x1 = self.branch_1x1(x)
        branch_5x5 = self.branch_5x5(x)
        branch_3x3dbl = self.branch_3x3dbl(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_1x1, branch_5x5, branch_3x3dbl, branch_pool), dim=1)

        return x


class InceptionReductionBlockA(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_3x3 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_3x3 = self.branch_3x3(x)
        branch_3x3dbl = self.branch_3x3dbl(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_3x3, branch_3x3dbl, branch_pool), dim=1)

        return x


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels: int, num_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(
                in_channels, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            ),
            Conv2dNormActivation(
                num_channels, num_channels, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False
            ),
            Conv2dNormActivation(num_channels, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
        )
        self.branch_7x7dbl = nn.Sequential(
            Conv2dNormActivation(
                in_channels, num_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
            ),
            Conv2dNormActivation(
                num_channels, num_channels, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False
            ),
            Conv2dNormActivation(
                num_channels, num_channels, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False
            ),
            Conv2dNormActivation(
                num_channels, num_channels, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False
            ),
            Conv2dNormActivation(num_channels, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_1x1 = self.branch_1x1(x)
        branch_7x7 = self.branch_7x7(x)
        branch_7x7dbl = self.branch_7x7dbl(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_1x1, branch_7x7, branch_7x7dbl, branch_pool), dim=1)

        return x


class InceptionReductionBlockB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_3x3 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_3x3 = self.branch_3x3(x)
        branch_7x7 = self.branch_7x7(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_3x3, branch_7x7, branch_pool), dim=1)

        return x


class InceptionBlockC(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 320, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3_a = Conv2dNormActivation(
            384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False
        )
        self.branch_3x3_b = Conv2dNormActivation(
            384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 448, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(448, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_3x3dbl_a = Conv2dNormActivation(
            384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False
        )
        self.branch_3x3dbl_b = Conv2dNormActivation(
            384, 384, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_1x1 = self.branch_1x1(x)

        branch_3x3 = self.branch_3x3(x)
        branch_3x3 = [
            self.branch_3x3_a(branch_3x3),
            self.branch_3x3_b(branch_3x3),
        ]
        branch_3x3 = torch.concat(branch_3x3, dim=1)

        branch_3x3dbl = self.branch_3x3dbl(x)
        branch_3x3dbl = [
            self.branch_3x3dbl_a(branch_3x3dbl),
            self.branch_3x3dbl_b(branch_3x3dbl),
        ]
        branch_3x3dbl = torch.concat(branch_3x3dbl, dim=1)

        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_1x1, branch_3x3, branch_3x3dbl, branch_pool), dim=1)

        return x


# pylint: disable=invalid-name
class Inception_v3(BaseNet):
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

        self.stem = Conv2dNormActivation(
            self.input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )
        stage1 = nn.Sequential(
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        stage2 = nn.Sequential(
            Conv2dNormActivation(64, 80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(80, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        stage3 = nn.Sequential(
            InceptionBlockA(192, pool_features=32),
            InceptionBlockA(256, pool_features=64),
            InceptionBlockA(288, pool_features=64),
            InceptionReductionBlockA(288),
        )
        stage4 = nn.Sequential(
            InceptionBlockB(768, num_channels=128),
            InceptionBlockB(768, num_channels=160),
            InceptionBlockB(768, num_channels=160),
            InceptionBlockB(768, num_channels=192),
            InceptionReductionBlockB(768),
        )
        stage5 = nn.Sequential(
            InceptionBlockC(1280),
            InceptionBlockC(2048),
        )
        self.body = nn.Sequential(stage1, stage2, stage3, stage4, stage5)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=0.5),
        )
        self.embedding_size = 2048
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)