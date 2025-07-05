"""
Paper "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning",
https://arxiv.org/abs/1602.07261
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import DetectorBackbone


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
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
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
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(192, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
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
            in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(128, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
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
            Conv2dNormActivation(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch2 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
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
    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        x = torch.concat((branch_1x1, branch_3x3), dim=1)
        x = self.conv2d(x)
        x = (x * self.scale) + identity
        x = self.relu(x)

        return x


# pylint: disable=invalid-name
class Inception_ResNet_v1(DetectorBackbone):
    default_size = (299, 299)
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
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
            ),
            Conv2dNormActivation(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            Conv2dNormActivation(64, 80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(80, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []

        # Stage 1
        layers = []
        for _ in range(5):
            layers.append(InceptionBlockA(256, scale=0.17))

        stages["stage1"] = nn.Sequential(*layers)
        return_channels.append(256)

        # Stage 2
        layers = []
        layers.append(InceptionReductionBlockA(256))
        for _ in range(10):
            layers.append(InceptionBlockB(896, scale=0.1))

        stages["stage2"] = nn.Sequential(*layers)
        return_channels.append(896)

        # Stage 3
        layers = []
        layers.append(InceptionReductionBlockB(896))
        for _ in range(5):
            layers.append(InceptionBlockC(1792, scale=0.2))

        stages["stage3"] = nn.Sequential(*layers)
        return_channels.append(1792)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.return_channels = return_channels
        self.return_stages = self.return_stages[:-1]
        self.embedding_size = 1792
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
