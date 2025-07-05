"""
Inception v4, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/inception_v4.py

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


class StemBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv1 = Conv2dNormActivation(
            in_channels, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )

        self.branch0 = nn.Sequential(
            Conv2dNormActivation(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
        )
        self.branch1 = nn.Sequential(
            Conv2dNormActivation(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(64, 64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), bias=False),
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.conv2 = Conv2dNormActivation(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch0 = self.max_pool1(x)
        branch1 = self.conv1(x)
        x = torch.concat((branch0, branch1), dim=1)

        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        x = torch.concat((branch0, branch1), dim=1)

        branch0 = self.max_pool2(x)
        branch1 = self.conv2(x)
        x = torch.concat((branch0, branch1), dim=1)

        return x


class InceptionBlockA(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = nn.Sequential(
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False),
            Conv2dNormActivation(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_1x1 = self.branch_1x1(x)
        branch_3x3 = self.branch_3x3(x)
        branch_3x3dbl = self.branch_3x3dbl(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_1x1, branch_3x3, branch_3x3dbl, branch_pool), dim=1)

        return x


class InceptionReductionBlockA(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_3x3 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 224, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(224, 256, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_3x3 = self.branch_3x3(x)
        branch_3x3dbl = self.branch_3x3dbl(x)
        branch_pool = self.branch_pool(x)
        x = torch.concat((branch_3x3, branch_3x3dbl, branch_pool), dim=1)

        return x


class InceptionBlockB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(224, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
        )
        self.branch_7x7dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
            Conv2dNormActivation(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(224, 224, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
            Conv2dNormActivation(224, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False),
            Conv2dNormActivation(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
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
            Conv2dNormActivation(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
        )
        self.branch_7x7 = nn.Sequential(
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            Conv2dNormActivation(256, 320, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0), bias=False),
            Conv2dNormActivation(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
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
            in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3 = Conv2dNormActivation(
            in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.branch_3x3_a = Conv2dNormActivation(
            384, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False
        )
        self.branch_3x3_b = Conv2dNormActivation(
            384, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False
        )
        self.branch_3x3dbl = nn.Sequential(
            Conv2dNormActivation(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Conv2dNormActivation(384, 448, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False),
            Conv2dNormActivation(448, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
        )
        self.branch_3x3dbl_a = Conv2dNormActivation(
            512, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False
        )
        self.branch_3x3dbl_b = Conv2dNormActivation(
            512, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), count_include_pad=False),
            Conv2dNormActivation(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
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
class Inception_v4(DetectorBackbone):
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
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []

        # Stage 1 (stem stage)
        stages["stage1"] = StemBlock(64)
        return_channels.append(384)

        # Stage 2
        stages["stage2"] = nn.Sequential(
            InceptionBlockA(384),
            InceptionBlockA(384),
            InceptionBlockA(384),
            InceptionBlockA(384),
        )
        return_channels.append(384)

        # Stage 3
        stages["stage3"] = nn.Sequential(
            InceptionReductionBlockA(384),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
            InceptionBlockB(1024),
        )
        return_channels.append(1024)

        # Stage 4
        stages["stage4"] = nn.Sequential(
            InceptionReductionBlockB(1024),
            InceptionBlockC(1536),
            InceptionBlockC(1536),
            InceptionBlockC(1536),
        )
        return_channels.append(1536)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=0.2),
        )
        self.return_channels = return_channels[1:]
        self.return_stages = self.return_stages[1:]
        self.embedding_size = 1536
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
