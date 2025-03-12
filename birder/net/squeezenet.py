"""
SqueezeNet v1.1, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

Paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size",
https://arxiv.org/abs/1602.07360
"""

# Reference license: BSD 3-Clause

from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.net.base import BaseNet


class Fire(nn.Module):
    def __init__(self, in_planes: int, squeeze: int, expand: int) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(in_planes, squeeze, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.left = nn.Conv2d(squeeze, expand, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.left_activation = nn.ReLU(inplace=True)
        self.right = nn.Conv2d(squeeze, expand, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.right_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        left = self.left_activation(self.left(x))
        right = self.right_activation(self.right(x))

        x = torch.concat((left, right), dim=1)

        return x


class SqueezeNet(BaseNet):
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
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True),
        )
        self.body = nn.Sequential(
            Fire(64, 16, 64),
            Fire(128, 16, 64),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True),
            Fire(128, 32, 128),
            Fire(256, 32, 128),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0), ceil_mode=True),
            Fire(256, 48, 192),
            Fire(384, 48, 192),
            Fire(384, 64, 256),
            Fire(512, 64, 256),
        )
        self.embedding_size = 512
        self.classifier = self.create_classifier()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def create_classifier(self, embed_dim: Optional[int] = None) -> nn.Module:
        if self.num_classes == 0:
            return nn.Identity()

        if embed_dim is None:
            embed_dim = self.embedding_size

        return nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Conv2d(
                embed_dim,
                self.num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
