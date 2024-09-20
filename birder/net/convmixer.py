"""
ConvMixer, adapted from
https://github.com/locuslab/convmixer

Paper "Patches Are All You Need?", https://arxiv.org/abs/2201.09792

Changes from original:
* 768/32 uses GELU instead of ReLU
"""

# Reference license: MIT

from collections.abc import Callable
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import BaseNet


class Residual(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class ConvMixer(BaseNet):
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
        net_param = int(self.net_param)

        if net_param == 0:
            # 768 / 32
            dim = 768
            depth = 32
            kernel_size = (7, 7)
            patch_size = (7, 7)

        elif net_param == 1:
            # 1024 / 20
            dim = 1024
            depth = 20
            kernel_size = (9, 9)
            patch_size = (14, 14)

        elif net_param == 2:
            # 1536 / 20
            dim = 1536
            depth = 20
            kernel_size = (9, 9)
            patch_size = (7, 7)

        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.stem = Conv2dNormActivation(
            self.input_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=(0, 0),
            activation_layer=nn.GELU,
            inplace=None,
        )

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.body = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        Conv2dNormActivation(
                            dim,
                            dim,
                            kernel_size=kernel_size,
                            stride=(1, 1),
                            padding=(padding),
                            groups=dim,
                            activation_layer=nn.GELU,
                            inplace=None,
                        )
                    ),
                    Conv2dNormActivation(
                        dim,
                        dim,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(1, 1),
                        activation_layer=nn.GELU,
                        inplace=None,
                    ),
                )
                for _ in range(depth)
            ]
        )

        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = dim
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias("convmixer_768_32", ConvMixer, 0)
registry.register_alias("convmixer_1024_20", ConvMixer, 1)
registry.register_alias("convmixer_1536_20", ConvMixer, 2)
