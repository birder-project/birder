"""
ConvMixer, adapted from
https://github.com/locuslab/convmixer

Paper "Patches Are All You Need?", https://arxiv.org/abs/2201.09792

Changes from original:
* 768/32 uses GELU instead of ReLU
"""

# Reference license: MIT

from collections.abc import Callable
from typing import Any
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
        assert self.config is not None, "must set config"

        dim: int = self.config["dim"]
        depth: int = self.config["depth"]
        kernel_size: tuple[int, int] = self.config["kernel_size"]
        patch_size: tuple[int, int] = self.config["patch_size"]

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
                            padding=padding,
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
                        padding=(0, 0),
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.features(x)


registry.register_model_config(
    "convmixer_768_32", ConvMixer, config={"dim": 768, "depth": 32, "kernel_size": (7, 7), "patch_size": (7, 7)}
)
registry.register_model_config(
    "convmixer_1024_20", ConvMixer, config={"dim": 1024, "depth": 20, "kernel_size": (9, 9), "patch_size": (14, 14)}
)
registry.register_model_config(
    "convmixer_1536_20", ConvMixer, config={"dim": 1536, "depth": 20, "kernel_size": (9, 9), "patch_size": (7, 7)}
)
