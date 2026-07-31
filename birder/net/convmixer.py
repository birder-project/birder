"""
ConvMixer, adapted from
https://github.com/locuslab/convmixer

Paper "Patches Are All You Need?", https://arxiv.org/abs/2201.09792
"""

# Reference license: MIT

from collections.abc import Callable
from typing import Any
from typing import Optional

import torch
from torch import nn

from birder.layers.activations import get_activation_module
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
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, config=config, size=size)
        assert self.config is not None, "must set config"

        dim: int = self.config["dim"]
        depth: int = self.config["depth"]
        kernel_size: tuple[int, int] = self.config["kernel_size"]
        patch_size: tuple[int, int] = self.config["patch_size"]
        activation_layer = get_activation_module(self.config["activation"])

        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, dim, kernel_size=patch_size, stride=patch_size, padding=(0, 0)),
            activation_layer(),
            nn.BatchNorm2d(dim),
        )

        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.body = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=(1, 1), padding=padding, groups=dim),
                            activation_layer(),
                            nn.BatchNorm2d(dim),
                        )
                    ),
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                        activation_layer(),
                        nn.BatchNorm2d(dim),
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

        self.max_stride = patch_size[0]
        self.stem_stride = patch_size[0]
        self.stem_width = dim
        self.feature_dim = dim

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.body(x)

    def embedding_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.features(features)


registry.register_model_config(
    "convmixer_768_32",
    ConvMixer,
    config={"dim": 768, "depth": 32, "kernel_size": (7, 7), "patch_size": (7, 7), "activation": "relu"},
)
registry.register_model_config(
    "convmixer_1024_20",
    ConvMixer,
    config={"dim": 1024, "depth": 20, "kernel_size": (9, 9), "patch_size": (14, 14), "activation": "gelu"},
)
registry.register_model_config(
    "convmixer_1536_20",
    ConvMixer,
    config={"dim": 1536, "depth": 20, "kernel_size": (9, 9), "patch_size": (7, 7), "activation": "gelu"},
)
