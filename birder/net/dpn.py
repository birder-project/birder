"""
Dual Path Networks, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/dpn.py

Paper "Dual Path Networks", https://arxiv.org/abs/1707.01629
"""

# Reference license: Apache-2.0

from typing import Any
from typing import Literal
from typing import Optional
from typing import overload

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import BaseNet


class NormActConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)

        return x


class DualPathBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_1x1_a: int,
        num_3x3_b: int,
        num_1x1_c: int,
        increase: int,
        groups: int,
        block_type: Literal["proj", "down", "normal"],
    ) -> None:
        super().__init__()
        self.num_1x1_c = num_1x1_c
        if block_type == "proj":
            key_stride = 1
            has_proj = True
        elif block_type == "down":
            key_stride = 2
            has_proj = True
        elif block_type == "normal":
            key_stride = 1
            has_proj = False
        else:
            raise ValueError(f"Unknown block type {block_type}")

        self.c1x1_w = None
        if has_proj is True:
            self.c1x1_w = NormActConv2d(
                in_channels,
                num_1x1_c + 2 * increase,
                kernel_size=(1, 1),
                stride=(key_stride, key_stride),
                padding=(0, 0),
                bias=False,
            )

        self.c1x1_a = NormActConv2d(
            in_channels, num_1x1_a, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )
        self.c3x3_b = NormActConv2d(
            num_1x1_a,
            num_3x3_b,
            kernel_size=(3, 3),
            stride=(key_stride, key_stride),
            padding=(1, 1),
            groups=groups,
            bias=False,
        )
        self.c1x1_c = NormActConv2d(
            num_3x3_b, num_1x1_c + increase, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

    # pylint: disable=protected-access
    @overload
    @torch.jit._overload_method  # type: ignore[misc]
    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @overload
    @torch.jit._overload_method  # type: ignore[misc]
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    # pylint: enable=protected-access

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, tuple):
            x_in = torch.concat(x, dim=1)
        else:
            x_in = x

        if self.c1x1_w is None:
            x_s1 = x[0]
            x_s2 = x[1]
        else:
            x_s = self.c1x1_w(x_in)
            x_s1 = x_s[:, : self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c :, :, :]

        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        x_in = self.c1x1_c(x_in)
        out1 = x_in[:, : self.num_1x1_c, :, :]
        out2 = x_in[:, self.num_1x1_c :, :, :]

        res = x_s1 + out1
        dense = torch.concat([x_s2, out2], dim=1)

        return (res, dense)


class DPN(BaseNet):
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

        bw_list = [64, 128, 256, 512]
        num_init_features: int = self.config["num_init_features"]
        k_sec: list[int] = self.config["k_sec"]
        inc_sec: list[int] = self.config["inc_sec"]
        k_r: int = self.config["k_r"]
        groups: int = self.config["groups"]

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                num_init_features,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        layers = []
        num_features = num_init_features
        for idx in range(4):
            bw = bw_list[idx] * 4
            inc = inc_sec[idx]
            r = (k_r * bw) // (64 * 4)
            layers.append(DualPathBlock(num_features, r, r, bw, inc, groups, "proj"))
            num_features = bw + 3 * inc
            for _ in range(2, k_sec[idx] + 1):
                layers.append(DualPathBlock(num_features, r, r, bw, inc, groups, "normal"))
                num_features += inc

        self.norm_act = nn.Sequential(nn.BatchNorm2d(num_features), nn.ELU())

        self.body = nn.Sequential(*layers)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = num_features
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = torch.concat(x, dim=1)
        x = self.norm_act(x)

        return self.features(x)


registry.register_alias(
    "dpn_92",
    DPN,
    config={"num_init_features": 64, "k_sec": [3, 4, 20, 3], "inc_sec": [16, 32, 24, 128], "k_r": 96, "groups": 32},
)
registry.register_alias(
    "dpn_98",
    DPN,
    config={"num_init_features": 96, "k_sec": [3, 6, 20, 3], "inc_sec": [16, 32, 32, 128], "k_r": 160, "groups": 40},
)
registry.register_alias(
    "dpn_131",
    DPN,
    config={"num_init_features": 128, "k_sec": [4, 8, 28, 3], "inc_sec": [16, 32, 32, 128], "k_r": 160, "groups": 40},
)
