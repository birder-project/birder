"""
RegNet, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py

Paper "Designing Network Design Spaces", https://arxiv.org/abs/2003.13678

Only the Y variant implemented.
"""

# Reference license: BSD 3-Clause

import math
from collections import OrderedDict
from collections.abc import Iterator
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class BlockParams:
    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_widths: list[int],
        bottleneck_multipliers: list[float],
        strides: list[int],
        se_ratio: float,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> "BlockParams":
        """
        Programmatically compute all the per-block settings, given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - w_a is the width progression slope
        - w_0 is the initial width
        - w_m is the width stepping in the log space

        In other terms
        log(block_width) = log(w_0) + w_m * block_capacity,
        with bock_capacity ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.
        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """

        QUANT = 8  # pylint: disable=invalid-name
        STRIDE = 2  # pylint: disable=invalid-name

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")

        # Compute the block widths. Each stage has one unique block width
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(block_widths + [0], [0] + block_widths, block_widths + [0], [0] + block_widths)
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        (stage_widths, group_widths) = cls._adjust_widths_groups_compatibility(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self) -> Iterator[tuple[int, int, int, int, float]]:
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibility(
        stage_widths: list[int], bottleneck_ratios: list[float], group_widths: list[int]
    ) -> tuple[list[int], list[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """

        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class BottleneckTransform(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> None:
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        width_se_out = int(round(se_ratio * width_in))

        self.block = nn.Sequential(
            Conv2dNormActivation(
                width_in,
                w_b,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            Conv2dNormActivation(
                w_b,
                w_b,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=g,
                bias=False,
            ),
            SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
            ),
            Conv2dNormActivation(
                w_b,
                width_out,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBottleneckBlock(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> None:
        super().__init__()

        if width_in != width_out or stride[0] != 1 or stride[1] != 1:
            self.proj = Conv2dNormActivation(
                width_in,
                width_out,
                kernel_size=(1, 1),
                stride=stride,
                padding=(0, 0),
                bias=False,
                activation_layer=None,
            )

        else:
            self.proj = nn.Identity()

        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride=stride,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            se_ratio=se_ratio,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.f(x)
        x = self.relu(x)

        return x


class AnyStage(nn.Module):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: tuple[int, int],
        depth: int,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: float,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(depth):
            if i == 0:
                in_ch = width_in
                cur_stride = stride
            else:
                in_ch = width_out
                cur_stride = (1, 1)

            layers.append(
                ResBottleneckBlock(
                    in_ch,
                    width_out,
                    stride=cur_stride,
                    group_width=group_width,
                    bottleneck_multiplier=bottleneck_multiplier,
                    se_ratio=se_ratio,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegNet(DetectorBackbone):
    default_size = 224

    # pylint: disable=too-many-branches
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"

        stem_width = 32
        bottleneck_multiplier = 1.0
        se_ratio = 0.25
        if self.net_param == 0.2:
            depth = 13
            w_0 = 24
            w_a = 36.44
            w_m = 2.49
            group_width = 8

        elif self.net_param == 0.4:
            depth = 16
            w_0 = 48
            w_a = 27.89
            w_m = 2.09
            group_width = 8

        elif self.net_param == 0.8:
            depth = 14
            w_0 = 56
            w_a = 38.84
            w_m = 2.4
            group_width = 16

        elif self.net_param == 1.6:
            depth = 27
            w_0 = 48
            w_a = 20.71
            w_m = 2.65
            group_width = 24

        elif self.net_param == 3.2:
            depth = 21
            w_0 = 80
            w_a = 42.63
            w_m = 2.66
            group_width = 24

        elif self.net_param == 6.4:
            depth = 25
            w_0 = 112
            w_a = 33.22
            w_m = 2.27
            group_width = 72

        elif self.net_param == 8:
            depth = 17
            w_0 = 192
            w_a = 76.82
            w_m = 2.19
            group_width = 56

        elif self.net_param == 12:
            depth = 19
            w_0 = 168
            w_a = 73.36
            w_m = 2.37
            group_width = 112

        elif self.net_param == 16:
            depth = 18
            w_0 = 200
            w_a = 106.23
            w_m = 2.48
            group_width = 112

        elif self.net_param == 32:
            depth = 20
            w_0 = 232
            w_a = 115.89
            w_m = 2.53
            group_width = 232

        elif self.net_param == 128:
            depth = 27
            w_0 = 456
            w_a = 160.83
            w_m = 2.52
            group_width = 264

        else:
            raise ValueError(f"net_param = {self.net_param} not supported")

        block_params = BlockParams.from_init_params(depth, w_0, w_a, w_m, group_width, bottleneck_multiplier, se_ratio)

        self.stem = Conv2dNormActivation(
            self.input_channels, stem_width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        current_width = stem_width
        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        i = 0
        for (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in block_params._get_expanded_params():
            stages[f"stage{i+1}"] = AnyStage(
                current_width, width_out, (stride, stride), depth, group_width, bottleneck_multiplier, se_ratio
            )
            return_channels.append(width_out)
            current_width = width_out
            i += 1

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = current_width
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) is True:
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) is True:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear) is True:
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)
