"""
ResNeXt, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

Paper "Aggregated Residual Transformations for Deep Neural Networks", https://arxiv.org/abs/1611.05431
"""

# Reference license: BSD 3-Clause

from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import SqueezeExcitation

from birder.net.base import DetectorBackbone


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        groups: int,
        base_width: int,
        expansion: int,
        squeeze_excitation: bool,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups
        self.block1 = nn.Sequential(
            Conv2dNormActivation(
                in_channels,
                width,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            Conv2dNormActivation(
                width,
                width,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
                bias=False,
            ),
            nn.Conv2d(
                width,
                out_channels * expansion,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * expansion),
        )

        if in_channels == out_channels * expansion:
            self.block2 = nn.Identity()
        else:
            self.block2 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * expansion,
                    kernel_size=(1, 1),
                    stride=stride,
                    padding=(0, 0),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * expansion),
            )

        self.relu = nn.ReLU(inplace=True)
        if squeeze_excitation is True:
            self.se = SqueezeExcitation(out_channels * expansion, out_channels * expansion // 16)
        else:
            self.se = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.block1(x)
        x = self.se(x)
        identity = self.block2(identity)
        x += identity
        x = self.relu(x)

        return x


class ResNeXt(DetectorBackbone):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
        squeeze_excitation: bool = False,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        assert self.net_param is not None, "must set net-param"
        num_layers = int(self.net_param)

        groups = 32
        base_width = 4
        expansion = 4
        filter_list = [64, 128, 256, 512]
        if num_layers == 50:
            units = [3, 4, 6, 3]

        elif num_layers == 101:
            units = [3, 4, 23, 3]

        elif num_layers == 152:
            units = [3, 8, 36, 3]

        else:
            raise ValueError(f"num_layers = {num_layers} not supported")

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                filter_list[0],
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Generate body layers
        in_channels = filter_list[0]

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []
        for i, (channels, num_blocks) in enumerate(zip(filter_list, units)):
            layers = []
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            for block_idx in range(num_blocks):
                if block_idx != 0:
                    stride = (1, 1)

                layers.append(
                    ResidualBlock(
                        in_channels,
                        channels,
                        stride=stride,
                        groups=groups,
                        base_width=base_width,
                        expansion=expansion,
                        squeeze_excitation=squeeze_excitation,
                    )
                )
                in_channels = channels * expansion

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(channels * expansion)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = filter_list[-1] * expansion
        self.classifier = self.create_classifier()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) is True:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)) is True:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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