"""
ShuffleNet v1, adapted from
https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/network.py

Paper "ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices",
https://arxiv.org/abs/1707.01083
"""

# Reference license: MIT

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    (batch_size, num_channels, height, width) = x.size()
    channels_per_group = num_channels // groups

    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # Transpose
    # contiguous required if transpose is used before view (https://github.com/pytorch/pytorch/issues/764)
    x = torch.transpose(x, 1, 2).contiguous()

    # Flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int) -> None:
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return channel_shuffle(x, groups=self.groups)


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groups: int, grouped_conv: bool) -> None:
        super().__init__()
        assert in_channels <= out_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.grouped_conv = grouped_conv
        bottleneck_channels = out_channels // 4

        if in_channels == out_channels:
            dw_conv_stride = 1
        elif in_channels < out_channels:
            dw_conv_stride = 2
            out_channels -= in_channels
        else:
            raise ValueError("in_channels must be smaller or equal to out_channels")

        if grouped_conv is True:
            first_groups = groups
        else:
            first_groups = 1

        self.bottleneck = Conv2dNormActivation(
            in_channels,
            bottleneck_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            groups=first_groups,
            bias=False,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=(3, 3),
                stride=(dw_conv_stride, dw_conv_stride),
                padding=(1, 1),
                groups=bottleneck_channels,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels),
            nn.Conv2d(
                bottleneck_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.bottleneck(x)
        if self.grouped_conv is True:
            x = channel_shuffle(x, groups=self.groups)

        x = self.expand(x)
        if self.in_channels == self.out_channels:
            x = x + residual

        else:
            residual = F.avg_pool2d(  # pylint: disable=not-callable
                residual, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            x = torch.concat((x, residual), dim=1)

        x = F.relu(x)
        return x


# pylint: disable=invalid-name
class ShuffleNet_v1(DetectorBackbone):
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

        groups: int = self.config["groups"]
        if groups == 1:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 144, 288, 576]

        elif groups == 2:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 200, 400, 800]

        elif groups == 3:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 240, 480, 960]

        elif groups == 4:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 272, 544, 1088]

        elif groups == 8:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 384, 768, 1536]

        else:
            raise ValueError(f"groups = {groups} not supported")

        self.stem = nn.Conv2d(
            self.input_channels, out_channels[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

        stages: OrderedDict[str, nn.Module] = OrderedDict()
        return_channels: list[int] = []

        stages["stage1"] = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        return_channels.append(out_channels[0])
        for i, repeat in enumerate(stage_repeats):
            layers = []
            if i == 0:
                grouped_conv = False
            else:
                grouped_conv = True

            layers.append(
                ShuffleUnit(
                    out_channels[i],
                    out_channels[i + 1],
                    groups=groups,
                    grouped_conv=grouped_conv,
                )
            )
            for _ in range(repeat):
                layers.append(
                    ShuffleUnit(
                        out_channels[i + 1],
                        out_channels[i + 1],
                        groups=groups,
                        grouped_conv=True,
                    )
                )

            stages[f"stage{i+2}"] = nn.Sequential(*layers)
            return_channels.append(out_channels[i + 1])

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.return_channels = return_channels
        self.embedding_size = out_channels[-1]
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

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_alias("shufflenet_v1_1", ShuffleNet_v1, config={"groups": 1})
registry.register_alias("shufflenet_v1_2", ShuffleNet_v1, config={"groups": 2})
registry.register_alias("shufflenet_v1_3", ShuffleNet_v1, config={"groups": 3})
registry.register_alias("shufflenet_v1_4", ShuffleNet_v1, config={"groups": 4})
registry.register_alias("shufflenet_v1_8", ShuffleNet_v1, config={"groups": 8})

registry.register_weights(
    "shufflenet_v1_4_il-common",
    {
        "description": "ShuffleNet v1 (g=4) model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 5.1,
                "sha256": "b3e816c4dfe75526ff38f3b276fd3d9bcea2776261caedacab78305750b98d7e",
            }
        },
        "net": {"network": "shufflenet_v1_4", "tag": "il-common"},
    },
)
