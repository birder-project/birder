"""
ShuffleNet v2, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py

Paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design",
https://arxiv.org/abs/1807.11164
"""

# Reference license: BSD 3-Clause

from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import BaseNet
from birder.net.shufflenet_v1 import channel_shuffle


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dw_conv_stride: int) -> None:
        super().__init__()
        self.dw_conv_stride = dw_conv_stride
        branch_channels = out_channels // 2

        if dw_conv_stride == 1:
            branch2_input = branch_channels
            self.branch1 = nn.Sequential()

        else:
            branch2_input = in_channels
            self.branch1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=(3, 3),
                    stride=(dw_conv_stride, dw_conv_stride),
                    padding=(1, 1),
                    groups=in_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(in_channels),
                Conv2dNormActivation(
                    in_channels,
                    branch_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                ),
            )

        self.branch2 = nn.Sequential(
            Conv2dNormActivation(
                branch2_input,
                branch_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=(3, 3),
                stride=(dw_conv_stride, dw_conv_stride),
                padding=(1, 1),
                groups=branch_channels,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            Conv2dNormActivation(
                branch_channels,
                branch_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dw_conv_stride == 1:
            (branch1, branch2) = x.chunk(2, dim=1)
            x = torch.concat((branch1, self.branch2(branch2)), dim=1)
        else:
            x = torch.concat((self.branch1(x), self.branch2(x)), dim=1)

        x = channel_shuffle(x, groups=2)
        return x


# pylint: disable=invalid-name
class ShuffleNet_v2(BaseNet):
    default_size = 224

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(input_channels, num_classes, net_param, size)
        multiplier = net_param
        if multiplier == 0.5:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 48, 96, 192, 1024]

        elif multiplier == 1:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 116, 232, 464, 1024]

        elif multiplier == 1.5:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 176, 352, 704, 1024]

        elif multiplier == 2:
            stage_repeats = [3, 7, 3]
            out_channels = [24, 244, 488, 976, 2048]

        else:
            raise ValueError(f"multiplier = {multiplier} not supported")

        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels,
                out_channels[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

        # Generate stages
        stages = []
        for i, repeat in enumerate(stage_repeats):
            stages.append(ShuffleUnit(in_channels=out_channels[i], out_channels=out_channels[i + 1], dw_conv_stride=2))
            for _ in range(repeat):
                stages.append(
                    ShuffleUnit(in_channels=out_channels[i + 1], out_channels=out_channels[i + 1], dw_conv_stride=1)
                )

        self.body = nn.Sequential(*stages)
        self.features = nn.Sequential(
            Conv2dNormActivation(
                out_channels[-2],
                out_channels[-1],
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
        )
        self.embedding_size = out_channels[-1]
        self.classifier = self.create_classifier()

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.features(x)


registry.register_weights(
    "shufflenet_v2_1_il-common",
    {
        "description": "ShuffleNet v2 1.0x output channels model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 6.4,
                "sha256": "6ba3e78ff5311802ada11852fbad625acfde4d87fd335b1741c2f89aaaa12b41",
            }
        },
        "net": {"network": "shufflenet_v2", "net_param": 1, "tag": "il-common"},
    },
)