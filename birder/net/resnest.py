"""
ResNeSt, adapted from
https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnest.py

Paper "ResNeSt: Split-Attention Networks", https://arxiv.org/abs/2004.08955
"""

# Reference license: Apache-2.0

from collections import OrderedDict
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.base import make_divisible


class RadixSoftmax(nn.Module):
    def __init__(self, radix: int, cardinality: int) -> None:
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.radix > 1:
            x = x.view(B, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(B, -1)

        else:
            x = torch.sigmoid(x)

        return x


class SplitAttn(nn.Module):
    """
    Split-Attention (aka Splat)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int],
        groups: int,
        bias: bool,
        radix: int = 2,
        rd_ratio: float = 0.25,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        self.radix = radix
        mid_chs = out_channels * radix
        attn_chs = make_divisible(in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)

        self.conv = Conv2dNormActivation(
            in_channels,
            mid_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups * radix,
            bias=bias,
        )

        self.fc1 = Conv2dNormActivation(
            out_channels, attn_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=groups
        )
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=groups)
        self.r_softmax = RadixSoftmax(radix, groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        (B, RC, H, W) = x.size()  # pylint: disable=invalid-name
        if self.radix > 1:
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x

        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = self.fc1(x_gap)
        x_attn = self.fc2(x_gap)

        x_attn = self.r_softmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn

        return out.contiguous()


class ResNeStBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: tuple[int, int], expansion: int, radix: int
    ) -> None:
        super().__init__()
        assert radix >= 1

        if stride[0] > 1 or stride[1] > 1:
            avd_stride = stride
            stride = (1, 1)

        else:
            avd_stride = (0, 0)

        self.conv_norm_act1 = Conv2dNormActivation(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.splat = SplitAttn(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=(1, 1),
            groups=1,
            bias=False,
            radix=radix,
        )
        if avd_stride[0] > 0 or avd_stride[1] > 0:
            self.avd_last = nn.AvgPool2d(kernel_size=(3, 3), stride=avd_stride, padding=(1, 1))

        else:
            self.avd_last = nn.Identity()

        self.conv_norm_act2 = Conv2dNormActivation(
            out_channels,
            out_channels * expansion,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
            activation_layer=None,
        )

        if in_channels == out_channels * expansion:
            self.downsample = nn.Identity()

        else:
            if avd_stride == (0, 0):
                self.downsample = Conv2dNormActivation(
                    in_channels,
                    out_channels * expansion,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    bias=False,
                    activation_layer=None,
                )

            else:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(2, 2), stride=avd_stride, padding=(0, 0), count_include_pad=False),
                    Conv2dNormActivation(
                        in_channels,
                        out_channels * expansion,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        bias=False,
                        activation_layer=None,
                    ),
                )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv_norm_act1(x)
        x = self.splat(x)
        x = self.avd_last(x)
        x = self.conv_norm_act2(x)

        identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x


class ResNeSt(DetectorBackbone):
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

        radix = 2
        expansion = 4
        filter_list = [64, 128, 256, 512]
        stem_width: int = self.config["stem_width"]
        units: list[int] = self.config["units"]
        final_drop: float = self.config["final_drop"]

        in_channels = stem_width * 2
        self.stem = nn.Sequential(
            Conv2dNormActivation(
                self.input_channels, stem_width, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
            ),
            Conv2dNormActivation(stem_width, stem_width, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            Conv2dNormActivation(
                stem_width, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

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

                layers.append(ResNeStBottleneck(in_channels, channels, stride, expansion, radix))
                in_channels = channels * expansion

            stages[f"stage{i+1}"] = nn.Sequential(*layers)
            return_channels.append(channels * expansion)

        self.body = nn.Sequential(stages)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1),
            nn.Dropout(p=final_drop),
        )
        self.return_channels = return_channels
        self.embedding_size = filter_list[-1] * expansion
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


registry.register_alias("resnest_14", ResNeSt, config={"stem_width": 32, "units": [1, 1, 1, 1], "final_drop": 0.0})
registry.register_alias("resnest_26", ResNeSt, config={"stem_width": 32, "units": [2, 2, 2, 2], "final_drop": 0.0})
registry.register_alias("resnest_50", ResNeSt, config={"stem_width": 32, "units": [3, 4, 6, 3], "final_drop": 0.0})
registry.register_alias("resnest_101", ResNeSt, config={"stem_width": 64, "units": [3, 4, 23, 3], "final_drop": 0.0})
registry.register_alias("resnest_200", ResNeSt, config={"stem_width": 64, "units": [3, 24, 36, 3], "final_drop": 0.2})
registry.register_alias("resnest_269", ResNeSt, config={"stem_width": 64, "units": [3, 30, 48, 8], "final_drop": 0.2})

registry.register_weights(
    "resnest_14_il-common",
    {
        "description": "ResNeSt 14 model trained on the il-common dataset",
        "resolution": (256, 256),
        "formats": {
            "pt": {
                "file_size": 35.7,
                "sha256": "d80c2d290bdab79b03496853df87cb48fc01f63690500ab7d15086a2149ff1ba",
            }
        },
        "net": {"network": "resnest_14", "tag": "il-common"},
    },
)
