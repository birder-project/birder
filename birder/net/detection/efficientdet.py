"""
EfficientDet, adapted from
https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/efficientdet.py

Paper "EfficientDet: Scalable and Efficient Object Detection", https://arxiv.org/abs/1911.09070
"""

# Reference license: Apache-2.0

from collections.abc import Callable
from functools import partial
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet


class Interpolate2d(nn.Module):
    """
    Resamples a 2d image

    The input data is assumed to be of the form
    batch x channels x [optional depth] x [optional height] x width.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor respectively.
    """

    def __init__(
        self,
        size: Optional[int | tuple[int, int]] = None,
        scale_factor: Optional[float | tuple[float, float]] = None,
        mode: str = "nearest",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, self.size, self.scale_factor, self.mode, self.align_corners, recompute_scale_factor=False
        )


class ResampleFeatureMap(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_size: tuple[int, int],
        output_size: tuple[int, int],
        downsample: Literal["max", "bilinear"],
        upsample: Literal["nearest", "bilinear"],
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            # padding = ((stride - 1) + (kernel_size - 1)) // 2
            self.add_module(
                "conv",
                Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    norm_layer=norm_layer,
                    bias=False,
                    activation_layer=None,
                ),
            )

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample == "max":
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                kernel_size = (stride_size_h + 1, stride_size_w + 1)
                stride = (stride_size_h, stride_size_w)
                padding = (
                    ((stride[0] - 1) + (kernel_size[0] - 1)) // 2,
                    ((stride[1] - 1) + (kernel_size[1] - 1)) // 2,
                )

                down_inst = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

            else:
                down_inst = Interpolate2d(size=output_size, mode=downsample)

            self.add_module("downsample", down_inst)

        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                self.add_module("upsample", Interpolate2d(size=output_size, mode=upsample))


class FNode(nn.Module):
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super().__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class HeadNet(nn.Module):
    def __init__(
        self,
        num_outputs: int,
        num_levels: int,
        repeats: int,
        fpn_channels: int,
        aspect_ratios: list[float],
        num_scales: int,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers = []
        for _ in range(repeats):
            layers.append(
                nn.Conv2d(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=fpn_channels,
                    bias=True,
                )
            )
            layers.append(
                Conv2dNormActivation(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 1),
                    norm_layer=norm_layer,
                    bias=False,
                    activation_layer=nn.SiLU,
                )
            )

        self.conv_repeat = nn.Sequential(*layers)
        num_anchors = len(aspect_ratios) * num_scales
        self.predict = nn.Sequential(
            nn.Conv2d(
                fpn_channels,
                fpn_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=fpn_channels,
                bias=True,
            ),
            nn.Conv2d(
                fpn_channels,
                num_outputs * num_anchors,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 1),
                bias=True,
            ),
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for level in range(self.num_levels):
            x_level = x[level]
            x_level = self.conv_repeat(x_level)
            outputs.append(self.predict(x_level))

        return outputs


class EfficientDet(DetectionBaseNet):
    default_size = 640

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        net_param: Optional[float] = None,
        size: Optional[int] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param, size)
        assert self.net_param is not None, "must set net-param"
        net_param = int(self.net_param)

        min_level = 3
        max_level = 7
        num_levels = max_level - min_level + 1
        aspect_ratios = [1.0, 2.0, 0.5]
        num_scales = 3

        if net_param == 0:
            box_class_repeats = 3
            fpn_channels = 64
        else:
            raise ValueError(f"net_param = {net_param} not supported")

        self.class_net = HeadNet(
            num_outputs=self.num_classes,
            num_levels=num_levels,
            repeats=box_class_repeats,
            fpn_channels=fpn_channels,
            aspect_ratios=aspect_ratios,
            num_scales=num_scales,
        )
        self.box_net = HeadNet(
            num_outputs=4,
            num_levels=num_levels,
            repeats=box_class_repeats,
            fpn_channels=fpn_channels,
            aspect_ratios=aspect_ratios,
            num_scales=num_scales,
        )

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        raise NotImplementedError
