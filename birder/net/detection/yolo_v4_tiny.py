"""
YOLO v4 Tiny, adapted from
https://github.com/Tianxiaomo/pytorch-YOLOv4

Paper "Scaled-YOLOv4: Scaling Cross Stage Partial Network", https://arxiv.org/abs/2011.08036
"""

# Reference license: Apache-2.0

from functools import partial
from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import DetectorBackbone
from birder.net.detection.yolo_v3 import YOLOAnchorGenerator
from birder.net.detection.yolo_v3 import YOLOHead
from birder.net.detection.yolo_v4 import YOLO_v4

# Default anchors from YOLO v4 Tiny (COCO)
DEFAULT_ANCHORS = [
    [(10, 14), (23, 27), (37, 58)],  # Medium
    [(81, 82), (135, 169), (344, 319)],  # Large
]

# Scale factors per detection scale to eliminate grid sensitivity
DEFAULT_SCALE_XY = [1.05, 1.05]  # [medium, large]


class YOLOTinyNeck(nn.Module):
    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        (c4, c5) = in_channels

        self.conv_c5 = Conv2dNormActivation(
            c5,
            c5 // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv_c5_out = Conv2dNormActivation(
            c5 // 2,
            c5,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv_upsample = Conv2dNormActivation(
            c5 // 2,
            c5 // 4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        concat_channels = c4 + c5 // 4
        self.conv_c4_out = Conv2dNormActivation(
            concat_channels,
            c4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )

        self.out_channels = [c4, c5]

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        feature_list = list(features.values())
        (c4, c5) = feature_list[-2:]

        p5 = self.conv_c5(c5)
        out_large = self.conv_c5_out(p5)

        p5_up = self.upsample(self.conv_upsample(p5))
        p4_cat = torch.concat([c4, p5_up], dim=1)

        out_medium = self.conv_c4_out(p4_cat)

        return [out_medium, out_large]


# pylint: disable=invalid-name
class YOLO_v4_Tiny(YOLO_v4):
    default_size = (416, 416)
    auto_register = True

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
        export_mode: bool = False,
    ) -> None:
        super().__init__(num_classes, backbone, config=config, size=size, export_mode=export_mode)
        assert self.config is None, "config not supported"

        # self.num_classes = self.num_classes - 1 (Subtracted at parent)

        score_thresh = 0.05
        nms_thresh = 0.45
        detections_per_img = 300
        self.ignore_thresh = 0.7

        # Loss coefficients - see YOLO v4 for detailed explanation
        self.noobj_coeff = 0.3
        self.coord_coeff = 0.07
        self.obj_coeff = 1.0
        self.cls_coeff = 1.0

        self.anchors = DEFAULT_ANCHORS
        self.scale_xy = DEFAULT_SCALE_XY
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.backbone.return_channels = self.backbone.return_channels[-2:]
        self.backbone.return_stages = self.backbone.return_stages[-2:]

        self.label_smoothing = 0.1
        self.smooth_positive = 1.0 - self.label_smoothing
        self.smooth_negative = self.label_smoothing / self.num_classes

        self.neck = YOLOTinyNeck(self.backbone.return_channels)

        self.anchor_generator = YOLOAnchorGenerator(self.anchors)
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = YOLOHead(self.neck.out_channels, num_anchors, self.num_classes)
