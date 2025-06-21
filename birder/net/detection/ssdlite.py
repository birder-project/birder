"""
SSDLite, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssdlite.py

Paper "MobileNetV2: Inverted Residuals and Linear Bottlenecks", https://arxiv.org/abs/1801.04381

Changes from original:
* Different backbone feature strides
"""

# Reference license: BSD 3-Clause

from typing import Any
from typing import Optional

import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation

from birder.net.base import DetectorBackbone
from birder.net.detection.base import BoxCoder
from birder.net.detection.ssd import SSD
from birder.net.detection.ssd import DefaultBoxGenerator
from birder.net.detection.ssd import SSDMatcher
from birder.net.detection.ssd import SSDScoringHead


class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: list[int], num_anchors: list[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(
                nn.Sequential(
                    Conv2dNormActivation(
                        channels,
                        channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        groups=channels,
                        activation_layer=nn.ReLU6,
                    ),
                    nn.Conv2d(channels, num_classes * anchors, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                )
            )

        # Weights initialization
        for layer in cls_logits.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        super().__init__(cls_logits, num_classes)


class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: list[int], num_anchors: list[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(
                nn.Sequential(
                    Conv2dNormActivation(
                        channels,
                        channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        groups=channels,
                        activation_layer=nn.ReLU6,
                    ),
                    nn.Conv2d(channels, 4 * anchors, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                )
            )

        # Weights initialization
        for layer in bbox_reg.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        super().__init__(bbox_reg, 4)


class SSDLiteHead(nn.Module):
    def __init__(self, in_channels: list[int], num_anchors: list[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors)

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }


class ExtraBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.add_module(
            "conv1",
            Conv2dNormActivation(
                in_channels,
                out_channels // 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=nn.ReLU6,
            ),
        )
        self.add_module(
            "conv2",
            Conv2dNormActivation(
                out_channels // 2,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=out_channels // 2,
                activation_layer=nn.ReLU6,
            ),
        )
        self.add_module(
            "conv3",
            Conv2dNormActivation(
                out_channels // 2,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=nn.ReLU6,
            ),
        )


class SSDLite(SSD):
    default_size = (512, 512)
    auto_register = True

    def __init__(
        self,
        num_classes: int,
        backbone: DetectorBackbone,
        *,
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is None, "config not supported"

        iou_thresh = 0.5
        score_thresh = 0.01
        nms_thresh = 0.45
        detections_per_img = 200
        topk_candidates = 300
        positive_fraction = 0.25

        self.backbone.return_channels = self.backbone.return_channels[-2:]
        self.backbone.return_stages = self.backbone.return_stages[-2:]
        self.extra_blocks = nn.ModuleList(
            [
                ExtraBlock(self.backbone.return_channels[-1], 512, stride=(2, 2)),
                ExtraBlock(512, 256, stride=(2, 2)),
                ExtraBlock(256, 256, stride=(1, 1)),
            ]
        )

        self.anchor_generator = DefaultBoxGenerator(
            [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
            steps=[8, 16, 32, 64, 100, 300],
        )
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDLiteHead(self.backbone.return_channels + [512, 256, 256], num_anchors, self.num_classes)
        self.proposal_matcher = SSDMatcher(iou_thresh)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes + 1
        self.head.classification_head = SSDLiteClassificationHead(
            self.backbone.return_channels + [512, 256, 256],
            self.anchor_generator.num_anchors_per_location(),
            self.num_classes,
        )

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.head.classification_head.parameters():
                param.requires_grad = True
