"""
RetinaNet, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py

Paper "Focal Loss for Dense Object Detection", https://arxiv.org/abs/1708.02002
"""

# Reference license: BSD 3-Clause

import math
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import boxes as box_ops
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from birder.net.base import DetectorBackbone
from birder.net.detection.base import AnchorGenerator
from birder.net.detection.base import BackboneWithFPN
from birder.net.detection.base import BoxCoder
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import Matcher


def _sum(x: list[torch.Tensor]) -> torch.Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i

    return res


class RetinaNetClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        prior_probability: float = 0.01,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))

        self.conv = nn.Sequential(*conv)

        # Weights initialization
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)

        # Weights initialization
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability))

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS  # pylint: disable=invalid-name

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # Create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # Find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # Compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits: torch.Tensor = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            (N, _, H, W) = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.concat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        giou_loss: bool = True,
    ) -> None:
        super().__init__()
        self.giou_loss = giou_loss

        conv = []
        for _ in range(4):
            conv.append(Conv2dNormActivation(in_channels, in_channels, norm_layer=norm_layer))

        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        nn.init.normal_(self.bbox_reg.weight, std=0.01)
        nn.init.zeros_(self.bbox_reg.bias)

        # Weights initialization
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # Determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # Select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # Compute the loss
            if self.giou_loss is True:
                bbox_per_image = self.box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
                loss = generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=1e-7)
            else:
                target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
                loss = F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum") / max(1, num_foreground)

            losses.append(loss / max(1, num_foreground))

        return _sum(losses) / max(1, len(targets))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_bbox_regression = []

        for features in x:
            bbox_regression: torch.Tensor = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            (N, _, H, W) = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.concat(all_bbox_regression, dim=1)


class RetinaNetHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        num_classes: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        giou_loss: bool = True,
    ) -> None:
        super().__init__()
        self.norm_layer = norm_layer
        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer=norm_layer
        )
        self.regression_head = RetinaNetRegressionHead(
            in_channels, num_anchors, norm_layer=norm_layer, giou_loss=giou_loss
        )

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            "classification": self.classification_head.compute_loss(targets, head_outputs, matched_idxs),
            "bbox_regression": self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x),
        }


class RetinaNet(DetectionBaseNet):
    default_size = (640, 640)
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

        self.num_classes = self.num_classes - 1

        fpn_width = 256
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.4
        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 300
        topk_candidates = 1000
        giou_loss = True

        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]
        self.backbone_with_fpn = BackboneWithFPN(
            # Skip stage1 because it generates too many anchors (according to their paper)
            self.backbone,
            fpn_width,
            extra_blocks=LastLevelP6P7(self.backbone.return_channels[-1], 256),
        )

        anchor_sizes = [[x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))] for x in [32, 64, 128, 256, 512]]
        aspect_ratios = [[0.5, 1.0, 2.0]] * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        self.head = RetinaNetHead(
            self.backbone_with_fpn.out_channels,
            self.anchor_generator.num_anchors_per_location()[0],
            self.num_classes,
            norm_layer=partial(nn.GroupNorm, 32),
            giou_loss=giou_loss,
        )
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes

        norm_layer = self.head.norm_layer
        self.head.classification_head = RetinaNetClassificationHead(
            self.backbone_with_fpn.out_channels,
            self.anchor_generator.num_anchors_per_location()[0],
            self.num_classes,
            norm_layer=norm_layer,
        )

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        matched_idxs = []
        for idx, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))
            targets[idx]["labels"] = targets_per_image["labels"] - 1  # No background

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    # pylint: disable=too-many-locals
    def postprocess_detections(
        self,
        head_outputs: dict[str, list[torch.Tensor]],
        anchors: list[list[torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: list[dict[str, torch.Tensor]] = []
        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image = anchors[index]
            image_shape = image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []
            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # Remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # Keep only topk scoring predictions
                num_topk = min(self.topk_candidates, int(topk_idxs.size(0)))
                (scores_per_level, idxs) = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes
                labels_per_level += 1  # Background offset

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.concat(image_boxes, dim=0)
            image_scores = torch.concat(image_scores, dim=0)
            image_labels = torch.concat(image_labels, dim=0)

            # Non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    # pylint: disable=invalid-name
    def forward(  # type: ignore[override]
        self, x: torch.Tensor, targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x)

        features: dict[str, torch.Tensor] = self.backbone_with_fpn(x)
        feature_list = list(features.values())
        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(images, feature_list)

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"
            losses = self.compute_loss(targets, head_outputs, anchors)

        else:
            # Recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]
            HW = 0
            for v in num_anchors_per_level:
                HW += v

            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # Split outputs per level
            split_head_outputs: dict[str, list[torch.Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))

            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # Compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)

        return (detections, losses)
