"""
FCOS, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/fcos.py

Paper "FCOS: Fully Convolutional One-Stage Object Detection", https://arxiv.org/abs/1904.01355
and
Paper "FCOS: A simple and strong anchor-free object detector", https://arxiv.org/abs/2006.09214
"""

# Reference license: BSD 3-Clause

import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import boxes as box_ops
from torchvision.ops import generalized_box_iou_loss
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from birder.net.base import DetectorBackbone
from birder.net.detection.base import AnchorGenerator
from birder.net.detection.base import BackboneWithFPN
from birder.net.detection.base import DetectionBaseNet


class BoxLinearCoder:
    """
    The linear box-to-box transform defined in FCOS. The transformation is parameterized
    by the distance from the center of (square) src box to 4 edges of the target box.
    """

    def __init__(self, normalize_by_size: bool) -> None:
        self.normalize_by_size = normalize_by_size

    def encode(self, reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
        """
        Encode a set of proposals with respect to some reference boxes
        """

        # Get the center of reference_boxes
        reference_boxes_ctr_x = 0.5 * (reference_boxes[..., 0] + reference_boxes[..., 2])
        reference_boxes_ctr_y = 0.5 * (reference_boxes[..., 1] + reference_boxes[..., 3])

        # Get box regression transformation deltas
        target_l = reference_boxes_ctr_x - proposals[..., 0]
        target_t = reference_boxes_ctr_y - proposals[..., 1]
        target_r = proposals[..., 2] - reference_boxes_ctr_x
        target_b = proposals[..., 3] - reference_boxes_ctr_y

        targets = torch.stack((target_l, target_t, target_r, target_b), dim=-1)

        if self.normalize_by_size is True:
            reference_boxes_w = reference_boxes[..., 2] - reference_boxes[..., 0]
            reference_boxes_h = reference_boxes[..., 3] - reference_boxes[..., 1]
            reference_boxes_size = torch.stack(
                (reference_boxes_w, reference_boxes_h, reference_boxes_w, reference_boxes_h), dim=-1
            )
            targets = targets / reference_boxes_size

        return targets

    def decode(self, rel_codes: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        """
        From a set of original boxes and encoded relative box offsets, get the decoded boxes

        This method assumes that 'rel_codes' and 'boxes' have same size for 0th dimension.
        """

        boxes = boxes.to(dtype=rel_codes.dtype)

        ctr_x = 0.5 * (boxes[..., 0] + boxes[..., 2])
        ctr_y = 0.5 * (boxes[..., 1] + boxes[..., 3])
        if self.normalize_by_size is True:
            boxes_w = boxes[..., 2] - boxes[..., 0]
            boxes_h = boxes[..., 3] - boxes[..., 1]

            list_box_size = torch.stack((boxes_w, boxes_h, boxes_w, boxes_h), dim=-1)
            rel_codes = rel_codes * list_box_size

        pred_boxes1 = ctr_x - rel_codes[..., 0]
        pred_boxes2 = ctr_y - rel_codes[..., 1]
        pred_boxes3 = ctr_x + rel_codes[..., 2]
        pred_boxes4 = ctr_y + rel_codes[..., 3]

        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=-1)

        return pred_boxes


class FCOSClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        # Weights initialization
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_cls_logits = []
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            (N, _, H, W) = cls_logits.size()
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # (N, HWA, 4)

            all_cls_logits.append(cls_logits)

        return torch.concat(all_cls_logits, dim=1)


class FCOSRegressionHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_convs: int) -> None:
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            conv.append(nn.GroupNorm(32, in_channels))
            conv.append(nn.ReLU())

        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bbox_ctrness = nn.Conv2d(in_channels, num_anchors * 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Weights initialization
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        all_bbox_regression = []
        all_bbox_ctrness = []
        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = F.relu(self.bbox_reg(bbox_feature))
            bbox_ctrness = self.bbox_ctrness(bbox_feature)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            (N, _, H, W) = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # (N, HWA, 4)
            all_bbox_regression.append(bbox_regression)

            # Permute bbox center-ness output from (N, 1 * A, H, W) to (N, HWA, 1).
            bbox_ctrness = bbox_ctrness.view(N, -1, 1, H, W)
            bbox_ctrness = bbox_ctrness.permute(0, 3, 4, 1, 2)
            bbox_ctrness = bbox_ctrness.reshape(N, -1, 1)
            all_bbox_ctrness.append(bbox_ctrness)

        return (torch.concat(all_bbox_regression, dim=1), torch.concat(all_bbox_ctrness, dim=1))


class FCOSHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int, num_convs: int) -> None:
        super().__init__()
        self.num_convs = num_convs
        self.box_coder = BoxLinearCoder(normalize_by_size=True)
        self.classification_head = FCOSClassificationHead(in_channels, num_anchors, num_classes, num_convs)
        self.regression_head = FCOSRegressionHead(in_channels, num_anchors, num_convs)

    # pylint: disable=too-many-locals
    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors_list: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cls_logits = head_outputs["cls_logits"]  # [N, HWA, C]
        bbox_regression = head_outputs["bbox_regression"]  # [N, HWA, 4]
        bbox_ctrness = head_outputs["bbox_ctrness"]  # [N, HWA, 1]

        all_gt_classes_targets_list = []
        all_gt_boxes_targets_list = []
        for targets_per_image, matched_idxs_per_image in zip(targets, matched_idxs):
            if len(targets_per_image["labels"]) == 0:
                gt_classes_targets = targets_per_image["labels"].new_zeros((len(matched_idxs_per_image),))
                gt_boxes_targets = targets_per_image["boxes"].new_zeros((len(matched_idxs_per_image), 4))
            else:
                gt_classes_targets = targets_per_image["labels"][matched_idxs_per_image.clip(min=0)]
                gt_boxes_targets = targets_per_image["boxes"][matched_idxs_per_image.clip(min=0)]

            gt_classes_targets[matched_idxs_per_image < 0] = -1  # Background
            all_gt_classes_targets_list.append(gt_classes_targets)
            all_gt_boxes_targets_list.append(gt_boxes_targets)

        all_gt_classes_targets = torch.stack(all_gt_classes_targets_list)
        all_gt_boxes_targets = torch.stack(all_gt_boxes_targets_list)
        anchors = torch.stack(anchors_list)

        # Compute foreground
        foreground_mask = all_gt_classes_targets >= 0
        num_foreground = foreground_mask.sum().item()

        # Classification loss
        gt_classes_targets = torch.zeros_like(cls_logits)
        gt_classes_targets[foreground_mask, all_gt_classes_targets[foreground_mask]] = 1.0
        loss_cls = sigmoid_focal_loss(cls_logits, gt_classes_targets, reduction="sum")

        pred_boxes = self.box_coder.decode(bbox_regression, anchors)

        # Regression loss: GIoU loss
        loss_bbox_reg = generalized_box_iou_loss(
            pred_boxes[foreground_mask],
            all_gt_boxes_targets[foreground_mask],
            reduction="sum",
        )

        # Center-ness loss
        bbox_reg_targets = self.box_coder.encode(anchors, all_gt_boxes_targets)

        if len(bbox_reg_targets) == 0:
            gt_ctrness_targets = bbox_reg_targets.new_zeros(bbox_reg_targets.size()[:-1])
        else:
            left_right = bbox_reg_targets[:, :, [0, 2]]
            top_bottom = bbox_reg_targets[:, :, [1, 3]]
            gt_ctrness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )

        pred_centerness = bbox_ctrness.squeeze(dim=2)
        loss_bbox_ctrness = F.binary_cross_entropy_with_logits(
            pred_centerness[foreground_mask], gt_ctrness_targets[foreground_mask], reduction="sum"
        )

        return {
            "classification": loss_cls / max(1, num_foreground),
            "bbox_regression": loss_bbox_reg / max(1, num_foreground),
            "bbox_ctrness": loss_bbox_ctrness / max(1, num_foreground),
        }

    def forward(self, x: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        cls_logits = self.classification_head(x)
        (bbox_regression, bbox_ctrness) = self.regression_head(x)

        return {
            "cls_logits": cls_logits,
            "bbox_regression": bbox_regression,
            "bbox_ctrness": bbox_ctrness,
        }


class FCOS(DetectionBaseNet):
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
        self.center_sampling_radius = 1.5
        self.score_thresh = 0.2
        self.nms_thresh = 0.6
        self.detections_per_img = 100
        self.topk_candidates = 1000

        self.backbone.return_channels = self.backbone.return_channels[1:]
        self.backbone.return_stages = self.backbone.return_stages[1:]
        self.backbone_with_fpn = BackboneWithFPN(
            # Skip stage1
            self.backbone,
            fpn_width,
            extra_blocks=LastLevelP6P7(256, 256),
        )

        anchor_sizes = [[8], [16], [32], [64], [128]]  # Equal to strides of multi-level feature map
        aspect_ratios = [[1.0]] * len(anchor_sizes)  # Set only one anchor
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        self.head = FCOSHead(
            self.backbone_with_fpn.out_channels,
            self.anchor_generator.num_anchors_per_location()[0],
            self.num_classes,
            num_convs=4,
        )

        self.box_coder = BoxLinearCoder(normalize_by_size=True)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes

        self.head.classification_head = FCOSClassificationHead(
            self.backbone_with_fpn.out_channels,
            self.anchor_generator.num_anchors_per_location()[0],
            self.num_classes,
            self.head.num_convs,
        )

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        num_anchors_per_level: list[int],
    ) -> dict[str, torch.Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            gt_boxes = targets_per_image["boxes"]
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2  # Nx2
            anchor_centers = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2  # N
            anchor_sizes = anchors_per_image[:, 2] - anchors_per_image[:, 0]

            # Center sampling: anchor point must be close enough to gt center.
            pairwise_match = (anchor_centers[:, None, :] - gt_centers[None, :, :]).abs_().max(
                dim=2
            ).values < self.center_sampling_radius * anchor_sizes[:, None]

            # Compute pairwise distance between N points and M boxes
            (x, y) = anchor_centers.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
            (x0, y0, x1, y1) = gt_boxes.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
            pairwise_dist = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)  # (N, M)

            # Anchor point must be inside gt
            pairwise_match &= pairwise_dist.min(dim=2).values > 0

            # Each anchor is only responsible for certain scale range.
            lower_bound = anchor_sizes * 4
            lower_bound[: num_anchors_per_level[0]] = 0
            upper_bound = anchor_sizes * 8
            upper_bound[-num_anchors_per_level[-1] :] = float("inf")
            pairwise_dist = pairwise_dist.max(dim=2).values
            pairwise_match &= (pairwise_dist > lower_bound[:, None]) & (pairwise_dist < upper_bound[:, None])

            # Match the GT box with minimum area, if there are multiple GT matches
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])  # N
            pairwise_match = pairwise_match.to(torch.float32) * (1e8 - gt_areas[None, :])
            min_values, matched_idx = pairwise_match.max(dim=1)  # R, per-anchor match
            matched_idx[min_values < 1e-5] = -1  # Unmatched anchors are assigned -1

            matched_idxs.append(matched_idx)

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
        box_ctrness = head_outputs["bbox_ctrness"]

        num_images = len(image_shapes)
        detections: list[dict[str, torch.Tensor]] = []
        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            box_ctrness_per_image = [bc[index] for bc in box_ctrness]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []
            for box_regression_per_level, logits_per_level, box_ctrness_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, box_ctrness_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # Remove low scoring boxes
                scores_per_level = torch.sqrt(
                    torch.sigmoid(logits_per_level) * torch.sigmoid(box_ctrness_per_level)
                ).flatten()
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

                boxes_per_level = self.box_coder.decode(
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

    def forward(  # type: ignore[override]
        self, x: torch.Tensor, targets: Optional[list[dict[str, torch.Tensor]]] = None
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x)

        features: dict[str, torch.Tensor] = self.backbone_with_fpn(x)
        feature_list = list(features.values())
        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(images, feature_list)

        # recover level sizes
        num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"
            for idx, target in enumerate(targets):
                targets[idx]["labels"] = target["labels"] - 1  # No background

            losses = self.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)

        else:
            # Split outputs per level
            split_head_outputs: dict[str, list[torch.Tensor]] = {}
            for k in head_outputs:
                split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))

            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # Compute the detections
            detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)

        return (detections, losses)
