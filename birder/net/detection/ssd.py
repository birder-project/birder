"""
SSD, adapted from
https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py
and from
https://github.com/dmlc/gluon-cv/blob/master/gluoncv/model_zoo/ssd/ssd.py

Paper "SSD: Single Shot MultiBox Detector", https://arxiv.org/abs/1512.02325

Changes from original:
* Changed extra block to be backbone agnostic
"""

# Reference license: BSD 3-Clause

import math
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import boxes as box_ops

from birder.net.base import DetectorBackbone
from birder.net.detection.base import BoxCoder
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import ImageList
from birder.net.detection.base import Matcher


class SSDMatcher(Matcher):
    def __init__(self, threshold: float) -> None:
        super().__init__(threshold, threshold, allow_low_quality_matches=False)

    def __call__(self, match_quality_matrix: torch.Tensor) -> torch.Tensor:
        matches = super().__call__(match_quality_matrix)

        # For each gt, find the prediction with which it has the highest quality
        (_, highest_quality_pred_foreach_gt) = match_quality_matrix.max(dim=1)
        matches[highest_quality_pred_foreach_gt] = torch.arange(
            highest_quality_pred_foreach_gt.size(0),
            dtype=torch.int64,
            device=highest_quality_pred_foreach_gt.device,
        )

        return matches


class DefaultBoxGenerator(nn.Module):
    """
    This module generates the default boxes of SSD for a set of feature maps and image sizes
    """

    def __init__(
        self,
        aspect_ratios: list[list[int]],
        min_ratio: float = 0.15,
        max_ratio: float = 0.9,
        scales: Optional[list[float]] = None,
        steps: Optional[list[int]] = None,
        clip: bool = True,
    ) -> None:
        super().__init__()
        if steps is not None and len(aspect_ratios) != len(steps):
            raise ValueError("aspect_ratios and steps should have the same length")

        self.aspect_ratios = aspect_ratios
        self.steps = steps
        self.clip = clip
        num_outputs = len(aspect_ratios)

        # Estimation of default boxes scales
        if scales is None:
            if num_outputs > 1:
                range_ratio = max_ratio - min_ratio
                self.scales = [min_ratio + range_ratio * k / (num_outputs - 1.0) for k in range(num_outputs)]
                self.scales.append(1.0)

            else:
                self.scales = [min_ratio, max_ratio]

        else:
            self.scales = scales

        self._wh_pairs = self._generate_wh_pairs(num_outputs)

    def _generate_wh_pairs(self, num_outputs: int, dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
        _wh_pairs: list[torch.Tensor] = []
        for k in range(num_outputs):
            # Adding the 2 default width-height pairs for aspect ratio 1 and scale s'k
            s_k = self.scales[k]
            s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
            wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # Adding 2 pairs for each aspect ratio of the feature map k
            for ar in self.aspect_ratios[k]:
                sq_ar = math.sqrt(ar)
                w = self.scales[k] * sq_ar
                h = self.scales[k] / sq_ar
                wh_pairs.extend([[w, h], [h, w]])

            _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype))

        return _wh_pairs

    def num_anchors_per_location(self) -> list[int]:
        # Estimate num of anchors based on aspect ratios: 2 default boxes + 2 * ratios of feature map.
        return [2 + 2 * len(r) for r in self.aspect_ratios]

    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(
        self, grid_sizes: list[list[int]], image_size: list[int], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k = image_size[1] / self.steps[k]
                y_f_k = image_size[0] / self.steps[k]
            else:
                (y_f_k, x_f_k) = f_k

            shifts_x = ((torch.arange(0, f_k[1]) + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((torch.arange(0, f_k[0]) + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

            default_box = torch.concat((shifts, wh_pairs), dim=1)
            default_boxes.append(default_box)

        return torch.concat(default_boxes, dim=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"aspect_ratios={self.aspect_ratios}"
            f", clip={self.clip}"
            f", scales={self.scales}"
            f", steps={self.steps}"
            ")"
        )

    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype = feature_maps[0].dtype
        device = feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)

        dboxes = []
        x_y_size = torch.tensor([image_size[1], image_size[0]], device=default_boxes.device)
        for _ in image_list.image_sizes:
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.concat(
                [
                    (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                    (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                ],
                -1,
            )
            dboxes.append(dboxes_in_image)

        return dboxes


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but TorchScript doesn't support this yet
        """

        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks

        out = x
        for i, module in enumerate(self.module_list):
            if i == idx:
                out = module(x)

        return out

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            (N, _, H, W) = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.concat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: list[int], num_anchors: list[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(
                nn.Conv2d(channels, num_classes * anchors, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )

        # Weights initialization
        for layer in cls_logits.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: list[int], num_anchors: list[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))

        # Weights initialization
        for layer in bbox_reg.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        super().__init__(bbox_reg, 4)


class SSDHead(nn.Module):
    def __init__(self, in_channels: list[int], num_anchors: list[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

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
            Conv2dNormActivation(in_channels, out_channels // 2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )
        self.add_module(
            "conv2",
            Conv2dNormActivation(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
            ),
        )


class SSD(DetectionBaseNet):
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
        self.head = SSDHead(self.backbone.return_channels + [512, 256, 256], num_anchors, self.num_classes)
        self.proposal_matcher = SSDMatcher(iou_thresh)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes + 1
        self.head.classification_head = SSDClassificationHead(
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

    # pylint: disable=too-many-locals
    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        head_outputs: dict[str, torch.Tensor],
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        # Match original targets with default boxes
        num_foreground = 0
        bbox_loss_list = []
        cls_targets_list = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # Produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Calculate regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            bbox_loss_list.append(F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum"))

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=targets_per_image["labels"].dtype,
                device=targets_per_image["labels"].device,
            )
            gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
                foreground_matched_idxs_per_image
            ]
            cls_targets_list.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss_list)
        cls_targets = torch.stack(cls_targets_list)

        # Calculate classification loss
        num_classes = cls_logits.size(-1)
        cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(
            cls_targets.size()
        )

        # Hard negative sampling
        foreground_idxs = cls_targets > 0
        num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        # num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
        negative_loss = cls_loss.clone()
        negative_loss[foreground_idxs] = -float("inf")  # Use -inf to detect positive values that creeped in the sample

        (_values, idx) = negative_loss.sort(1, descending=True)
        # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
        background_idxs = idx.sort(1)[1] < num_negative

        N = max(1, num_foreground)

        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        }

    def postprocess_detections(
        self,
        head_outputs: dict[str, torch.Tensor],
        image_anchors: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.size(-1)
        device = pred_scores.device
        detections: list[dict[str, torch.Tensor]] = []
        for boxes, scores, anchors, image_shape in zip(bbox_regression, pred_scores, image_anchors, image_shapes):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            list_empty = True
            image_boxes_list = []
            image_scores_list = []
            image_labels_list = []
            for label in range(1, num_classes):
                score = scores[:, label]

                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # Keep only topk scoring predictions
                num_topk = min(self.topk_candidates, int(score.size(0)))
                (score, idxs) = score.topk(num_topk)
                box = box[idxs]
                if len(box) == 0 and list_empty is False:
                    continue

                image_boxes_list.append(box)
                image_scores_list.append(score)
                image_labels_list.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
                list_empty = False

            image_boxes = torch.concat(image_boxes_list, dim=0)
            image_scores = torch.concat(image_scores_list, dim=0)
            image_labels = torch.concat(image_labels_list, dim=0)

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

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        features = self.backbone.detection_features(x)
        feature_list = list(features.values())
        for extra_block in self.extra_blocks:
            feature_list.append(extra_block(feature_list[-1]))

        head_outputs = self.head(feature_list)
        anchors = self.anchor_generator(images, feature_list)

        losses = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"

            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full(
                            (anchors_per_image.size(0),),
                            -1,
                            dtype=torch.int64,
                            device=anchors_per_image.device,
                        )
                    )
                    continue

                match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))

            losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)

        else:
            detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)

        return (detections, losses)
