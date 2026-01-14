"""
EfficientDet, adapted from
https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/efficientdet.py

Paper "EfficientDet: Scalable and Efficient Object Detection", https://arxiv.org/abs/1911.09070
"""

# Reference license: Apache-2.0

import itertools
from collections.abc import Callable
from functools import partial
from typing import Any
from typing import Literal
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss

from birder.model_registry import registry
from birder.net.base import DetectorBackbone
from birder.net.detection.base import AnchorGenerator
from birder.net.detection.base import BoxCoder
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import Matcher
from birder.ops.soft_nms import SoftNMS


def get_bifpn_config(min_level: int, max_level: int, weight_method: Literal["fastattn", "sum"]) -> list[dict[str, Any]]:
    num_levels = max_level - min_level + 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    id_cnt = itertools.count(num_levels)

    nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # Top-down
        nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": [node_ids[i][-1], node_ids[i + 1][-1]],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # Bottom-up
        nodes.append(
            {
                "feat_level": i,
                "inputs_offsets": node_ids[i] + [node_ids[i - 1][-1]],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return nodes


def _sum(x: list[torch.Tensor]) -> torch.Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i

    return res


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
        mode: str = "nearest",
        align_corners: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners
        if mode == "nearest":
            self.align_corners = None

    def forward(self, x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        size_list = [size[0], size[1]]
        return F.interpolate(x, size_list, None, self.mode, self.align_corners, recompute_scale_factor=False)


class ResampleFeatureMap(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Literal["max", "bilinear"],
        upsample: Literal["nearest", "bilinear"],
        norm_layer: Optional[Callable[..., nn.Module]],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_mode = downsample

        if in_channels != out_channels:
            # padding = ((stride - 1) + (kernel_size - 1)) // 2
            self.conv = Conv2dNormActivation(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=norm_layer,
                bias=False,
                activation_layer=None,
            )
        else:
            self.conv = None

        self.downsample = None
        if downsample != "max":
            self.downsample = Interpolate2d(mode=downsample)

        self.upsample = Interpolate2d(mode=upsample)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)

        (in_h, in_w) = x.shape[-2:]
        (target_h, target_w) = target_size
        if in_h == target_h and in_w == target_w:
            return x

        downsample_needed = in_h > target_h or in_w > target_w
        upsample_needed = in_h < target_h or in_w < target_w

        if downsample_needed is True and upsample_needed is False:
            if self.downsample_mode == "max":
                stride_size_h = int((in_h - 1) // target_h + 1)
                stride_size_w = int((in_w - 1) // target_w + 1)
                kernel_size = (stride_size_h + 1, stride_size_w + 1)
                stride = (stride_size_h, stride_size_w)
                padding = (
                    ((stride[0] - 1) + (kernel_size[0] - 1)) // 2,
                    ((stride[1] - 1) + (kernel_size[1] - 1)) // 2,
                )
                return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)

            if self.downsample is not None:
                return self.downsample(x, size=target_size)

        if upsample_needed is True and downsample_needed is False:
            return self.upsample(x, size=target_size)

        if self.downsample is not None and self.downsample_mode != "max":
            return self.downsample(x, size=target_size)

        return self.upsample(x, size=target_size)


class FpnCombine(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        fpn_channels: int,
        inputs_offsets: list[int],
        downsample: Literal["max", "bilinear"],
        upsample: Literal["nearest", "bilinear"],
        norm_layer: Optional[Callable[..., nn.Module]],
        weight_method: Literal["attn", "fastattn", "sum"] = "attn",
    ):
        super().__init__()
        self.weight_method = weight_method
        self.inputs_offsets = inputs_offsets
        self.target_offset = inputs_offsets[0]

        self.resample = nn.ModuleDict()
        for offset in inputs_offsets:
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels[offset],
                fpn_channels,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
            )

        if weight_method in {"attn", "fastattn"}:
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        dtype = x[0].dtype
        target = x[self.target_offset]
        target_size = (int(target.shape[-2]), int(target.shape[-1]))
        nodes = []
        for offset, resample in self.resample.items():
            input_node = x[int(offset)]
            input_node = resample(input_node, target_size=target_size)
            nodes.append(input_node)

        if self.weight_method == "attn":
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == "fastattn":
            edge_weights = F.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1
            )
        elif self.weight_method == "sum":
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError(f"unknown weight_method {self.weight_method}")

        out = torch.sum(out, dim=-1)
        return out


class FNode(nn.Module):
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super().__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        fpn_config: list[dict[str, Any]],
        fpn_channels: int,
        num_levels: int,
        downsample: Literal["max", "bilinear"],
        upsample: Literal["nearest", "bilinear"],
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.fnode = nn.ModuleList()
        for fnode_cfg in fpn_config:
            combine = FpnCombine(
                in_channels,
                fpn_channels,
                inputs_offsets=fnode_cfg["inputs_offsets"],
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                weight_method=fnode_cfg["weight_method"],
            )

            after_combine = nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    groups=fpn_channels,
                    bias=False,
                ),
                Conv2dNormActivation(
                    fpn_channels,
                    fpn_channels,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    norm_layer=norm_layer,
                    activation_layer=None,
                ),
            )

            self.fnode.append(FNode(combine=combine, after_combine=after_combine))

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for fn in self.fnode:
            x.append(fn(x))

        return x[-self.num_levels : :]


class BiFpn(nn.Module):
    def __init__(
        self,
        num_levels: int,
        backbone_channels: list[int],
        fpn_channels: int,
        fpn_cell_repeats: int,
        bifpn_config: list[dict[str, Any]],
    ):
        super().__init__()
        self.resample = nn.ModuleList()
        num_backbone_levels = len(backbone_channels)
        extra_levels = max(0, num_levels - num_backbone_levels)
        in_channels = backbone_channels[-1]
        for _ in range(extra_levels):
            self.resample.append(
                ResampleFeatureMap(
                    in_channels=in_channels,
                    out_channels=fpn_channels,
                    downsample="max",
                    upsample="nearest",
                    norm_layer=nn.BatchNorm2d,
                )
            )
            in_channels = fpn_channels
            backbone_channels.append(in_channels)

        self.cells = nn.ModuleList()
        fpn_combine_channels = backbone_channels
        for _ in range(fpn_cell_repeats):
            fpn_combine_channels = fpn_combine_channels + [fpn_channels for _ in bifpn_config]
            fpn_layer = BiFpnLayer(
                in_channels=fpn_combine_channels,
                fpn_config=bifpn_config,
                fpn_channels=fpn_channels,
                num_levels=num_levels,
                downsample="max",
                upsample="nearest",
                norm_layer=nn.BatchNorm2d,
            )
            self.cells.append(fpn_layer)
            fpn_combine_channels = fpn_combine_channels[-num_levels::]

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for resample in self.resample:
            input_node = x[-1]
            target_size = ((input_node.shape[-2] - 1) // 2 + 1, (input_node.shape[-1] - 1) // 2 + 1)
            x.append(resample(input_node, target_size=target_size))

        for cell in self.cells:
            x = cell(x)

        return x


class HeadNet(nn.Module):
    def __init__(self, num_outputs: int, repeats: int, fpn_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.num_outputs = num_outputs
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
                    padding=(0, 0),
                    norm_layer=norm_layer,
                    bias=False,
                    activation_layer=nn.SiLU,
                )
            )

        self.conv_repeat = nn.Sequential(*layers)
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
                padding=(0, 0),
                bias=True,
            ),
        )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class ClassificationHead(HeadNet):
    def __init__(self, num_outputs: int, repeats: int, fpn_channels: int, num_anchors: int) -> None:
        super().__init__(num_outputs, repeats, fpn_channels, num_anchors)
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_logits: torch.Tensor,
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
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
                    gamma=1.5,
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits: torch.Tensor = self.conv_repeat(features)
            cls_logits = self.predict(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            (N, _, H, W) = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_outputs, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_outputs)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.concat(all_cls_logits, dim=1)


class RegressionHead(HeadNet):
    def __init__(self, num_outputs: int, repeats: int, fpn_channels: int, num_anchors: int) -> None:
        super().__init__(num_outputs, repeats, fpn_channels, num_anchors)
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        bbox_regression: torch.Tensor,
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = []
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
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            losses.append(
                F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum") / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_bbox_regression = []

        for features in x:
            bbox_regression: torch.Tensor = self.conv_repeat(features)
            bbox_regression = self.predict(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            (N, _, H, W) = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.concat(all_bbox_regression, dim=1)


class EfficientDet(DetectionBaseNet):
    default_size = (640, 640)

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
        assert self.config is not None, "must set config"

        self.num_classes = self.num_classes - 1

        min_level = 3
        max_level = 7
        num_levels = max_level - min_level + 1
        anchor_sizes = [[x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))] for x in [32, 64, 128, 256, 512]]
        aspect_ratios = [[0.5, 1.0, 2.0]] * len(anchor_sizes)

        score_thresh = 0.001
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.3
        topk_candidates = 2000
        fpn_cell_repeats: int = self.config["fpn_cell_repeats"]
        box_class_repeats: int = self.config["box_class_repeats"]
        fpn_channels: int = self.config["fpn_channels"]
        weight_method: Literal["fastattn", "sum"] = self.config["weight_method"]
        detections_per_img: int = self.config.get("detections_per_img", 100)
        nms_thresh: float = self.config.get("nms_thresh", 0.5)
        soft_nms: bool = self.config.get("soft_nms", False)

        self.box_class_repeats = box_class_repeats
        self.fpn_channels = fpn_channels
        self.soft_nms = None
        if soft_nms is True:
            self.soft_nms = SoftNMS()

        bifpn_config = get_bifpn_config(min_level, max_level, weight_method)
        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]

        self.bifpn = BiFpn(
            num_levels=num_levels,
            backbone_channels=self.backbone.return_channels,
            fpn_channels=fpn_channels,
            fpn_cell_repeats=fpn_cell_repeats,
            bifpn_config=bifpn_config,
        )
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.class_net = ClassificationHead(
            num_outputs=self.num_classes,
            repeats=box_class_repeats,
            fpn_channels=fpn_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
        )
        self.box_net = RegressionHead(
            num_outputs=4,
            repeats=box_class_repeats,
            fpn_channels=fpn_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
        )
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        self.score_thresh = score_thresh
        self.topk_candidates = topk_candidates
        self.detections_per_img = detections_per_img
        self.nms_thresh = nms_thresh

        if self.export_mode is False:
            self.forward = torch.compiler.disable(recursive=False)(self.forward)  # type: ignore[method-assign]

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.class_net = ClassificationHead(
            num_outputs=self.num_classes,
            repeats=self.box_class_repeats,
            fpn_channels=self.fpn_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
        )

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad_(False)

        if freeze_classifier is False:
            for param in self.class_net.parameters():
                param.requires_grad_(True)

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_logits: torch.Tensor,
        box_output: torch.Tensor,
        anchors: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        return {
            "classification": self.class_net.compute_loss(targets, cls_logits, matched_idxs),
            "bbox_regression": self.box_net.compute_loss(targets, box_output, anchors, matched_idxs),
        }

    # pylint: disable=too-many-locals
    def postprocess_detections(
        self,
        class_logits: list[torch.Tensor],
        box_regression: list[torch.Tensor],
        anchors: list[list[torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        num_images = len(image_shapes)

        detections: list[dict[str, torch.Tensor]] = []
        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image = anchors[index]
            image_shape = image_shapes[index]

            image_boxes_list = []
            image_scores_list = []
            image_labels_list = []
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

                image_boxes_list.append(boxes_per_level)
                image_scores_list.append(scores_per_level)
                image_labels_list.append(labels_per_level)

            image_boxes = torch.concat(image_boxes_list, dim=0)
            image_scores = torch.concat(image_scores_list, dim=0)
            image_labels = torch.concat(image_labels_list, dim=0)

            # Non-maximum suppression
            if self.soft_nms is not None:
                (soft_scores, keep) = self.soft_nms(image_boxes, image_scores, image_labels, score_threshold=0.001)
                image_scores[keep] = soft_scores
            else:
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
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[list[int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        feature_list = list(features.values())
        feature_list = self.bifpn(feature_list)
        cls_logits = self.class_net(feature_list)
        box_output = self.box_net(feature_list)
        anchors = self.anchor_generator(images, feature_list)

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"
            for idx, target in enumerate(targets):
                targets[idx]["labels"] = target["labels"] - 1  # No background

            losses = self.compute_loss(targets, cls_logits, box_output, anchors)

        else:
            # Recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in feature_list]
            HW = 0
            for v in num_anchors_per_level:
                HW += v

            HWA = cls_logits.size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # Split outputs per level
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # Compute the detections
            detections = self.postprocess_detections(
                list(cls_logits.split(num_anchors_per_level, dim=1)),
                list(box_output.split(num_anchors_per_level, dim=1)),
                split_anchors,
                images.image_sizes,
            )

        return (detections, losses)


registry.register_model_config(
    "efficientdet_d0",
    EfficientDet,
    config={"fpn_cell_repeats": 3, "box_class_repeats": 3, "fpn_channels": 64, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d1",
    EfficientDet,
    config={"fpn_cell_repeats": 4, "box_class_repeats": 3, "fpn_channels": 88, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d2",
    EfficientDet,
    config={"fpn_cell_repeats": 5, "box_class_repeats": 3, "fpn_channels": 112, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d3",
    EfficientDet,
    config={"fpn_cell_repeats": 6, "box_class_repeats": 4, "fpn_channels": 160, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d4",
    EfficientDet,
    config={"fpn_cell_repeats": 7, "box_class_repeats": 4, "fpn_channels": 224, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d5",
    EfficientDet,
    config={"fpn_cell_repeats": 7, "box_class_repeats": 4, "fpn_channels": 288, "weight_method": "fastattn"},
)
registry.register_model_config(
    "efficientdet_d6",
    EfficientDet,
    config={"fpn_cell_repeats": 8, "box_class_repeats": 5, "fpn_channels": 384, "weight_method": "sum"},
)
