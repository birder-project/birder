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
        size: Optional[int | tuple[int, int]] = None,
        scale_factor: Optional[float | tuple[float, float]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        if mode == "nearest":
            self.align_corners = None

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
        norm_layer: Optional[Callable[..., nn.Module]],
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

                down_inst = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

            else:
                down_inst = Interpolate2d(size=output_size, mode=downsample)

            self.add_module("downsample", down_inst)

        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                self.add_module("upsample", Interpolate2d(size=output_size, mode=upsample))


class FpnCombine(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        fpn_channels: int,
        inputs_offsets: list[int],
        input_size: list[tuple[int, int]],
        output_size: tuple[int, int],
        downsample: Literal["max", "bilinear"],
        upsample: Literal["nearest", "bilinear"],
        norm_layer: Optional[Callable[..., nn.Module]],
        weight_method: Literal["attn", "fastattn", "sum"] = "attn",
    ):
        super().__init__()
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for offset in inputs_offsets:
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels[offset],
                fpn_channels,
                input_size=input_size[offset],
                output_size=output_size,
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
        nodes = []
        for offset, resample in self.resample.items():
            input_node = x[int(offset)]
            input_node = resample(input_node)
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
        input_size: list[tuple[int, int]],
        feat_sizes: list[tuple[int, int]],
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
                input_size=input_size,
                output_size=feat_sizes[fnode_cfg["feat_level"]],
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
        image_size: tuple[int, int],
        min_level: int,
        max_level: int,
        num_levels: int,
        backbone_channels: list[int],
        fpn_channels: int,
        fpn_cell_repeats: int,
        bifpn_config: list[dict[str, Any]],
    ):
        super().__init__()
        feat_size = image_size
        feat_sizes = [feat_size]
        for _ in range(1, max_level + 1):
            feat_size = ((feat_size[0] - 1) // 2 + 1, (feat_size[1] - 1) // 2 + 1)
            feat_sizes.append(feat_size)

        input_size = feat_sizes.copy()
        input_size = input_size[-num_levels:]
        prev_feat_size = feat_sizes[min_level]
        self.resample = nn.ModuleDict()
        for level in range(num_levels):
            feat_size = feat_sizes[level + min_level]
            if level < len(backbone_channels):
                in_channels = backbone_channels[level]
                input_size[level] = feat_size
            else:
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_channels,
                    out_channels=fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    downsample="max",
                    upsample="nearest",
                    norm_layer=nn.BatchNorm2d,
                )
                in_channels = fpn_channels
                backbone_channels.append(in_channels)

            prev_feat_size = feat_size

        self.cells = nn.ModuleList()
        fpn_combine_channels = backbone_channels
        for _ in range(fpn_cell_repeats):
            fpn_combine_channels = fpn_combine_channels + [fpn_channels for _ in bifpn_config]
            input_size = input_size + [feat_sizes[fc["feat_level"]] for fc in bifpn_config]
            fpn_layer = BiFpnLayer(
                in_channels=fpn_combine_channels,
                input_size=input_size,
                feat_sizes=feat_sizes,
                fpn_config=bifpn_config,
                fpn_channels=fpn_channels,
                num_levels=num_levels,
                downsample="max",
                upsample="nearest",
                norm_layer=nn.BatchNorm2d,
            )
            self.cells.append(fpn_layer)
            fpn_combine_channels = fpn_combine_channels[-num_levels::]
            input_size = input_size[-num_levels::]

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for resample in self.resample.values():
            x.append(resample(x[-1]))

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
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS  # pylint: disable=invalid-name

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
        net_param: Optional[float] = None,
        config: Optional[dict[str, Any]] = None,
        size: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(num_classes, backbone, net_param=net_param, config=config, size=size)
        assert self.net_param is None, "net-param not supported"
        assert self.config is not None, "must set config"

        self.num_classes = self.num_classes - 1

        score_thresh = 0.001
        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.3
        nms_thresh = 0.5
        detections_per_img = 100
        topk_candidates = 5000

        min_level = 3
        max_level = 7
        num_levels = max_level - min_level + 1
        anchor_sizes = [[x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))] for x in [32, 64, 128, 256, 512]]
        aspect_ratios = [[0.5, 1.0, 2.0]] * len(anchor_sizes)

        fpn_cell_repeats: int = self.config["fpn_cell_repeats"]
        box_class_repeats: int = self.config["box_class_repeats"]
        fpn_channels: int = self.config["fpn_channels"]
        weight_method: Literal["fastattn", "sum"] = self.config["weight_method"]

        self.box_class_repeats = box_class_repeats
        self.fpn_channels = fpn_channels

        bifpn_config = get_bifpn_config(min_level, max_level, weight_method)
        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]

        self.bifpn = BiFpn(
            image_size=self.size,
            min_level=min_level,
            max_level=max_level,
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
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.class_net = ClassificationHead(
            num_outputs=self.num_classes,
            repeats=self.box_class_repeats,
            fpn_channels=self.fpn_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0],
        )

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

        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        feature_list = list(features.values())
        feature_list = self.bifpn(feature_list)
        cls_logits = self.class_net(feature_list)
        box_output = self.box_net(feature_list)
        anchors = self.anchor_generator(images, feature_list)

        losses = {}
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


registry.register_alias(
    "efficientdet_d0",
    EfficientDet,
    config={"fpn_cell_repeats": 3, "box_class_repeats": 3, "fpn_channels": 64, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d1",
    EfficientDet,
    config={"fpn_cell_repeats": 4, "box_class_repeats": 3, "fpn_channels": 88, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d2",
    EfficientDet,
    config={"fpn_cell_repeats": 5, "box_class_repeats": 3, "fpn_channels": 112, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d3",
    EfficientDet,
    config={"fpn_cell_repeats": 6, "box_class_repeats": 4, "fpn_channels": 160, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d4",
    EfficientDet,
    config={"fpn_cell_repeats": 7, "box_class_repeats": 4, "fpn_channels": 224, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d5",
    EfficientDet,
    config={"fpn_cell_repeats": 7, "box_class_repeats": 4, "fpn_channels": 288, "weight_method": "fastattn"},
)
registry.register_alias(
    "efficientdet_d6",
    EfficientDet,
    config={"fpn_cell_repeats": 8, "box_class_repeats": 5, "fpn_channels": 384, "weight_method": "sum"},
)
