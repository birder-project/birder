"""
EfficientDet, adapted from
https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/efficientdet.py

Paper "EfficientDet: Scalable and Efficient Object Detection", https://arxiv.org/abs/1911.09070

Changes from original:
* Head BatchNorm layers are shared across feature-pyramid levels instead of using per-level statistics
* Applies a score threshold before selecting the global top N pre-NMS candidates
"""

# Reference license: Apache-2.0

import itertools
import math
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
from birder.net.detection.base import BoxCoder
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.base import ImageList
from birder.net.detection.base import Matcher
from birder.net.detection.base import clip_boxes_to_image
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


class EfficientDetAnchorGenerator(nn.Module):
    def __init__(
        self,
        num_levels: int,
        num_scales: int = 3,
        aspect_ratios: tuple[float, ...] = (1.0, 2.0, 0.5),
        anchor_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.aspect_ratios = aspect_ratios
        self.anchor_scale = anchor_scale

    def num_anchors_per_location(self) -> list[int]:
        num_anchors = self.num_scales * len(self.aspect_ratios)
        return [num_anchors for _ in range(self.num_levels)]

    def forward(self, image_list: ImageList, feature_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        torch._assert(
            len(feature_maps) == self.num_levels,
            f"Expected {self.num_levels} feature maps, got {len(feature_maps)}",
        )

        image_height, image_width = image_list.tensors.shape[-2:]
        dtype = image_list.tensors.dtype
        device = feature_maps[0].device
        octave_scales = 2 ** (torch.arange(self.num_scales, dtype=dtype, device=device) / float(self.num_scales))
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=dtype, device=device)
        aspect_x = torch.sqrt(aspect_ratios)
        aspect_y = 1.0 / aspect_x

        anchors_over_all_feature_maps = []
        for feature_map in feature_maps:
            grid_height, grid_width = feature_map.shape[-2:]
            stride_height = image_height / grid_height
            stride_width = image_width / grid_width
            base_anchor_widths = (
                self.anchor_scale * stride_width * octave_scales[:, None] * aspect_x[None, :]
            ).reshape(-1)
            base_anchor_heights = (
                self.anchor_scale * stride_height * octave_scales[:, None] * aspect_y[None, :]
            ).reshape(-1)
            base_anchors = (
                torch.stack(
                    [-base_anchor_widths, -base_anchor_heights, base_anchor_widths, base_anchor_heights],
                    dim=1,
                )
                / 2
            )

            shifts_x = (torch.arange(grid_width, dtype=dtype, device=device) + 0.5) * stride_width
            shifts_y = (torch.arange(grid_height, dtype=dtype, device=device) + 0.5) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack(
                [shift_x.reshape(-1), shift_y.reshape(-1), shift_x.reshape(-1), shift_y.reshape(-1)],
                dim=1,
            )
            anchors = (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)
            anchors_over_all_feature_maps.append(anchors)

        anchors_in_image = torch.concat(anchors_over_all_feature_maps, dim=0)
        return [anchors_in_image for _ in range(len(image_list.image_sizes))]


class EfficientDetMatcher(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.matcher = Matcher(threshold, threshold)

    def forward(self, match_quality_matrix: torch.Tensor) -> torch.Tensor:
        matches = self.matcher(match_quality_matrix)

        # Select the first best anchor for every ground-truth box, then prefer
        # the lowest ground-truth index if multiple boxes select the same anchor
        best_anchor_per_target = torch.argmax(match_quality_matrix, dim=1)
        target_indices = torch.arange(
            match_quality_matrix.size(0), dtype=torch.int64, device=match_quality_matrix.device
        )
        forced_matches = torch.full_like(matches, match_quality_matrix.size(0))
        forced_matches.scatter_reduce_(
            0,
            best_anchor_per_target,
            target_indices,
            reduce="amin",
            include_self=True,
        )
        force_match_mask = forced_matches < match_quality_matrix.size(0)

        return torch.where(force_match_mask, forced_matches, matches)


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

        in_h, in_w = x.shape[-2:]
        target_h, target_w = target_size
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
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)))  # WSM
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
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        backbone_channels = backbone_channels.copy()
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
                    norm_layer=norm_layer,
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
                norm_layer=norm_layer,
            )
            self.cells.append(fpn_layer)
            fpn_combine_channels = fpn_combine_channels[-num_levels::]

        # Weights initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                receptive_field_size = module.weight[0, 0].numel()
                fan_in = module.weight.size(1) * receptive_field_size
                fan_out = module.weight.size(0) * receptive_field_size // module.groups
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                nn.init.uniform_(module.weight, -limit, limit)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

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
                    bias=False,
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
                bias=False,
            ),
            nn.Conv2d(fpn_channels, num_outputs * num_anchors, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

        # Weights initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                receptive_field_size = module.weight[0, 0].numel()
                fan_in = module.weight.size(1) * receptive_field_size
                nn.init.normal_(module.weight, std=math.sqrt(1.0 / fan_in))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class ClassificationHead(HeadNet):
    def __init__(self, num_outputs: int, repeats: int, fpn_channels: int, num_anchors: int) -> None:
        super().__init__(num_outputs, repeats, fpn_channels, num_anchors)
        self.BETWEEN_THRESHOLDS = Matcher.BETWEEN_THRESHOLDS

        # Weight initialization
        prior_probability = 0.01
        nn.init.constant_(self.predict[-1].bias, -math.log((1 - prior_probability) / prior_probability))

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        cls_logits: torch.Tensor,
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = cls_logits.float().sum() * 0.0
        num_foreground = torch.zeros((), dtype=torch.float32, device=cls_logits.device)
        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(targets, cls_logits, matched_idxs):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground += foreground_idxs_per_image.sum()

            # Create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0

            # Find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # Compute the classification loss
            loss += sigmoid_focal_loss(
                cls_logits_per_image[valid_idxs_per_image].float(),
                gt_classes_target[valid_idxs_per_image].float(),
                alpha=0.25,
                gamma=1.5,
                reduction="sum",
            )

        return loss / (num_foreground + 1.0)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_cls_logits = []

        for features in x:
            cls_logits: torch.Tensor = self.conv_repeat(features)
            cls_logits = self.predict(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_outputs, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_outputs)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.concat(all_cls_logits, dim=1)


class RegressionHead(HeadNet):
    def __init__(self, num_outputs: int, repeats: int, fpn_channels: int, num_anchors: int) -> None:
        super().__init__(num_outputs, repeats, fpn_channels, num_anchors)
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.loss_delta = 0.1
        self.loss_weight = 50.0

    def compute_loss(
        self,
        targets: list[dict[str, torch.Tensor]],
        bbox_regression: torch.Tensor,
        anchors: list[torch.Tensor],
        matched_idxs: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = bbox_regression.float().sum() * 0.0
        num_foreground = torch.zeros((), dtype=torch.float32, device=bbox_regression.device)
        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # Determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground += foreground_idxs_per_image.numel()

            # Select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # Compute the loss
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            loss += F.huber_loss(
                bbox_regression_per_image.float(),
                target_regression.float(),
                delta=self.loss_delta,
                reduction="sum",
            )

        return self.loss_weight * loss / (4.0 * (num_foreground + 1.0))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        all_bbox_regression = []

        for features in x:
            bbox_regression: torch.Tensor = self.conv_repeat(features)
            bbox_regression = self.predict(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
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
        score_thresh = 0.001
        fg_iou_thresh = 0.5
        topk_candidates = 5000
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
        self.anchor_generator = EfficientDetAnchorGenerator(num_levels)
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
        self.proposal_matcher = EfficientDetMatcher(fg_iou_thresh)
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

    @torch.jit.unused  # type: ignore[untyped-decorator]
    @torch.compiler.disable()  # type: ignore[untyped-decorator]
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

    def postprocess_detections(
        self,
        class_logits: list[torch.Tensor],
        box_regression: list[torch.Tensor],
        anchors: list[list[torch.Tensor]],
        image_sizes: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        num_images = image_sizes.size(0)

        detections: list[dict[str, torch.Tensor]] = []
        for index in range(num_images):
            box_regression_per_image = torch.concat([br[index] for br in box_regression], dim=0)
            logits_per_image = torch.concat([cl[index] for cl in class_logits], dim=0)
            anchors_per_image = torch.concat(anchors[index], dim=0)
            image_shape = image_sizes[index]

            # Remove low scoring boxes
            num_classes = logits_per_image.shape[-1]
            image_scores = torch.sigmoid(logits_per_image).flatten()
            keep_idxs = image_scores > self.score_thresh
            image_scores = image_scores[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # Keep only the global top-k scoring predictions
            num_topk = min(self.topk_candidates, topk_idxs.size(0))
            image_scores, idxs = image_scores.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            image_labels = topk_idxs % num_classes
            image_labels += 1  # Background offset

            image_boxes = self.box_coder.decode_single(
                box_regression_per_image[anchor_idxs], anchors_per_image[anchor_idxs]
            )
            image_boxes = clip_boxes_to_image(image_boxes, image_shape)

            if self.export_mode is False:
                # Non-maximum suppression
                if self.soft_nms is not None:
                    soft_scores, keep = self.soft_nms(image_boxes, image_scores, image_labels, score_threshold=0.001)
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
            else:
                detections.append(
                    {
                        "boxes": image_boxes,
                        "scores": image_scores,
                        "labels": image_labels,
                    }
                )

        return detections

    def forward_net(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        features: dict[str, torch.Tensor] = self.backbone.detection_features(x)
        feature_list = list(features.values())
        feature_list = self.bifpn(feature_list)
        cls_logits = self.class_net(feature_list)
        box_output = self.box_net(feature_list)

        return (cls_logits, box_output, feature_list)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[list[tuple[int, int]]] = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        self._input_check(targets)
        images = self._to_img_list(x, image_sizes)

        cls_logits, box_output, feature_list = self.forward_net(x)
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

            HWA = cls_logits.size(1)  # pylint: disable=invalid-name
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
