"""
YOLO v4, adapted from
https://github.com/Tianxiaomo/pytorch-YOLOv4

Paper "YOLOv4: Optimal Speed and Accuracy of Object Detection", https://arxiv.org/abs/2004.10934
"""

# Reference license: Apache-2.0

from functools import partial
from typing import Any
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import Conv2dNormActivation
from torchvision.ops import boxes as box_ops

from birder.net.base import DetectorBackbone
from birder.net.detection.base import DetectionBaseNet
from birder.net.detection.yolo_v3 import YOLOAnchorGenerator
from birder.net.detection.yolo_v3 import YOLOHead

# Default anchors from YOLO v4 (COCO)
DEFAULT_ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],  # Small
    [(36, 75), (76, 55), (72, 146)],  # Medium
    [(142, 110), (192, 243), (459, 401)],  # Large
]

# Scale factors per detection scale to eliminate grid sensitivity
DEFAULT_SCALE_XY = [1.2, 1.1, 1.05]  # [small, medium, large]


def decode_predictions(
    predictions: torch.Tensor,
    anchors: torch.Tensor,
    grid: torch.Tensor,
    strides: torch.Tensor,
    num_classes: int,
    scale_xy: float,
) -> torch.Tensor:
    """
    Decode YOLO predictions to bounding boxes.

    Parameters
    ----------
    predictions
        Raw predictions from network (N, num_anchors * (5 + num_classes), H, W).
    anchors
        Anchor boxes for this scale (num_anchors, 2).
    grid
        Grid coordinates (H, W, 2).
    strides
        Strides tensor [stride_h, stride_w].
    num_classes
        Number of classes.
    scale_xy
        Scale factor for grid sensitivity elimination.
        YOLOv4 uses 1.05-1.2 depending on scale level.

    Returns
    -------
    Decoded predictions (N, H*W*num_anchors, 5 + num_classes).
    """

    (N, _, H, W) = predictions.size()
    num_anchors = anchors.shape[0]
    stride_h = strides[0]
    stride_w = strides[1]

    predictions = predictions.view(N, num_anchors, 5 + num_classes, H, W)
    predictions = predictions.permute(0, 3, 4, 1, 2).contiguous()

    # Decode center coordinates with scale factor
    pred_xy_raw = torch.sigmoid(predictions[..., :2])
    pred_xy = pred_xy_raw * scale_xy - (scale_xy - 1) / 2

    pred_wh = predictions[..., 2:4]
    pred_conf = predictions[..., 4:5]
    pred_cls = predictions[..., 5:]

    # Add grid offset and scale to pixel coordinates
    grid_expanded = grid.unsqueeze(2)  # (H, W, 1, 2)
    stride_tensor = torch.stack([stride_w, stride_h])
    pred_xy = (pred_xy + grid_expanded) * stride_tensor

    # Scale width/height by anchors
    anchors_expanded = anchors.view(1, 1, num_anchors, 2)
    pred_wh = torch.exp(pred_wh) * anchors_expanded

    # Convert to x1, y1, x2, y2 format
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    pred_boxes = torch.concat([pred_x1y1, pred_x2y2], dim=-1)

    # Apply sigmoid to confidence and class predictions
    pred_conf = torch.sigmoid(pred_conf)
    pred_cls = torch.sigmoid(pred_cls)

    # Reshape to (N, H*W*num_anchors, ...)
    pred_boxes = pred_boxes.view(N, -1, 4)
    pred_conf = pred_conf.view(N, -1, 1)
    pred_cls = pred_cls.view(N, -1, num_classes)

    return torch.concat([pred_boxes, pred_conf, pred_cls], dim=-1)


# pylint: disable=too-many-locals
def compute_ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute Complete IoU (CIoU) loss.

    CIoU = IoU - (distance**2 / c**2) - alpha * v
    where v measures aspect ratio consistency and c is diagonal of enclosing box.

    Parameters
    ----------
    pred_boxes
        Predicted boxes in [x1, y1, x2, y2] format.
    target_boxes
        Target boxes in [x1, y1, x2, y2] format.
    eps
        Small epsilon for numerical stability.

    Returns
    -------
    CIoU loss (1 - CIoU) for each box pair, shape (N,).
    """

    # Extract coordinates
    (pred_x1, pred_y1, pred_x2, pred_y2) = pred_boxes.unbind(dim=-1)
    (target_x1, target_y1, target_x2, target_y2) = target_boxes.unbind(dim=-1)

    # Calculate box dimensions (clamped for stability)
    pred_w = torch.clamp(pred_x2 - pred_x1, min=eps)
    pred_h = torch.clamp(pred_y2 - pred_y1, min=eps)
    target_w = torch.clamp(target_x2 - target_x1, min=eps)
    target_h = torch.clamp(target_y2 - target_y1, min=eps)

    # Calculate areas
    pred_area = pred_w * pred_h
    target_area = target_w * target_h

    # Calculate intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # Calculate IoU
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + eps)

    # Calculate center coordinates
    pred_cx = (pred_x1 + pred_x2) / 2
    pred_cy = (pred_y1 + pred_y2) / 2
    target_cx = (target_x1 + target_x2) / 2
    target_cy = (target_y1 + target_y2) / 2

    # Calculate squared distance between centers
    center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

    # Calculate smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    # Calculate squared diagonal of enclosing box
    enclose_w = torch.clamp(enclose_x2 - enclose_x1, min=eps)
    enclose_h = torch.clamp(enclose_y2 - enclose_y1, min=eps)
    enclose_diag_sq = enclose_w**2 + enclose_h**2

    # Calculate aspect ratio consistency term
    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # CIoU = IoU - distance_penalty - aspect_ratio_penalty
    ciou = iou - (center_dist_sq / (enclose_diag_sq + eps)) - alpha * v

    return 1 - ciou


class SPPBlock(nn.Module):
    """
    Spatial Pyramid Pooling block.
    Concatenates max-pooled features at multiple scales.
    """

    def __init__(self, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=(k, k), stride=(1, 1), padding=(k // 2, k // 2)) for k in kernel_sizes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x] + [pool(x) for pool in self.pools]
        return torch.concat(features, dim=1)


class YOLONeckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int) -> None:
        super().__init__()
        assert num_layers % 2 == 1

        layers = []
        prev_channels = in_channels
        for i in range(num_layers):
            if i % 2 == 0:
                layers.append(
                    Conv2dNormActivation(
                        prev_channels,
                        out_channels,
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        padding=(0, 0),
                        activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
                    )
                )
                prev_channels = out_channels
            else:
                layers.append(
                    Conv2dNormActivation(
                        prev_channels,
                        out_channels * 2,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                        activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
                    )
                )
                prev_channels = out_channels * 2

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YOLONeck(nn.Module):
    """
    Path Aggregation Network (PAN) neck.
    Combines FPN (top-down) with bottom-up path augmentation.
    """

    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        (c3, c4, c5) = in_channels

        # Top-down pathway (FPN-like) with SPP
        self.p5_pre_spp = YOLONeckBlock(c5, c5 // 2, num_layers=3)
        self.spp = SPPBlock(kernel_sizes=(5, 9, 13))
        self.p5_post_spp = YOLONeckBlock(c5 * 2, c5 // 2, num_layers=3)

        self.p5_to_p4 = Conv2dNormActivation(
            c5 // 2,
            c4 // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.upsample_p5 = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_reduce = Conv2dNormActivation(
            c4,
            c4 // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.p4_block = YOLONeckBlock(c4, c4 // 2, num_layers=5)
        self.p4_to_p3 = Conv2dNormActivation(
            c4 // 2,
            c3 // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.upsample_p4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.p3_reduce = Conv2dNormActivation(
            c3,
            c3 // 2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.p3_block = YOLONeckBlock(c3, c3 // 2, num_layers=5)

        # Bottom-up pathway (PAN augmentation)
        self.p3_out = Conv2dNormActivation(
            c3 // 2,
            c3,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.downsample_p3 = Conv2dNormActivation(
            c3 // 2,
            c4 // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.n4_block = YOLONeckBlock(c4, c4 // 2, num_layers=5)
        self.n4_out = Conv2dNormActivation(
            c4 // 2,
            c4,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.downsample_n4 = Conv2dNormActivation(
            c4 // 2,
            c5 // 2,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.n5_block = YOLONeckBlock(c5, c5 // 2, num_layers=5)
        self.n5_out = Conv2dNormActivation(
            c5 // 2,
            c5,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )

        self.out_channels = [c3, c4, c5]

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        feature_list = list(features.values())
        (c3, c4, c5) = feature_list[-3:]

        # Top-down pathway with SPP
        p5 = self.p5_pre_spp(c5)
        p5 = self.spp(p5)
        p5 = self.p5_post_spp(p5)

        p5_up = self.upsample_p5(self.p5_to_p4(p5))
        p4_cat = torch.concat([self.p4_reduce(c4), p5_up], dim=1)
        p4 = self.p4_block(p4_cat)

        p4_up = self.upsample_p4(self.p4_to_p3(p4))
        p3_cat = torch.concat([self.p3_reduce(c3), p4_up], dim=1)
        p3 = self.p3_block(p3_cat)

        # Bottom-up pathway (PAN)
        n3_out = self.p3_out(p3)

        n4_cat = torch.concat([self.downsample_p3(p3), p4], dim=1)
        n4 = self.n4_block(n4_cat)
        n4_out = self.n4_out(n4)

        n5_cat = torch.concat([self.downsample_n4(n4), p5], dim=1)
        n5 = self.n5_block(n5_cat)
        n5_out = self.n5_out(n5)

        return [n3_out, n4_out, n5_out]


# pylint: disable=invalid-name
class YOLO_v4(DetectionBaseNet):
    default_size = (608, 608)
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

        self.num_classes = self.num_classes - 1

        score_thresh = 0.05
        nms_thresh = 0.45
        detections_per_img = 300
        self.ignore_thresh = 0.7

        # Loss coefficients
        # Note: coord_coeff=0.07 matches darknet's iou_normalizer for CIoU loss.
        # However, darknet uses squared deltas (loss = sum(delta^2) / batch) while we compute
        # CIoU loss directly (loss = coeff * sum(ciou) / num_obj). This different formulation
        # means darknet's obj_normalizer=1.0 overweights background loss relative to box
        # regression in our implementation. We use a lower noobj_coeff (vs darknet's 1.0) to
        # restore a better balance, similar to YOLOv3's noobj_coeff=0.2.
        self.noobj_coeff = 0.3
        self.coord_coeff = 0.07
        self.obj_coeff = 1.0
        self.cls_coeff = 1.0

        self.anchors = DEFAULT_ANCHORS
        self.scale_xy = DEFAULT_SCALE_XY
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]

        self.label_smoothing = 0.1
        self.smooth_positive = 1.0 - self.label_smoothing
        self.smooth_negative = self.label_smoothing / self.num_classes

        self.neck = YOLONeck(self.backbone.return_channels)

        self.anchor_generator = YOLOAnchorGenerator(self.anchors)
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = YOLOHead(self.neck.out_channels, num_anchors, self.num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.smooth_negative = self.label_smoothing / self.num_classes
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = YOLOHead(self.neck.out_channels, num_anchors, self.num_classes)

    def freeze(self, freeze_classifier: bool = True) -> None:
        for param in self.parameters():
            param.requires_grad = False

        if freeze_classifier is False:
            for param in self.head.parameters():
                param.requires_grad = True

    def _compute_anchor_iou(self, box_wh: torch.Tensor, anchor_wh: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between boxes and anchors using only width/height (centered at origin).

        Parameters
        ----------
        box_wh
            Box dimensions (num_boxes, 2).
        anchor_wh
            Anchor dimensions (num_anchors, 2).

        Returns
        -------
        IoU matrix (num_boxes, num_anchors).
        """

        inter_w = torch.min(box_wh[:, None, 0], anchor_wh[None, :, 0])
        inter_h = torch.min(box_wh[:, None, 1], anchor_wh[None, :, 1])
        inter_area = inter_w * inter_h
        box_area = box_wh[:, 0] * box_wh[:, 1]
        anchor_area = anchor_wh[:, 0] * anchor_wh[:, 1]
        iou = inter_area / (box_area[:, None] + anchor_area[None, :] - inter_area + 1e-7)

        return iou

    def _build_targets(  # pylint: disable=too-many-locals
        self,
        predictions: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        anchors: list[torch.Tensor],
        strides: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Build targets for YOLO loss computation.

        Uses global anchor assignment: each ground truth box is assigned to the single
        best-matching anchor across all scales, following the original YOLO v3 paper.

        Returns target tensors, object masks, and no-object masks for each scale.
        """

        device = predictions[0].device
        dtype = predictions[0].dtype
        batch_size = predictions[0].shape[0]
        num_scales = len(predictions)

        # Build flat list of all anchors with their scale indices
        all_anchors = torch.concat(anchors, dim=0)
        anchors_per_scale = [len(self.anchors[i]) for i in range(num_scales)]
        cumsum_anchors = torch.tensor([0] + anchors_per_scale, device=device).cumsum(0)

        # Get grid sizes and strides for each scale
        grid_sizes: list[tuple[int, int]] = []
        stride_tensors: list[torch.Tensor] = []
        for scale_idx in range(num_scales):
            (_, _, H, W) = predictions[scale_idx].size()
            grid_sizes.append((H, W))
            stride_tensors.append(strides[scale_idx])

        # Initialize targets and masks for each scale
        target_tensors: list[torch.Tensor] = []
        obj_masks: list[torch.Tensor] = []
        noobj_masks: list[torch.Tensor] = []
        for scale_idx in range(num_scales):
            (H, W) = grid_sizes[scale_idx]
            num_anchors_scale = anchors_per_scale[scale_idx]
            target_tensors.append(
                torch.zeros((batch_size, num_anchors_scale, H, W, 5 + self.num_classes), device=device, dtype=dtype)
            )
            obj_masks.append(torch.zeros((batch_size, num_anchors_scale, H, W), device=device, dtype=torch.bool))
            noobj_masks.append(torch.ones((batch_size, num_anchors_scale, H, W), device=device, dtype=torch.bool))

        # Process each image in the batch
        for batch_idx, target_per_image in enumerate(targets):
            boxes = target_per_image["boxes"]
            labels = target_per_image["labels"] - 1  # Remove background offset
            num_boxes = boxes.shape[0]

            if num_boxes == 0:
                continue

            # Compute box dimensions and centers
            box_wh = boxes[:, 2:] - boxes[:, :2]
            box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2

            # Compute IoU with ALL anchors globally and find best anchor per box
            iou_with_all_anchors = self._compute_anchor_iou(box_wh, all_anchors)
            best_anchor_global = iou_with_all_anchors.argmax(dim=1)

            # Determine scale index for each box
            scale_indices = torch.bucketize(best_anchor_global, cumsum_anchors[1:-1], right=True)

            # Local anchor index within each scale
            local_anchor_indices = best_anchor_global - cumsum_anchors[scale_indices]

            # Assign targets for each scale
            for scale_idx in range(num_scales):
                mask = scale_indices == scale_idx
                if not mask.any():
                    continue

                (H, W) = grid_sizes[scale_idx]
                stride_h, stride_w = stride_tensors[scale_idx][0], stride_tensors[scale_idx][1]
                anchors_scale = anchors[scale_idx]

                # Get boxes assigned to this scale
                scale_boxes = boxes[mask]
                scale_box_wh = box_wh[mask]
                scale_box_centers = box_centers[mask]
                scale_labels = labels[mask]
                scale_local_anchors = local_anchor_indices[mask]

                # Compute grid coordinates
                grid_x = (scale_box_centers[:, 0] / stride_w).clamp(0, W - 1)
                grid_y = (scale_box_centers[:, 1] / stride_h).clamp(0, H - 1)
                gi = grid_x.long()
                gj = grid_y.long()

                # Target offsets (relative to grid cell)
                tx = (grid_x - gi.float()).to(dtype)
                ty = (grid_y - gj.float()).to(dtype)

                # Target width/height (log-space relative to anchor)
                anchor_wh = anchors_scale[scale_local_anchors]
                tw = torch.log(scale_box_wh[:, 0] / anchor_wh[:, 0] + 1e-7).to(dtype)
                th = torch.log(scale_box_wh[:, 1] / anchor_wh[:, 1] + 1e-7).to(dtype)

                # Build indices for scatter
                num_scale_boxes = scale_boxes.shape[0]
                batch_indices = torch.full((num_scale_boxes,), batch_idx, device=device, dtype=torch.long)

                # Assign targets using advanced indexing
                target_tensors[scale_idx][batch_indices, scale_local_anchors, gj, gi, 0] = tx
                target_tensors[scale_idx][batch_indices, scale_local_anchors, gj, gi, 1] = ty
                target_tensors[scale_idx][batch_indices, scale_local_anchors, gj, gi, 2] = tw
                target_tensors[scale_idx][batch_indices, scale_local_anchors, gj, gi, 3] = th
                target_tensors[scale_idx][batch_indices, scale_local_anchors, gj, gi, 4] = 1.0

                # Class assignment
                class_indices = 5 + scale_labels.long()
                for i in range(num_scale_boxes):
                    target_tensors[scale_idx][
                        batch_indices[i], scale_local_anchors[i], gj[i], gi[i], class_indices[i]
                    ] = 1.0

                obj_masks[scale_idx][batch_indices, scale_local_anchors, gj, gi] = True
                noobj_masks[scale_idx][batch_indices, scale_local_anchors, gj, gi] = False

            # Compute ignore mask: anchors with IoU > ignore_thresh should not contribute to noobj_loss
            for scale_idx in range(num_scales):
                (H, W) = grid_sizes[scale_idx]
                stride_h, stride_w = stride_tensors[scale_idx][0], stride_tensors[scale_idx][1]
                anchors_scale = anchors[scale_idx]
                num_anchors_scale = anchors_per_scale[scale_idx]

                # Create grid of anchor box centers
                (grid_y, grid_x) = torch.meshgrid(
                    torch.arange(H, device=device, dtype=dtype),
                    torch.arange(W, device=device, dtype=dtype),
                    indexing="ij",
                )
                centers_x = (grid_x + 0.5) * stride_w  # (H, W)
                centers_y = (grid_y + 0.5) * stride_h  # (H, W)

                # Build all anchor boxes
                anchor_w = anchors_scale[:, 0].view(-1, 1, 1)
                anchor_h = anchors_scale[:, 1].view(-1, 1, 1)
                pred_boxes = torch.stack(
                    [
                        centers_x.unsqueeze(0) - anchor_w / 2,
                        centers_y.unsqueeze(0) - anchor_h / 2,
                        centers_x.unsqueeze(0) + anchor_w / 2,
                        centers_y.unsqueeze(0) + anchor_h / 2,
                    ],
                    dim=-1,
                )

                # Compute IoU for all anchors
                iou = box_ops.box_iou(pred_boxes.view(-1, 4), boxes)
                max_iou = iou.max(dim=1)[0].view(num_anchors_scale, H, W)

                # Mark as ignore where IoU > threshold (and not already an object)
                ignore_mask = (max_iou > self.ignore_thresh) & ~obj_masks[scale_idx][batch_idx]
                noobj_masks[scale_idx][batch_idx] = noobj_masks[scale_idx][batch_idx] & ~ignore_mask

        return (target_tensors, obj_masks, noobj_masks)

    def compute_loss(  # pylint: disable=too-many-locals
        self,
        predictions: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        anchors: list[torch.Tensor],
        strides: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        (target_tensors, obj_masks, noobj_masks) = self._build_targets(predictions, targets, anchors, strides)

        device = predictions[0].device
        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        num_obj = 0
        for scale_idx, pred in enumerate(predictions):
            (N, _, H, W) = pred.size()
            num_anchors_scale = len(self.anchors[scale_idx])
            stride_h = strides[scale_idx][0]
            stride_w = strides[scale_idx][1]
            scale_xy = self.scale_xy[scale_idx]

            pred = pred.view(N, num_anchors_scale, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            target = target_tensors[scale_idx]
            obj_mask = obj_masks[scale_idx]
            noobj_mask = noobj_masks[scale_idx]

            if obj_mask.any():
                indices = torch.nonzero(obj_mask)

                # Get raw predictions for positive samples
                pred_obj = pred[obj_mask]

                # Get grid coordinates
                grid_y = indices[:, 2].float()
                grid_x = indices[:, 3].float()

                # Decode predictions to pixel coordinates with scale factor
                pred_xy_scaled = torch.sigmoid(pred_obj[:, :2]) * scale_xy - (scale_xy - 1) / 2
                p_x = (pred_xy_scaled[:, 0] + grid_x) * stride_w
                p_y = (pred_xy_scaled[:, 1] + grid_y) * stride_h

                # Width/Height decoding (anchors are in pixel units)
                anchors_scale = anchors[scale_idx].to(device)
                anchor_w = anchors_scale[indices[:, 1], 0]
                anchor_h = anchors_scale[indices[:, 1], 1]
                p_w = torch.exp(pred_obj[:, 2]) * anchor_w
                p_h = torch.exp(pred_obj[:, 3]) * anchor_h

                # Decode targets to pixel coordinates
                t_tx = target[obj_mask][:, 0]
                t_ty = target[obj_mask][:, 1]
                t_tw = target[obj_mask][:, 2]
                t_th = target[obj_mask][:, 3]

                t_x = (t_tx + grid_x) * stride_w
                t_y = (t_ty + grid_y) * stride_h
                t_w = torch.exp(t_tw) * anchor_w
                t_h = torch.exp(t_th) * anchor_h

                # Convert center-wh to x1y1x2y2
                pred_boxes = torch.stack([p_x - p_w / 2, p_y - p_h / 2, p_x + p_w / 2, p_y + p_h / 2], dim=1)
                target_boxes = torch.stack([t_x - t_w / 2, t_y - t_h / 2, t_x + t_w / 2, t_y + t_h / 2], dim=1)

                # Calculate CIoU loss
                ciou = compute_ciou_loss(pred_boxes, target_boxes)
                coord_loss = coord_loss + ciou.sum()

                # Objectness loss
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    pred[..., 4][obj_mask], target[..., 4][obj_mask], reduction="sum"
                )

                # Classification loss
                cls_targets = target[..., 5:][obj_mask]
                cls_targets_smooth = cls_targets * (self.smooth_positive - self.smooth_negative) + self.smooth_negative
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    pred[..., 5:][obj_mask], cls_targets_smooth, reduction="sum"
                )

                num_obj += obj_mask.sum().item()

            # No-object loss: BCE with logits against zeros simplifies to softplus
            noobj_loss = noobj_loss + F.softplus(pred[..., 4][noobj_mask]).sum()  # pylint: disable=not-callable

        # Normalize by number of positive samples (average per object)
        num_obj = max(1, num_obj)
        coord_loss = self.coord_coeff * coord_loss / num_obj
        obj_loss = self.obj_coeff * obj_loss / num_obj
        cls_loss = self.cls_coeff * cls_loss / num_obj

        # Normalize background loss by batch size (average per image)
        batch_size = predictions[0].shape[0]
        noobj_loss = self.noobj_coeff * noobj_loss / batch_size

        return {
            "bbox_regression": coord_loss,
            "objectness": obj_loss + noobj_loss,
            "classification": cls_loss,
        }

    def postprocess_detections(
        self,
        decoded_predictions: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        batch_size = decoded_predictions.shape[0]
        detections: list[dict[str, torch.Tensor]] = []

        for idx in range(batch_size):
            pred = decoded_predictions[idx]  # (num_predictions, 5 + num_classes)
            boxes = pred[:, :4]
            objectness = pred[:, 4]
            class_scores = pred[:, 5:]

            # Combine objectness and class scores
            scores = objectness.unsqueeze(-1) * class_scores
            (scores, labels) = scores.max(dim=-1)
            labels = labels + 1  # Add background offset

            # Filter by score threshold
            keep = scores > self.score_thresh
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # Clip boxes to image
            image_shape = image_shapes[idx]
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # Remove small boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1.0)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # NMS
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores[keep],
                    "labels": labels[keep],
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
        neck_features = self.neck(features)
        predictions = self.head(neck_features)
        (anchors, grids, strides) = self.anchor_generator(images, neck_features)

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"
            losses = self.compute_loss(predictions, targets, anchors, strides)

        else:
            all_decoded: list[torch.Tensor] = []
            for scale_idx, pred in enumerate(predictions):
                decoded = decode_predictions(
                    pred,
                    anchors[scale_idx],
                    grids[scale_idx],
                    strides[scale_idx],
                    self.num_classes,
                    self.scale_xy[scale_idx],
                )
                all_decoded.append(decoded)

            decoded_predictions = torch.concat(all_decoded, dim=1)
            detections = self.postprocess_detections(decoded_predictions, images.image_sizes)

        return (detections, losses)
