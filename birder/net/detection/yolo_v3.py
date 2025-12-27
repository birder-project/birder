"""
YOLO v3, adapted from
https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch

Paper "YOLOv3: An Incremental Improvement", https://arxiv.org/abs/1804.02767
"""

# Reference license: BSD 3-Clause

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
from birder.net.detection.base import ImageList

# Default anchors from YOLO v3 paper (sorted by area, small to large)
# These values are in absolute pixels (width, height) computed using K-Means
# on the COCO dataset with a reference input size of 416x416.
DEFAULT_ANCHORS = [
    [(10, 13), (16, 30), (33, 23)],  # Small objects (stride 8)
    [(30, 61), (62, 45), (59, 119)],  # Medium objects (stride 16)
    [(116, 90), (156, 198), (373, 326)],  # Large objects (stride 32)
]


def decode_predictions(
    predictions: torch.Tensor,
    anchors: torch.Tensor,
    grid: torch.Tensor,
    strides: torch.Tensor,
    num_classes: int,
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

    # Decode center coordinates
    pred_xy = torch.sigmoid(predictions[..., :2])
    pred_wh = predictions[..., 2:4]
    pred_conf = predictions[..., 4:5]
    pred_cls = predictions[..., 5:]

    # Add grid offset and scale
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


class YOLOAnchorGenerator(nn.Module):
    def __init__(self, anchors: list[list[tuple[int, int]]]) -> None:
        super().__init__()
        self.anchors = anchors
        self.num_scales = len(anchors)

    def num_anchors_per_location(self) -> list[int]:
        return [len(a) for a in self.anchors]

    def forward(
        self, image_list: ImageList, feature_maps: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Generate anchor boxes for each feature map.

        Returns scaled anchors, grids, and strides for each feature map.
        """

        device = feature_maps[0].device
        dtype = feature_maps[0].dtype
        image_size = image_list.tensors.shape[-2:]

        all_anchors: list[torch.Tensor] = []
        all_grids: list[torch.Tensor] = []
        all_strides: list[torch.Tensor] = []
        for idx, feature_map in enumerate(feature_maps):
            (_, _, H, W) = feature_map.size()
            stride_h = image_size[0] / H
            stride_w = image_size[1] / W

            # Create grid
            (grid_y, grid_x) = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=-1)

            # Select anchors for this scale
            anchors_for_scale = torch.tensor(self.anchors[idx], device=device, dtype=dtype)

            # Store strides as tensor
            strides = torch.tensor([stride_h, stride_w], device=device, dtype=dtype)

            all_anchors.append(anchors_for_scale)
            all_grids.append(grid)
            all_strides.append(strides)

        return (all_anchors, all_grids, all_strides)


class YOLOHead(nn.Module):
    def __init__(self, in_channels: list[int], num_anchors: list[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        for in_ch, n_anchors in zip(in_channels, num_anchors):
            out_ch = n_anchors * (5 + num_classes)  # 4 bbox + 1 objectness
            self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

        # Initialize weights
        for layer in self.conv_layers:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def _get_conv_for_idx(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.conv_layers[idx](x),
        but TorchScript doesn't support this yet
        """

        out = x
        for i, conv in enumerate(self.conv_layers):
            if i == idx:
                out = conv(x)

        return out

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, feature in enumerate(features):
            outputs.append(self._get_conv_for_idx(feature, i))

        return outputs


class DetectionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels

        self.conv1 = Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv2 = Conv2dNormActivation(
            mid_channels,
            mid_channels * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv3 = Conv2dNormActivation(
            mid_channels * 2,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv4 = Conv2dNormActivation(
            mid_channels,
            mid_channels * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv5 = Conv2dNormActivation(
            mid_channels * 2,
            mid_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.conv6 = Conv2dNormActivation(
            mid_channels,
            mid_channels * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        branch = self.conv5(x)  # Branch for upsampling
        out = self.conv6(branch)

        return (out, branch)


class YOLONeck(nn.Module):
    def __init__(self, in_channels: list[int]) -> None:
        super().__init__()
        self.det_block3 = DetectionBlock(in_channels[2], in_channels[2] // 2)  # Large objects

        self.upsample1 = nn.Sequential(
            Conv2dNormActivation(
                in_channels[2] // 2,
                in_channels[1] // 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.det_block2 = DetectionBlock(in_channels[1] + in_channels[1] // 2, in_channels[1] // 2)  # Medium objects

        self.upsample2 = nn.Sequential(
            Conv2dNormActivation(
                in_channels[1] // 2,
                in_channels[0] // 2,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
            ),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.det_block1 = DetectionBlock(in_channels[0] + in_channels[0] // 2, in_channels[0] // 2)  # Small objects

        self.out_channels = [in_channels[0], in_channels[1], in_channels[2]]

    def forward(self, features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        feature_list = list(features.values())
        (c3, c4, c5) = feature_list[-3:]

        # Large objects
        (out3, branch3) = self.det_block3(c5)

        # Medium objects
        up3 = self.upsample1(branch3)
        c4_concat = torch.concat([up3, c4], dim=1)
        (out2, branch2) = self.det_block2(c4_concat)

        # Small objects
        up2 = self.upsample2(branch2)
        c3_concat = torch.concat([up2, c3], dim=1)
        (out1, _) = self.det_block1(c3_concat)

        return [out1, out2, out3]


# pylint: disable=invalid-name
class YOLO_v3(DetectionBaseNet):
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

        self.num_classes = self.num_classes - 1

        score_thresh = 0.05
        nms_thresh = 0.45
        detections_per_img = 300
        self.ignore_thresh = 0.5
        self.noobj_coeff = 0.2
        self.coord_coeff = 5.0
        self.obj_coeff = 1.0
        self.cls_coeff = 1.0

        self.anchors = DEFAULT_ANCHORS
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.backbone.return_channels = self.backbone.return_channels[-3:]
        self.backbone.return_stages = self.backbone.return_stages[-3:]

        self.neck = YOLONeck(self.backbone.return_channels)

        self.anchor_generator = YOLOAnchorGenerator(self.anchors)
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = YOLOHead(self.neck.out_channels, num_anchors, self.num_classes)

        if self.export_mode is False:
            self.forward = torch.compiler.disable(recursive=False)(self.forward)  # type: ignore[method-assign]

    def reset_classifier(self, num_classes: int) -> None:
        self.num_classes = num_classes
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
            Box dimensions (num_boxes, 2)
        anchor_wh
            Anchor dimensions (num_anchors, 2)

        Returns
        -------
        IoU matrix (num_boxes, num_anchors)
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

    @torch.jit.unused  # type: ignore[untyped-decorator]
    @torch.compiler.disable()  # type: ignore[untyped-decorator]
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

            pred = pred.view(N, num_anchors_scale, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            target = target_tensors[scale_idx]
            obj_mask = obj_masks[scale_idx]
            noobj_mask = noobj_masks[scale_idx]

            if obj_mask.any():
                # XY loss: BCE with logits (targets are in [0,1], predictions are logits)
                pred_xy = pred[..., :2]
                target_xy = target[..., :2]
                xy_loss = F.binary_cross_entropy_with_logits(pred_xy[obj_mask], target_xy[obj_mask], reduction="sum")

                # WH loss: MSE on raw values (both are in log-space)
                pred_wh = pred[..., 2:4]
                target_wh = target[..., 2:4]
                wh_loss = F.mse_loss(pred_wh[obj_mask], target_wh[obj_mask], reduction="sum")

                coord_loss = coord_loss + xy_loss + wh_loss

                # Object confidence loss
                obj_loss = obj_loss + F.binary_cross_entropy_with_logits(
                    pred[..., 4][obj_mask], target[..., 4][obj_mask], reduction="sum"
                )

                # Classification loss
                cls_loss = cls_loss + F.binary_cross_entropy_with_logits(
                    pred[..., 5:][obj_mask], target[..., 5:][obj_mask], reduction="sum"
                )

                num_obj += obj_mask.sum().item()

            # No-object loss: BCE with logits against zeros simplifies to softplus
            noobj_loss = noobj_loss + F.softplus(pred[..., 4][noobj_mask]).sum()  # pylint: disable=not-callable

        # Normalize by number of positive samples (average per object)
        num_obj = max(1, num_obj)
        coord_loss = self.coord_coeff * coord_loss / num_obj
        obj_loss = self.obj_coeff * obj_loss / num_obj
        cls_loss = self.cls_coeff * cls_loss / num_obj
        noobj_loss = self.noobj_coeff * noobj_loss / num_obj

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
                )
                all_decoded.append(decoded)

            decoded_predictions = torch.concat(all_decoded, dim=1)
            detections = self.postprocess_detections(decoded_predictions, images.image_sizes)

        return (detections, losses)
