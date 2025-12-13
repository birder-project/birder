"""
YOLO v2, adapted from
https://github.com/longcw/yolo2-pytorch

Paper "YOLO9000: Better, Faster, Stronger", https://arxiv.org/abs/1612.08242
"""

# Reference license: MIT

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

# Default anchors from yolo.cfg (COCO dataset)
DEFAULT_ANCHORS = [
    (0.57273, 0.677385),
    (1.87446, 2.06253),
    (3.33843, 5.47434),
    (7.88282, 3.52778),
    (9.77052, 9.16828),
]


def decode_predictions(
    predictions: torch.Tensor,
    anchors: torch.Tensor,
    grid: torch.Tensor,
    stride: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """
    Decode YOLO predictions to bounding boxes.

    Parameters
    ----------
    predictions
        Raw predictions from network (N, num_anchors * (5 + num_classes), H, W).
    anchors
        Anchor boxes (num_anchors, 2).
    grid
        Grid coordinates (H, W, 2).
    stride
        Stride tensor [stride_h, stride_w].
    num_classes
        Number of classes.

    Returns
    -------
    Decoded predictions (N, H*W*num_anchors, 5 + num_classes).
    """

    (N, _, H, W) = predictions.size()
    num_anchors = anchors.shape[0]
    stride_h = stride[0]
    stride_w = stride[1]

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

    # Apply sigmoid to confidence and softmax to class predictions
    pred_conf = torch.sigmoid(pred_conf)
    pred_cls = F.softmax(pred_cls, dim=-1)

    # Reshape to (N, H*W*num_anchors, ...)
    pred_boxes = pred_boxes.view(N, -1, 4)
    pred_conf = pred_conf.view(N, -1, 1)
    pred_cls = pred_cls.view(N, -1, num_classes)

    return torch.concat([pred_boxes, pred_conf, pred_cls], dim=-1)


class YOLOAnchorGenerator(nn.Module):
    def __init__(self, anchors: list[tuple[float, float]]) -> None:
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)

    def num_anchors_per_location(self) -> int:
        return self.num_anchors

    def forward(
        self, image_list: ImageList, feature_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate anchor boxes for the feature map.

        Returns scaled anchors, grid, and stride.
        """

        device = feature_map.device
        dtype = feature_map.dtype
        image_size = image_list.tensors.shape[-2:]
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

        # Scale anchors to feature map stride (anchors are in grid units)
        anchors_tensor = torch.tensor(self.anchors, device=device, dtype=dtype)
        anchors_tensor = anchors_tensor * torch.tensor([stride_w, stride_h], device=device, dtype=dtype)

        # Store strides as tensor
        stride = torch.tensor([stride_h, stride_w], device=device, dtype=dtype)

        return (anchors_tensor, grid, stride)


class YOLONeck(nn.Module):
    def __init__(self, in_channels: list[int], mid_channels: int) -> None:
        super().__init__()
        assert len(in_channels) >= 2

        self.out_channels = mid_channels
        self.conv_fine = Conv2dNormActivation(
            in_channels[-2],
            64,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )
        self.reorg = nn.PixelUnshuffle(downscale_factor=2)
        self.conv_coarse = nn.Sequential(
            Conv2dNormActivation(
                in_channels[-1],
                mid_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
            ),
            Conv2dNormActivation(
                mid_channels,
                mid_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
            ),
        )

        concat_channels = mid_channels + (64 * 4)
        self.conv_post_concat = Conv2dNormActivation(
            concat_channels,
            mid_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            activation_layer=partial(nn.LeakyReLU, negative_slope=0.1),
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        feature_list = list(features.values())
        fine_grained = feature_list[-2]
        coarse = feature_list[-1]

        x_fine = self.conv_fine(fine_grained)
        x_fine = self.reorg(x_fine)

        x_coarse = self.conv_coarse(coarse)

        x = torch.concat((x_fine, x_coarse), dim=1)
        x = self.conv_post_concat(x)

        return x


class YOLOHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Initialize weights
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# pylint: disable=invalid-name
class YOLO_v2(DetectionBaseNet):
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
        mid_channels = 1024
        self.ignore_thresh = 0.5
        self.noobj_coeff = 0.5
        self.coord_coeff = 5.0
        self.obj_coeff = 1.0
        self.cls_coeff = 1.0

        self.anchors = DEFAULT_ANCHORS
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.backbone.return_channels = self.backbone.return_channels[-2:]
        self.backbone.return_stages = self.backbone.return_stages[-2:]

        self.neck = YOLONeck(self.backbone.return_channels, mid_channels)

        self.anchor_generator = YOLOAnchorGenerator(self.anchors)
        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = YOLOHead(self.neck.out_channels, num_anchors, self.num_classes)

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
        predictions: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build targets for YOLO loss computation.

        Returns target tensor, object mask, and no-object mask.
        """

        device = predictions.device
        dtype = predictions.dtype
        (batch_size, _, H, W) = predictions.size()
        num_anchors = len(self.anchors)

        stride_h = stride[0]
        stride_w = stride[1]

        # Initialize targets and masks
        target_tensor = torch.zeros((batch_size, num_anchors, H, W, 5 + self.num_classes), device=device, dtype=dtype)
        obj_mask = torch.zeros((batch_size, num_anchors, H, W), device=device, dtype=torch.bool)
        noobj_mask = torch.ones((batch_size, num_anchors, H, W), device=device, dtype=torch.bool)

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

            # Find best anchor for each ground truth box
            iou_with_anchors = self._compute_anchor_iou(box_wh, anchors)
            best_anchors = iou_with_anchors.argmax(dim=1)

            # Compute grid coordinates
            grid_x = (box_centers[:, 0] / stride_w).clamp(0, W - 1)
            grid_y = (box_centers[:, 1] / stride_h).clamp(0, H - 1)
            gi = grid_x.long()
            gj = grid_y.long()

            # Target offsets (relative to grid cell)
            tx = (grid_x - gi.float()).to(dtype)
            ty = (grid_y - gj.float()).to(dtype)

            # Target width/height (log-space relative to anchor)
            anchor_wh = anchors[best_anchors]
            tw = torch.log(box_wh[:, 0] / anchor_wh[:, 0] + 1e-7).to(dtype)
            th = torch.log(box_wh[:, 1] / anchor_wh[:, 1] + 1e-7).to(dtype)

            # Assign targets using advanced indexing
            batch_indices = torch.full((num_boxes,), batch_idx, device=device, dtype=torch.long)

            target_tensor[batch_indices, best_anchors, gj, gi, 0] = tx
            target_tensor[batch_indices, best_anchors, gj, gi, 1] = ty
            target_tensor[batch_indices, best_anchors, gj, gi, 2] = tw
            target_tensor[batch_indices, best_anchors, gj, gi, 3] = th
            target_tensor[batch_indices, best_anchors, gj, gi, 4] = 1.0

            # Class assignment (one-hot encoding)
            class_indices = 5 + labels.long()
            for i in range(num_boxes):
                target_tensor[batch_indices[i], best_anchors[i], gj[i], gi[i], class_indices[i]] = 1.0

            obj_mask[batch_indices, best_anchors, gj, gi] = True
            noobj_mask[batch_indices, best_anchors, gj, gi] = False

            # Compute ignore mask: anchors with IoU > ignore_thresh
            (grid_y_all, grid_x_all) = torch.meshgrid(
                torch.arange(H, device=device, dtype=dtype),
                torch.arange(W, device=device, dtype=dtype),
                indexing="ij",
            )
            centers_x = (grid_x_all + 0.5) * stride_w  # (H, W)
            centers_y = (grid_y_all + 0.5) * stride_h  # (H, W)

            # Build all anchor boxes
            anchor_w = anchors[:, 0].view(-1, 1, 1)
            anchor_h = anchors[:, 1].view(-1, 1, 1)
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
            max_iou = iou.max(dim=1)[0].view(num_anchors, H, W)

            # Mark as ignore where IoU > threshold (and not already an object)
            ignore_mask = (max_iou > self.ignore_thresh) & ~obj_mask[batch_idx]
            noobj_mask[batch_idx] = noobj_mask[batch_idx] & ~ignore_mask

        return (target_tensor, obj_mask, noobj_mask)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        anchors: torch.Tensor,
        stride: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        (target_tensor, obj_mask, noobj_mask) = self._build_targets(predictions, targets, anchors, stride)

        device = predictions.device
        (N, _, H, W) = predictions.size()
        num_anchors = len(self.anchors)

        predictions = predictions.view(N, num_anchors, 5 + self.num_classes, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)

        if obj_mask.any():
            # XY loss: BCE with logits
            pred_xy = predictions[..., :2]
            target_xy = target_tensor[..., :2]
            xy_loss = F.binary_cross_entropy_with_logits(pred_xy[obj_mask], target_xy[obj_mask], reduction="sum")

            # WH loss: MSE on raw values
            pred_wh = predictions[..., 2:4]
            target_wh = target_tensor[..., 2:4]
            wh_loss = F.mse_loss(pred_wh[obj_mask], target_wh[obj_mask], reduction="sum")

            coord_loss = xy_loss + wh_loss

            # Object confidence loss
            obj_loss = F.binary_cross_entropy_with_logits(
                predictions[..., 4][obj_mask], target_tensor[..., 4][obj_mask], reduction="sum"
            )

            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(
                predictions[..., 5:][obj_mask], target_tensor[..., 5:][obj_mask], reduction="sum"
            )

            num_obj = obj_mask.sum().item()

        else:
            num_obj = 1

        # No-object loss
        noobj_loss = F.softplus(predictions[..., 4][noobj_mask]).sum()  # pylint: disable=not-callable

        # Normalize by number of positive samples (average per object)
        num_obj = max(1, num_obj)
        coord_loss = self.coord_coeff * coord_loss / num_obj
        obj_loss = self.obj_coeff * obj_loss / num_obj
        cls_loss = self.cls_coeff * cls_loss / num_obj

        # Normalize background loss by batch size (average per image)
        batch_size = predictions.shape[0]
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
        (anchors, grid, stride) = self.anchor_generator(images, neck_features)

        losses: dict[str, torch.Tensor] = {}
        detections: list[dict[str, torch.Tensor]] = []
        if self.training is True:
            assert targets is not None, "targets should not be none when in training mode"
            losses = self.compute_loss(predictions, targets, anchors, stride)

        else:
            decoded_predictions = decode_predictions(predictions, anchors, grid, stride, self.num_classes)
            detections = self.postprocess_detections(decoded_predictions, images.image_sizes)

        return (detections, losses)
